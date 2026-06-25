"""
CLI entry point for VLM Render Mapper.

Usage:
    vlm-render-mapper --session session.json --output render_plan.json
    vlm-render-mapper --session session.jsonl --fps 60 --speed 1.5 --target remotion
    vlm-render-mapper --validate render_plan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

from vlm_render_mapper.parser import parse_session_file, SessionParseError
from vlm_render_mapper.mapper import MapperConfig, RenderMapper
from vlm_render_mapper.schema import (
    RenderPlan,
    RenderTarget,
    CaptionPosition,
    validate_against_json_schema,
)
from vlm_render_mapper.timing import TimingConfig


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vlm-render-mapper",
        description="Translate a VLM browser-action session into a programmatic render plan JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--session",
        "-s",
        metavar="FILE",
        help="Input session log (JSON array or JSON Lines).",
    )
    mode.add_argument(
        "--validate",
        "-v",
        metavar="PLAN_FILE",
        help="Validate an existing render plan JSON against the schema.",
    )

    p.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        default="render_plan.json",
        help="Output render plan JSON file path.",
    )

    # ---- Render settings ----
    p.add_argument("--fps", type=int, default=30, metavar="N", help="Output frame rate.")
    p.add_argument(
        "--width", type=int, default=1280, metavar="PX", help="Viewport width in pixels."
    )
    p.add_argument(
        "--height", type=int, default=720, metavar="PX", help="Viewport height in pixels."
    )
    p.add_argument(
        "--target",
        choices=["ffmpeg", "remotion", "both"],
        default="ffmpeg",
        help="Render target.",
    )

    # ---- Timing ----
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="X",
        help="Playback speed multiplier (>1 = faster).",
    )
    p.add_argument(
        "--min-frame-ms",
        type=float,
        default=200.0,
        metavar="MS",
        help="Minimum per-frame duration in ms.",
    )
    p.add_argument(
        "--max-gap-ms",
        type=float,
        default=3000.0,
        metavar="MS",
        help="Maximum gap between events before clamping.",
    )

    # ---- Visual options ----
    p.add_argument(
        "--no-captions", action="store_true", help="Disable automatic caption generation."
    )
    p.add_argument(
        "--caption-position",
        choices=["top", "bottom", "center", "top_left", "top_right", "bottom_left", "bottom_right"],
        default="bottom",
        help="Caption position on screen.",
    )
    p.add_argument(
        "--no-zoom", action="store_true", help="Disable zoom regions on click/type actions."
    )
    p.add_argument(
        "--no-highlight",
        action="store_true",
        help="Disable highlight regions on click/hover actions.",
    )
    p.add_argument(
        "--click-zoom",
        type=float,
        default=1.5,
        metavar="SCALE",
        help="Zoom scale factor for click events.",
    )

    # ---- Session metadata ----
    p.add_argument("--session-id", metavar="ID", help="Override session ID in metadata.")

    p.add_argument("--pretty", action="store_true", help="Pretty-print output JSON (indent=2).")

    p.add_argument(
        "--skip-schema-validation",
        action="store_true",
        help="Skip jsonschema validation before writing output.",
    )

    return p


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_validate(plan_path: str) -> int:
    """Validate a render plan JSON file against both Pydantic and JSON Schema."""
    path = Path(plan_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 1
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Pydantic structural validation
        RenderPlan.model_validate(raw)
        # JSON Schema contract validation
        validate_against_json_schema(raw)
        print(f"[OK] {path} is a valid RenderPlan.")
        return 0
    except jsonschema.ValidationError as exc:
        print(f"[ERROR] JSON Schema validation failed: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Validation failed: {exc}", file=sys.stderr)
        return 1


def cmd_map(args: argparse.Namespace) -> int:
    """Parse session + map → validate → write render plan."""

    # 1. Parse session
    try:
        events = parse_session_file(args.session)
    except SessionParseError as exc:
        print(f"[ERROR] Session parse failed: {exc}", file=sys.stderr)
        return 1

    # Intermediate assertion: a successful parse must yield at least one event.
    assert len(events) > 0, (
        "parse_session_file returned an empty list without raising SessionParseError"
    )

    print(f"[INFO] Parsed {len(events)} events from {args.session}")

    # 2. Build config
    timing_cfg = TimingConfig(
        speed_multiplier=args.speed,
        min_frame_duration_ms=args.min_frame_ms,
        max_gap_duration_ms=args.max_gap_ms,
        frame_rate=args.fps,
    )

    mapper_cfg = MapperConfig(
        frame_rate=args.fps,
        viewport_width=args.width,
        viewport_height=args.height,
        render_target=RenderTarget(args.target),
        source_session_file=str(Path(args.session).resolve()),
        generate_captions=not args.no_captions,
        caption_position=CaptionPosition(args.caption_position),
        click_zoom_scale=args.click_zoom,
        timing=timing_cfg,
    )
    if args.session_id:
        mapper_cfg.session_id = args.session_id

    # 3. Map
    try:
        mapper = RenderMapper(mapper_cfg)
        plan = mapper.map(events)
    except Exception as exc:
        print(f"[ERROR] Mapping failed: {exc}", file=sys.stderr)
        return 1

    # 4. Apply feature flags
    if args.no_zoom:
        for frame in plan.frames:
            frame.zoom_region = None
    if args.no_highlight:
        for frame in plan.frames:
            frame.highlight_regions = []

    # 5. Serialise to dict via model_dump_json() so Pydantic converts
    #    datetime → ISO-8601 string before validation and writing.
    indent = 2 if args.pretty else None
    plan_dict = json.loads(plan.model_dump_json())

    # 6. Validate against JSON Schema before saving
    if not args.skip_schema_validation:
        try:
            validate_against_json_schema(plan_dict)
            print("[INFO] JSON Schema validation passed.")
        except jsonschema.ValidationError as exc:
            print(
                f"[ERROR] Output failed JSON Schema validation: {exc.message}",
                file=sys.stderr,
            )
            return 1
        except FileNotFoundError as exc:
            print(
                f"[WARN] Could not locate render_plan_schema.json — skipping: {exc}",
                file=sys.stderr,
            )

    # 7. Write to disk
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(plan_dict, indent=indent),
        encoding="utf-8",
    )

    total_sec = plan.metadata.total_duration_ms / 1000.0
    print(
        f"[OK] Render plan written → {out_path}\n"
        f"     Frames    : {len(plan.frames)}\n"
        f"     Duration  : {total_sec:.2f}s\n"
        f"     Captions  : {len(plan.captions)}\n"
        f"     Cursor KFs: {len(plan.cursor_path)}\n"
        f"     Transitions: {len(plan.transitions)}"
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.validate:
        sys.exit(cmd_validate(args.validate))
    else:
        sys.exit(cmd_map(args))


if __name__ == "__main__":
    main()
