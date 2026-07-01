"""Render timeline-driven code animation frames with Pillow.

Produces ``frame_NNNNN.png`` files: a titled, dark-themed editor that can type,
highlight, and scroll code according to a validated animation timeline. When no
valid timeline is available, rendering falls back to the original duration-based
character reveal so older jobs and web explainers keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from app.core.logging import logger
from app.core.schemas import Timeline

WIDTH, HEIGHT = 1280, 720
BG = (30, 30, 46)
TITLE_BG = (49, 50, 68)
FG = (205, 214, 244)
ACCENT = (137, 180, 250)
DIM = (108, 112, 134)
HIGHLIGHT_BG = (59, 62, 85)
RUN_BG = (24, 24, 37)
RUN_FG = (166, 227, 161)

_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class _RenderEvent:
    """Renderer-friendly event with audio-reconciled timing."""

    event_type: str
    start_ms: float
    end_ms: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class _FrameState:
    """Animation state for a single rendered frame."""

    reveal_chars: int
    first_line_index: int
    highlight_start_line: int | None
    highlight_end_line: int | None
    run_text: str | None


def _font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to Pillow's default."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _coerce_timeline(timeline: Timeline | dict[str, Any] | None) -> Timeline | None:
    """Validate a raw timeline payload, returning ``None`` when unusable."""
    if timeline is None:
        return None
    if isinstance(timeline, Timeline):
        return timeline
    try:
        return Timeline.model_validate(timeline)
    except ValidationError as exc:
        logger.warning("timeline_render_fallback", reason="invalid_timeline", error=str(exc))
        return None


def _reconciled_events(timeline: Timeline, duration_s: float) -> list[_RenderEvent]:
    """Scale timeline event timings to the measured narration duration."""
    target_ms = max(duration_s, 0.001) * 1000.0
    source_end_ms = max(float(event.end_ms) for event in timeline.events)
    if source_end_ms <= 0:
        return []

    scale = target_ms / source_end_ms
    events: list[_RenderEvent] = []
    for event in timeline.events:
        raw = event.model_dump()
        start_ms = min(max(float(event.start_ms) * scale, 0.0), target_ms)
        end_ms = min(max(float(event.end_ms) * scale, start_ms), target_ms)
        if end_ms <= start_ms:
            continue
        events.append(
            _RenderEvent(
                event_type=str(raw["event_type"]),
                start_ms=start_ms,
                end_ms=end_ms,
                payload=raw,
            )
        )
    return sorted(events, key=lambda item: (item.start_ms, item.end_ms))


def _timeline_reveal_chars(events: list[_RenderEvent], time_ms: float, total_chars: int) -> int:
    """Return how many code characters should be visible at ``time_ms``."""
    type_events = [event for event in events if event.event_type == "type"]
    if not type_events:
        return total_chars

    total_type_ms = sum(event.end_ms - event.start_ms for event in type_events)
    if total_type_ms <= 0:
        return total_chars

    elapsed_type_ms = 0.0
    for event in type_events:
        if time_ms >= event.end_ms:
            elapsed_type_ms += event.end_ms - event.start_ms
            continue
        if event.start_ms <= time_ms < event.end_ms:
            elapsed_type_ms += time_ms - event.start_ms
            break
        if time_ms < event.start_ms:
            break

    fraction = min(max(elapsed_type_ms / total_type_ms, 0.0), 1.0)
    return min(total_chars, int(round(total_chars * fraction)))


def _duration_reveal_chars(frame_index: int, total_frames: int, total_chars: int, type_fraction: float) -> int:
    """Return the original duration-based typing reveal amount."""
    type_frames = max(1, int(total_frames * type_fraction))
    if frame_index >= type_frames:
        return total_chars
    return int(total_chars * (frame_index / type_frames))


def _active_event(events: list[_RenderEvent], time_ms: float, event_type: str) -> _RenderEvent | None:
    """Return the last active event of ``event_type`` at ``time_ms``."""
    active = [event for event in events if event.event_type == event_type and event.start_ms <= time_ms < event.end_ms]
    if not active:
        return None
    return active[-1]


def _first_line_index_for_time(
    events: list[_RenderEvent],
    time_ms: float,
    total_lines: int,
    visible_lines: int,
) -> int:
    """Return the first zero-based line index visible at ``time_ms``."""
    max_first_line = max(0, total_lines - visible_lines)
    scroll_event = _active_event(events, time_ms, "scroll")
    if scroll_event is None:
        return 0

    target_line = int(scroll_event.payload.get("target_line", 1))
    return min(max(target_line - 1, 0), max_first_line)


def _frame_state(
    *,
    frame_index: int,
    fps: int,
    total_frames: int,
    total_chars: int,
    total_lines: int,
    visible_lines: int,
    events: list[_RenderEvent] | None,
    type_fraction: float,
) -> _FrameState:
    """Compute typing, highlight, scroll, and run state for one frame."""
    if events is None:
        return _FrameState(
            reveal_chars=_duration_reveal_chars(frame_index, total_frames, total_chars, type_fraction),
            first_line_index=0,
            highlight_start_line=None,
            highlight_end_line=None,
            run_text=None,
        )

    time_ms = (frame_index / max(fps, 1)) * 1000.0
    highlight_event = _active_event(events, time_ms, "highlight")
    run_event = _active_event(events, time_ms, "run")
    run_text: str | None = None
    if run_event is not None:
        command = str(run_event.payload.get("command", "")).strip()
        expected_output = str(run_event.payload.get("expected_output") or "").strip()
        run_text = command if not expected_output else f"{command}\n{expected_output}"

    return _FrameState(
        reveal_chars=_timeline_reveal_chars(events, time_ms, total_chars),
        first_line_index=_first_line_index_for_time(events, time_ms, total_lines, visible_lines),
        highlight_start_line=int(highlight_event.payload["start_line"]) if highlight_event is not None else None,
        highlight_end_line=int(highlight_event.payload["end_line"]) if highlight_event is not None else None,
        run_text=run_text,
    )


def _draw_run_panel(
    draw: ImageDraw.ImageDraw, run_text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont
) -> None:
    """Draw a compact command/output panel at the bottom of the frame."""
    panel_top = HEIGHT - 135
    draw.rectangle([30, panel_top, WIDTH - 30, HEIGHT - 30], fill=RUN_BG, outline=DIM)
    y = panel_top + 16
    for line in run_text.splitlines()[:3]:
        draw.text((55, y), line[:110], font=font, fill=RUN_FG)
        y += 30


def render_frames(
    code: str,
    title: str,
    frames_dir: str | Path,
    fps: int = 10,
    duration_s: float = 10.0,
    type_fraction: float = 0.85,
    timeline: Timeline | dict[str, Any] | None = None,
) -> int:
    """Render the code animation to ``frames_dir`` and return the frame count.

    Args:
        code: Source code to type.
        title: Title shown in the header bar.
        frames_dir: Output directory for PNG frames.
        fps: Frames per second.
        duration_s: Total animation duration after measuring narration audio.
        type_fraction: Fallback fraction spent typing when no valid timeline exists.
        timeline: Optional structured timeline whose events drive type, highlight,
            scroll, and run effects.

    Returns:
        Number of frames written.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    code = code.rstrip() or "# (no code)"
    lines = code.split("\n")
    total_chars = len(code)
    total_frames = max(1, int(round(duration_s * fps)))

    code_font = _font(_MONO, 26)
    title_font = _font(_BOLD, 34)
    ln_font = _font(_MONO, 20)
    run_font = _font(_MONO, 22)
    line_height = 36
    x_code = 90
    y_top = 110
    visible_lines = max(1, (HEIGHT - y_top - 40) // line_height)

    parsed_timeline = _coerce_timeline(timeline)
    events = _reconciled_events(parsed_timeline, duration_s) if parsed_timeline is not None else None
    timeline_used = bool(events)

    for i in range(total_frames):
        state = _frame_state(
            frame_index=i,
            fps=fps,
            total_frames=total_frames,
            total_chars=total_chars,
            total_lines=len(lines),
            visible_lines=visible_lines,
            events=events,
            type_fraction=type_fraction,
        )
        shown_lines = code[: state.reveal_chars].split("\n")

        img = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, WIDTH, 70], fill=TITLE_BG)
        draw.text((30, 16), title[:60], font=title_font, fill=ACCENT)

        y = y_top
        for idx in range(state.first_line_index, len(lines)):
            line_number = idx + 1
            if state.highlight_start_line is not None and state.highlight_end_line is not None:
                if state.highlight_start_line <= line_number <= state.highlight_end_line:
                    draw.rectangle([80, y - 3, WIDTH - 30, y + 33], fill=HIGHLIGHT_BG)

            draw.text((30, y), str(line_number).rjust(2), font=ln_font, fill=DIM)
            if idx < len(shown_lines):
                text = shown_lines[idx]
                draw.text((x_code, y), text, font=code_font, fill=FG)
                is_cursor_line = idx == len(shown_lines) - 1 and state.reveal_chars < total_chars
                if is_cursor_line:
                    width = draw.textlength(text, font=code_font)
                    draw.rectangle([x_code + width + 2, y + 2, x_code + width + 13, y + 30], fill=ACCENT)
            y += line_height
            if y > HEIGHT - 40:
                break

        if state.run_text:
            _draw_run_panel(draw, state.run_text, run_font)

        img.save(frames_dir / f"frame_{i + 1:05d}.png")

    logger.info(
        "frames_rendered",
        count=total_frames,
        fps=fps,
        duration_s=round(duration_s, 2),
        timeline_used=timeline_used,
    )
    return total_frames
