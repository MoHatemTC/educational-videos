"""
Action-to-Frame mapper.

Takes normalised SessionEvents + FrameTimings and produces the full
RenderPlan (frames, highlights, zooms, captions, transitions, timeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from vlm_render_mapper.parser import SessionEvent
from vlm_render_mapper.timing import (
    FrameTiming,
    TimingConfig,
    build_cursor_path,
    compute_frame_timings,
    total_duration_ms,
)
from vlm_render_mapper.schema import (
    Action,
    ActionType,
    Caption,
    CaptionPosition,
    CaptionStyle,
    CursorState,
    CursorStyle,
    EasingType,
    FrameDescriptor,
    HighlightRegion,
    Modifier,
    Point,
    RenderMetadata,
    RenderPlan,
    RenderTarget,
    Resolution,
    Timeline,
    TimelineSegment,
    Transition,
    TransitionType,
    Viewport,
    ZoomRegion,
    session_id_from_events,
)


# ---------------------------------------------------------------------------
# Mapper configuration
# ---------------------------------------------------------------------------


@dataclass
class MapperConfig:
    session_id: str = ""  # empty = derive deterministically from log content
    frame_rate: int = 30
    viewport_width: int = 1280
    viewport_height: int = 720
    device_pixel_ratio: float = 1.0
    render_target: RenderTarget = RenderTarget.FFMPEG
    source_session_file: Optional[str] = None

    # Highlight settings
    click_highlight_color: str = "#FF4444"
    click_highlight_opacity: float = 0.35
    hover_highlight_color: str = "#4488FF"
    hover_highlight_opacity: float = 0.25
    highlight_size: float = 60.0

    # Zoom settings
    click_zoom_scale: float = 1.5
    type_zoom_scale: float = 1.3
    zoom_transition_ms: int = 300
    zoom_padding: float = 80.0

    # Caption settings
    generate_captions: bool = True
    caption_position: CaptionPosition = CaptionPosition.BOTTOM
    caption_offset_ms: int = 0

    # Transition settings
    default_transition: TransitionType = TransitionType.CUT
    navigate_transition: TransitionType = TransitionType.FADE
    transition_duration_ms: int = 300

    # Timing
    timing: TimingConfig = field(default_factory=TimingConfig)


# ---------------------------------------------------------------------------
# Caption text generation
# ---------------------------------------------------------------------------

_CAPTION_TEMPLATES: dict[str, str] = {
    ActionType.CLICK.value: "Click on {target}",
    ActionType.DOUBLE_CLICK.value: "Double-click on {target}",
    ActionType.RIGHT_CLICK.value: "Right-click on {target}",
    ActionType.HOVER.value: "Hover over {target}",
    ActionType.TYPE.value: 'Type "{value}"',
    ActionType.KEY_PRESS.value: "Press {value}",
    ActionType.NAVIGATE.value: "Navigate to {value}",
    ActionType.PAGE_LOAD.value: "Page loaded: {value}",
    ActionType.SCROLL.value: "Scroll {scroll_dir}",
    ActionType.DRAG.value: "Drag element",
    ActionType.SCREENSHOT.value: "Screenshot captured",
    ActionType.WAIT.value: "Pause",
    ActionType.FOCUS.value: "Focus on {target}",
    ActionType.BLUR.value: "Blur {target}",
}


def _make_caption_text(ev: SessionEvent) -> str:
    tmpl = _CAPTION_TEMPLATES.get(ev["action"], "Action: {action}")
    target = ev.get("target") or "element"
    value = ev.get("value") or ""
    scroll_dy = ev.get("scroll_delta_y") or 0
    scroll_dir = "down" if (scroll_dy and scroll_dy > 0) else "up"
    return tmpl.format(
        target=target,
        value=value,
        action=ev["action"],
        scroll_dir=scroll_dir,
    )


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------

_ACTION_CURSOR_STYLE: dict[str, CursorStyle] = {
    ActionType.CLICK.value: CursorStyle.POINTER,
    ActionType.DOUBLE_CLICK.value: CursorStyle.POINTER,
    ActionType.RIGHT_CLICK.value: CursorStyle.POINTER,
    ActionType.HOVER.value: CursorStyle.POINTER,
    ActionType.TYPE.value: CursorStyle.TEXT,
    ActionType.DRAG.value: CursorStyle.GRABBING,
    ActionType.FOCUS.value: CursorStyle.TEXT,
}


def _cursor_state(ev: SessionEvent) -> Optional[CursorState]:
    if ev.get("x") is None or ev.get("y") is None:
        return None
    return CursorState(
        x=float(ev["x"]),
        y=float(ev["y"]),
        visible=True,
        style=_ACTION_CURSOR_STYLE.get(ev["action"], CursorStyle.DEFAULT),
    )


# ---------------------------------------------------------------------------
# Highlight region builders
# ---------------------------------------------------------------------------


def _build_click_highlight(
    ev: SessionEvent, region_id: str, cfg: MapperConfig
) -> Optional[HighlightRegion]:
    if ev.get("x") is None or ev.get("y") is None:
        return None
    half = cfg.highlight_size / 2
    action = ev["action"]
    if action in (ActionType.HOVER.value,):
        color = cfg.hover_highlight_color
        opacity = cfg.hover_highlight_opacity
    else:
        color = cfg.click_highlight_color
        opacity = cfg.click_highlight_opacity

    return HighlightRegion(
        region_id=region_id,
        x=float(ev["x"]) - half,
        y=float(ev["y"]) - half,
        width=cfg.highlight_size,
        height=cfg.highlight_size,
        color=color,
        opacity=opacity,
        border_radius=cfg.highlight_size / 2,
        label=_make_caption_text(ev) if action == ActionType.CLICK.value else None,
    )


_HIGHLIGHT_ACTIONS = frozenset(
    {
        ActionType.CLICK.value,
        ActionType.DOUBLE_CLICK.value,
        ActionType.RIGHT_CLICK.value,
        ActionType.HOVER.value,
        ActionType.FOCUS.value,
    }
)


# ---------------------------------------------------------------------------
# Zoom region builders
# ---------------------------------------------------------------------------


def _build_zoom_region(
    ev: SessionEvent, cfg: MapperConfig, vp_w: int, vp_h: int
) -> Optional[ZoomRegion]:
    action = ev["action"]
    if action not in (
        ActionType.CLICK.value,
        ActionType.DOUBLE_CLICK.value,
        ActionType.TYPE.value,
        ActionType.FOCUS.value,
    ):
        return None
    if ev.get("x") is None or ev.get("y") is None:
        return None

    scale = (
        cfg.click_zoom_scale
        if action in (ActionType.CLICK.value, ActionType.DOUBLE_CLICK.value)
        else cfg.type_zoom_scale
    )
    pad = cfg.zoom_padding
    cx, cy = float(ev["x"]), float(ev["y"])

    w = vp_w / scale
    h = vp_h / scale
    x = max(0.0, min(cx - w / 2, vp_w - w))
    y = max(0.0, min(cy - h / 2, vp_h - h))

    x = max(0.0, x - pad)
    y = max(0.0, y - pad)
    w = min(w + 2 * pad, float(vp_w))
    h = min(h + 2 * pad, float(vp_h))

    return ZoomRegion(
        x=round(x, 2),
        y=round(y, 2),
        width=round(w, 2),
        height=round(h, 2),
        scale=scale,
        easing=EasingType.EASE_IN_OUT,
        transition_duration_ms=cfg.zoom_transition_ms,
    )


# ---------------------------------------------------------------------------
# Transition builder
# ---------------------------------------------------------------------------


def _build_transition(
    ev: SessionEvent,
    from_frame_id: str,
    to_frame_id: str,
    timing: FrameTiming,
    cfg: MapperConfig,
) -> Optional[Transition]:
    action = ev["action"]
    if action in (ActionType.NAVIGATE.value, ActionType.PAGE_LOAD.value):
        tr_type = cfg.navigate_transition
        dur = cfg.transition_duration_ms
    elif action == ActionType.SCREENSHOT.value:
        tr_type = TransitionType.CUT
        dur = 0
    else:
        return None

    start = timing.timestamp_ms + timing.duration_ms - dur
    return Transition(
        transition_id=f"tr_{from_frame_id}_{to_frame_id}",
        type=tr_type,
        start_ms=max(0, start),
        duration_ms=dur,
        from_frame_id=from_frame_id,
        to_frame_id=to_frame_id,
    )


# ---------------------------------------------------------------------------
# Action → schema.Action
# ---------------------------------------------------------------------------


def _build_action(ev: SessionEvent) -> Action:
    coords = None
    if ev.get("x") is not None and ev.get("y") is not None:
        coords = Point(x=float(ev["x"]), y=float(ev["y"]))

    drag_start = None
    drag_end = None
    if ev.get("drag_start"):
        ds = ev["drag_start"]
        drag_start = Point(x=float(ds.get("x", 0)), y=float(ds.get("y", 0)))
    if ev.get("drag_end"):
        de = ev["drag_end"]
        drag_end = Point(x=float(de.get("x", 0)), y=float(de.get("y", 0)))

    scroll_delta = None
    if ev.get("scroll_delta_x") is not None or ev.get("scroll_delta_y") is not None:
        scroll_delta = Point(
            x=float(ev.get("scroll_delta_x") or 0),
            y=float(ev.get("scroll_delta_y") or 0),
        )

    modifiers: list[Modifier] = []
    raw_mods = ev.get("meta", {}).get("modifiers") or []
    for m in raw_mods:
        try:
            modifiers.append(Modifier(str(m).lower()))
        except ValueError:
            pass

    return Action(
        type=ActionType(ev["action"]),
        target=ev.get("target"),
        value=ev.get("value"),
        coordinates=coords,
        modifiers=modifiers,
        scroll_delta=scroll_delta,
        drag_start=drag_start,
        drag_end=drag_end,
    )


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------


class RenderMapper:
    """Map a list of SessionEvents to a complete RenderPlan."""

    def __init__(self, config: Optional[MapperConfig] = None) -> None:
        self.cfg = config or MapperConfig()

    def map(self, events: list[SessionEvent]) -> RenderPlan:
        if not events:
            raise ValueError("Cannot build a RenderPlan from an empty event list")

        cfg = self.cfg
        timing_cfg = cfg.timing

        # 0. Derive session_id deterministically from log content when not overridden.
        if not cfg.session_id:
            cfg.session_id = session_id_from_events([ev["_raw"] for ev in events])

        # 1. Compute per-frame timings (integer ms).
        timings = compute_frame_timings(events, timing_cfg)

        # Intermediate assertion: parser must never return an empty list without
        # raising SessionParseError — guard against silent data loss.
        assert len(timings) == len(events), (
            f"Timing count mismatch: expected {len(events)} timings, got {len(timings)}. "
            "Parser may have silently dropped events."
        )

        # Intermediate assertion: timestamps must be monotonically non-decreasing.
        for _j in range(1, len(timings)):
            assert timings[_j].timestamp_ms >= timings[_j - 1].timestamp_ms, (
                f"Timing monotonicity violated at index {_j}: "
                f"timestamp_ms {timings[_j].timestamp_ms} < {timings[_j - 1].timestamp_ms}"
            )

        # 2. Build cursor path (segment-relative integer timestamps).
        cursor_path = build_cursor_path(events, timings, timing_cfg)

        # 3. Build frames.
        frames: list[FrameDescriptor] = []
        captions: list[Caption] = []
        transitions: list[Transition] = []

        # Frame counter tracks the output frame index (≥ event index because
        # click-like actions expand into two frames).
        frame_seq: int = 0

        for i, (ev, timing) in enumerate(zip(events, timings)):
            is_click_action = ev["action"] in (
                ActionType.CLICK.value,
                ActionType.DOUBLE_CLICK.value,
                ActionType.RIGHT_CLICK.value,
            )

            # ---- frame A: cursor-move (or the sole frame for non-click actions) ----
            frame_id_a = f"frame_{frame_seq:06d}"

            if is_click_action:
                # Split duration: cursor-move gets first half, highlight gets second.
                half_dur = max(timing.duration_ms // 2, 1)
                dur_a = half_dur
                dur_b = timing.duration_ms - half_dur
                ts_a = timing.timestamp_ms
                ts_b = timing.timestamp_ms + dur_a
            else:
                dur_a = timing.duration_ms
                ts_a = timing.timestamp_ms

            # Caption applies to the action (frame A for non-click, frame B for click)
            cap_ids_a: list[str] = []
            cap_ids_b: list[str] = []
            if cfg.generate_captions:
                cap_id = f"cap_frame_{frame_seq:06d}"
                cap_text = _make_caption_text(ev)
                if is_click_action:
                    # Caption rides on the highlight frame (B)
                    cap_start = ts_b + cfg.caption_offset_ms
                    cap_end = cap_start + dur_b
                else:
                    cap_start = ts_a + cfg.caption_offset_ms
                    cap_end = cap_start + dur_a
                if cap_end > cap_start:
                    caption = Caption(
                        caption_id=cap_id,
                        text=cap_text,
                        start_ms=cap_start,
                        end_ms=cap_end,
                        position=cfg.caption_position,
                        style=CaptionStyle(),
                    )
                    captions.append(caption)
                    if is_click_action:
                        cap_ids_b.append(cap_id)
                    else:
                        cap_ids_a.append(cap_id)

            # Cursor-move frame: no highlight, no zoom
            frame_a = FrameDescriptor(
                frame_id=frame_id_a,
                source_action_index=i,
                start_ms=ts_a,
                duration_ms=dur_a,
                screenshot_ref=ev["screenshot"],
                action=_build_action(ev),
                cursor=_cursor_state(ev),
                highlight_regions=[],
                zoom_region=None,
                caption_ids=cap_ids_a,
            )
            frames.append(frame_a)
            frame_seq += 1

            if is_click_action:
                # ---- frame B: click-highlight ----
                frame_id_b = f"frame_{frame_seq:06d}"
                hl = _build_click_highlight(ev, f"hl_{frame_id_b}", cfg)
                zoom = _build_zoom_region(ev, cfg, cfg.viewport_width, cfg.viewport_height)
                frame_b = FrameDescriptor(
                    frame_id=frame_id_b,
                    source_action_index=i,
                    start_ms=ts_b,
                    duration_ms=dur_b,
                    screenshot_ref=ev["screenshot"],
                    action=_build_action(ev),
                    cursor=_cursor_state(ev),
                    highlight_regions=[hl] if hl else [],
                    zoom_region=zoom,
                    caption_ids=cap_ids_b,
                )
                frames.append(frame_b)
                frame_seq += 1
            else:
                # Non-click: highlights + zoom go on the single frame
                highlights: list[HighlightRegion] = []
                if ev["action"] in _HIGHLIGHT_ACTIONS:
                    hl = _build_click_highlight(ev, f"hl_{frame_id_a}", cfg)
                    if hl:
                        highlights.append(hl)
                zoom = _build_zoom_region(ev, cfg, cfg.viewport_width, cfg.viewport_height)
                # Mutate frame_a in-place to attach visuals
                frame_a.highlight_regions = highlights
                frame_a.zoom_region = zoom

            # Transition (based on the last output frame for this event)
            last_frame_id = f"frame_{frame_seq - 1:06d}"
            next_first_frame_id = f"frame_{frame_seq:06d}"
            if i + 1 < len(events):
                tr = _build_transition(
                    ev,
                    from_frame_id=last_frame_id,
                    to_frame_id=next_first_frame_id,
                    timing=timing,
                    cfg=cfg,
                )
                if tr:
                    transitions.append(tr)

        # 4. Build timeline.
        timeline = self._build_timeline(frames)

        # 5. Compute total duration (integer ms).
        total_ms = total_duration_ms(timings)

        # 6. Assemble render plan — using `recorded_at` per schema contract.
        plan = RenderPlan(
            version="1.0.0",
            metadata=RenderMetadata(
                session_id=cfg.session_id,
                recorded_at=datetime.now(timezone.utc),
                total_duration_ms=total_ms,
                frame_rate=cfg.frame_rate,
                resolution=Resolution(
                    width=cfg.viewport_width,
                    height=cfg.viewport_height,
                ),
                source_session_file=cfg.source_session_file,
                render_target=cfg.render_target,
            ),
            viewport=Viewport(
                width=cfg.viewport_width,
                height=cfg.viewport_height,
                device_pixel_ratio=cfg.device_pixel_ratio,
            ),
            frames=frames,
            timeline=timeline,
            cursor_path=cursor_path,
            captions=captions,
            transitions=transitions,
        )
        return plan

    @staticmethod
    def _build_timeline(frames: list[FrameDescriptor]) -> Timeline:
        """Group frames into a single contiguous timeline segment."""
        if not frames:
            return Timeline(segments=[], total_duration_ms=0)

        total = frames[-1].timestamp_ms + frames[-1].duration_ms
        segment = TimelineSegment(
            segment_id="seg_000",
            start_ms=0,
            end_ms=total,
            frame_ids=[f.frame_id for f in frames],
            label="Full session",
            speed_multiplier=1.0,
        )
        return Timeline(segments=[segment], total_duration_ms=total)
