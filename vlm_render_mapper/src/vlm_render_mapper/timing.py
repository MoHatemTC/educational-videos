"""
Timing computation and cursor-path easing.

Responsibilities
----------------
* Compute per-frame durations (integer ms) from raw session timestamps.
* Apply a global speed multiplier / min-duration floor.
* Cap the last event's duration with the session's maximum frame duration
  (instead of action-specific dwell heuristics).
* Generate interpolated CursorKeyframe sequences using the smoothstep cubic
  formula  ``t² * (3 − 2t)``  with segment-relative integer timestamps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from vlm_render_mapper.schema import (
    ActionType,
    CursorKeyframe,
    EasingType,
)
from vlm_render_mapper.parser import SessionEvent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TimingConfig:
    """Knobs that control timing behaviour."""

    # Minimum duration a single frame can have (ms).
    min_frame_duration_ms: float = 200.0

    # Maximum gap before we clamp it (ms).
    max_gap_duration_ms: float = 3_000.0

    # Global playback speed multiplier (>1 = faster).
    speed_multiplier: float = 1.0

    # Easing for cursor movement (default: smoothstep cubic).
    cursor_easing: EasingType = EasingType.SMOOTHSTEP

    # Number of interpolated steps between cursor keyframes.
    cursor_interpolation_steps: int = 10

    # Frames per second for the render.
    frame_rate: int = 30

    # ----------------------------------------------------------------
    # Legacy dwell fields kept for backwards-compatible config objects.
    # They are NOT used for last-frame capping (see compute_frame_timings).
    # ----------------------------------------------------------------
    click_dwell_ms: float = 400.0
    type_dwell_ms: float = 600.0
    navigate_dwell_ms: float = 1_500.0
    scroll_duration_ms: float = 500.0


# ---------------------------------------------------------------------------
# Frame timing
# ---------------------------------------------------------------------------


@dataclass
class FrameTiming:
    frame_index: int
    timestamp_ms: int  # integer milliseconds from session start
    duration_ms: int  # integer display duration in ms


def _to_int_ms(value_ms: float) -> int:
    """Round a float millisecond value to the nearest integer."""
    return int(math.floor(value_ms + 0.5))


def compute_frame_timings(
    events: list[SessionEvent],
    config: Optional[TimingConfig] = None,
) -> list[FrameTiming]:
    """Derive per-frame integer timestamps and durations.

    Algorithm
    ---------
    1. Normalise all timestamps to ms relative to t=0.
    2. Each frame's duration is the gap to the next event, clamped to
       ``max_gap_duration_ms``, then divided by ``speed_multiplier``,
       then floored to ``min_frame_duration_ms``.
    3. For action-type overrides (navigate, page_load, scroll) we still
       apply an override minimum, but only for non-last events.
    4. **Last event**: its duration is set to ``max(all other durations)``,
       i.e. it is capped by the longest frame already computed.  This is
       cleaner than bespoke dwell heuristics.
    5. All values are rounded to integers.
    """
    cfg = config or TimingConfig()

    if not events:
        return []

    # Normalise to float ms first (precision needed for speed division).
    t0 = events[0]["timestamp"]
    raw_ms_float = [(ev["timestamp"] - t0) * 1000.0 for ev in events]

    timings: list[FrameTiming] = []
    for i, (ev, abs_ms) in enumerate(zip(events, raw_ms_float)):
        ts_int = _to_int_ms(abs_ms)

        if i + 1 < len(raw_ms_float):
            raw_gap = raw_ms_float[i + 1] - abs_ms
            clamped = min(raw_gap, cfg.max_gap_duration_ms)
            adjusted = clamped / cfg.speed_multiplier
            duration = max(adjusted, cfg.min_frame_duration_ms)

            # Minimum-duration override for slow action types.
            override = _action_duration_override(ev["action"], cfg)
            if override is not None:
                duration = max(override / cfg.speed_multiplier, cfg.min_frame_duration_ms)

            timings.append(
                FrameTiming(
                    frame_index=i,
                    timestamp_ms=ts_int,
                    duration_ms=_to_int_ms(duration),
                )
            )
        else:
            # Last event: placeholder — will be fixed in pass 2.
            timings.append(
                FrameTiming(
                    frame_index=i,
                    timestamp_ms=ts_int,
                    duration_ms=0,  # filled below
                )
            )

    # Pass 2: cap the last frame's duration at the session maximum.
    if len(timings) == 1:
        # Single-event session: use the min_frame_duration floor.
        timings[0] = FrameTiming(
            frame_index=0,
            timestamp_ms=timings[0].timestamp_ms,
            duration_ms=_to_int_ms(cfg.min_frame_duration_ms),
        )
    else:
        max_dur = max(t.duration_ms for t in timings[:-1])
        last = timings[-1]
        timings[-1] = FrameTiming(
            frame_index=last.frame_index,
            timestamp_ms=last.timestamp_ms,
            duration_ms=max_dur,
        )

    return timings


def _action_duration_override(action: str, cfg: TimingConfig) -> float | None:
    """Some action types enforce a minimum meaningful duration (ms)."""
    table = {
        ActionType.NAVIGATE.value: cfg.navigate_dwell_ms,
        ActionType.PAGE_LOAD.value: cfg.navigate_dwell_ms,
        ActionType.SCROLL.value: cfg.scroll_duration_ms,
    }
    return table.get(action)


# ---------------------------------------------------------------------------
# Easing functions  (t in [0, 1] → value in [0, 1])
# ---------------------------------------------------------------------------


def ease_linear(t: float) -> float:
    return t


def ease_in(t: float) -> float:
    return t * t


def ease_out(t: float) -> float:
    return t * (2 - t)


def ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return -1 + (4 - 2 * t) * t


def ease_smoothstep(t: float) -> float:
    """Cubic smoothstep: t² * (3 − 2t).

    This is the exact formula required by the schema spec for cursor easing.
    It produces the same boundary values as ease_in_out but via the simpler
    Hermite interpolation polynomial.
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_spring(t: float, omega: float = 12.0, zeta: float = 0.5) -> float:
    """Under-damped spring approximation."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    wd = omega * math.sqrt(max(1 - zeta * zeta, 1e-9))
    decay = math.exp(-zeta * omega * t)
    return 1 - decay * (math.cos(wd * t) + (zeta * omega / wd) * math.sin(wd * t))


_EASING_FN = {
    EasingType.LINEAR: ease_linear,
    EasingType.EASE_IN: ease_in,
    EasingType.EASE_OUT: ease_out,
    EasingType.EASE_IN_OUT: ease_in_out,
    EasingType.SMOOTHSTEP: ease_smoothstep,
    EasingType.SPRING: ease_spring,
}


def apply_easing(t: float, easing: EasingType) -> float:
    t = max(0.0, min(1.0, t))
    fn = _EASING_FN.get(easing, ease_smoothstep)
    return fn(t)


def interpolate_value(start: float, end: float, t: float, easing: EasingType) -> float:
    et = apply_easing(t, easing)
    return start + (end - start) * et


# ---------------------------------------------------------------------------
# Cursor path generation
# ---------------------------------------------------------------------------


def build_cursor_path(
    events: list[SessionEvent],
    timings: list[FrameTiming],
    config: Optional[TimingConfig] = None,
) -> list[CursorKeyframe]:
    """Generate a smooth cursor keyframe list from session events.

    Timestamps are **segment-relative integers**: each keyframe's
    ``timestamp_ms`` is the integer offset from the *start of its segment*
    (i.e. from the anchor frame's ``timestamp_ms``), not the session-absolute
    time.  The first anchor always has ``timestamp_ms = 0``.

    Easing uses the smoothstep cubic  ``t² * (3 − 2t)``  by default.
    """
    cfg = config or TimingConfig()

    # Collect anchor points (events with known cursor coordinates).
    anchors: list[tuple[int, float, float]] = []  # (segment-rel ts_ms int, x, y)
    for ev, timing in zip(events, timings):
        if ev.get("x") is not None and ev.get("y") is not None:
            anchors.append((timing.timestamp_ms, float(ev["x"]), float(ev["y"])))

    if not anchors:
        return []

    keyframes: list[CursorKeyframe] = []

    for i, (abs_ts, x, y) in enumerate(anchors):
        if i == 0:
            # First anchor: segment-relative ts = 0.
            keyframes.append(
                CursorKeyframe(
                    timestamp_ms=0,
                    x=x,
                    y=y,
                    easing=cfg.cursor_easing,
                    duration_ms=0,
                )
            )
            continue

        prev_abs_ts, prev_x, prev_y = anchors[i - 1]
        travel_ms_float = float(abs_ts - prev_abs_ts)
        travel_ms_int = _to_int_ms(travel_ms_float)

        if travel_ms_int <= 0 or cfg.cursor_interpolation_steps <= 1:
            # No interpolation — emit the destination keyframe only.
            # Segment-relative timestamp = cumulative travel from anchor 0.
            seg_ts = _to_int_ms(float(abs_ts - anchors[0][0]))
            keyframes.append(
                CursorKeyframe(
                    timestamp_ms=seg_ts,
                    x=x,
                    y=y,
                    easing=cfg.cursor_easing,
                    duration_ms=travel_ms_int,
                )
            )
            continue

        steps = cfg.cursor_interpolation_steps
        step_dur = _to_int_ms(travel_ms_float / steps)

        for step in range(1, steps + 1):
            t = step / steps
            # Segment-relative timestamp (integer).
            seg_ts = _to_int_ms(float(prev_abs_ts - anchors[0][0]) + travel_ms_float * t)
            kf_x = interpolate_value(prev_x, x, t, cfg.cursor_easing)
            kf_y = interpolate_value(prev_y, y, t, cfg.cursor_easing)
            keyframes.append(
                CursorKeyframe(
                    timestamp_ms=seg_ts,
                    x=round(kf_x, 2),
                    y=round(kf_y, 2),
                    easing=cfg.cursor_easing,
                    duration_ms=step_dur,
                )
            )

    # De-duplicate exact timestamps (keep last occurrence).
    seen: dict[int, CursorKeyframe] = {}
    for kf in keyframes:
        seen[kf.timestamp_ms] = kf
    return sorted(seen.values(), key=lambda k: k.timestamp_ms)


# ---------------------------------------------------------------------------
# Utility: total session duration (integer ms)
# ---------------------------------------------------------------------------


def total_duration_ms(timings: list[FrameTiming]) -> int:
    if not timings:
        return 0
    last = timings[-1]
    return last.timestamp_ms + last.duration_ms
