"""Tests for vlm_render_mapper.timing"""

from __future__ import annotations

import pytest

from vlm_render_mapper.timing import (
    TimingConfig,
    FrameTiming,
    compute_frame_timings,
    build_cursor_path,
    total_duration_ms,
    apply_easing,
    interpolate_value,
    ease_linear,
    ease_in,
    ease_out,
    ease_in_out,
    ease_smoothstep,
    ease_spring,
    _to_int_ms,
)
from vlm_render_mapper.schema import EasingType
from vlm_render_mapper.parser import parse_session_text
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_events(timestamps_and_actions: list[tuple[float, str, dict]]):
    raw = [
        {
            "timestamp": ts,
            "action": action,
            "screenshot": f"frame_{i:06d}.png",
            **extras,
        }
        for i, (ts, action, extras) in enumerate(timestamps_and_actions)
    ]
    return parse_session_text(json.dumps(raw))


DEFAULT_CFG = TimingConfig(min_frame_duration_ms=100.0, speed_multiplier=1.0)


# ---------------------------------------------------------------------------
# _to_int_ms
# ---------------------------------------------------------------------------


class TestToIntMs:
    def test_rounds_half_up(self):
        assert _to_int_ms(100.5) == 101
        assert _to_int_ms(200.4) == 200

    def test_exact_int(self):
        assert _to_int_ms(300.0) == 300

    def test_returns_int_type(self):
        result = _to_int_ms(250.7)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Easing functions
# ---------------------------------------------------------------------------


class TestEasingFunctions:
    @pytest.mark.parametrize(
        "fn", [ease_linear, ease_in, ease_out, ease_in_out, ease_smoothstep, ease_spring]
    )
    def test_boundary_zero(self, fn):
        assert fn(0.0) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        "fn", [ease_linear, ease_in, ease_out, ease_in_out, ease_smoothstep, ease_spring]
    )
    def test_boundary_one(self, fn):
        assert fn(1.0) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("fn", [ease_linear, ease_in, ease_out, ease_in_out, ease_smoothstep])
    def test_monotone(self, fn):
        pts = [fn(t / 10) for t in range(11)]
        for a, b in zip(pts, pts[1:]):
            assert b >= a - 1e-9

    def test_linear_midpoint(self):
        assert ease_linear(0.5) == pytest.approx(0.5)

    def test_ease_in_slow_start(self):
        assert ease_in(0.1) < 0.1

    def test_ease_out_fast_start(self):
        assert ease_out(0.1) > ease_in(0.1)

    def test_ease_in_out_symmetric(self):
        for t in [0.1, 0.2, 0.3, 0.4]:
            assert ease_in_out(t) == pytest.approx(1 - ease_in_out(1 - t), abs=1e-9)

    def test_spring_overshoots(self):
        values = [ease_spring(t / 100) for t in range(101)]
        assert any(v > 1.0 for v in values)


class TestSmoothstep:
    """Validate the exact smoothstep formula: t² * (3 − 2t)."""

    def test_formula_at_quarter(self):
        t = 0.25
        expected = t * t * (3 - 2 * t)  # 0.25² * (3 - 0.5) = 0.0625 * 2.5 = 0.15625
        assert ease_smoothstep(t) == pytest.approx(expected)

    def test_formula_at_half(self):
        t = 0.5
        expected = t * t * (3 - 2 * t)  # 0.25 * 2 = 0.5
        assert ease_smoothstep(t) == pytest.approx(expected)

    def test_formula_at_three_quarters(self):
        t = 0.75
        expected = t * t * (3 - 2 * t)  # 0.5625 * 1.5 = 0.84375
        assert ease_smoothstep(t) == pytest.approx(expected)

    def test_clips_below_zero(self):
        assert ease_smoothstep(-0.5) == pytest.approx(0.0)

    def test_clips_above_one(self):
        assert ease_smoothstep(1.5) == pytest.approx(1.0)

    def test_symmetric_around_half(self):
        """smoothstep is point-symmetric around (0.5, 0.5)."""
        for t in [0.1, 0.2, 0.3, 0.4]:
            assert ease_smoothstep(t) == pytest.approx(1 - ease_smoothstep(1 - t), abs=1e-9)

    def test_easing_type_smoothstep(self):
        result = apply_easing(0.5, EasingType.SMOOTHSTEP)
        assert result == pytest.approx(0.5)


class TestApplyEasing:
    def test_clips_below_zero(self):
        assert apply_easing(-0.5, EasingType.LINEAR) == pytest.approx(0.0)

    def test_clips_above_one(self):
        assert apply_easing(1.5, EasingType.LINEAR) == pytest.approx(1.0)

    def test_all_easing_types_work(self):
        for et in EasingType:
            result = apply_easing(0.5, et)
            assert 0.0 <= result <= 1.5  # allow spring overshoot at t=0.5


class TestInterpolateValue:
    def test_start_at_t0(self):
        assert interpolate_value(10, 20, 0.0, EasingType.LINEAR) == pytest.approx(10.0)

    def test_end_at_t1(self):
        assert interpolate_value(10, 20, 1.0, EasingType.LINEAR) == pytest.approx(20.0)

    def test_midpoint_linear(self):
        assert interpolate_value(0, 100, 0.5, EasingType.LINEAR) == pytest.approx(50.0)

    def test_negative_range(self):
        v = interpolate_value(-100, 100, 0.5, EasingType.LINEAR)
        assert v == pytest.approx(0.0)

    def test_smoothstep_midpoint(self):
        # smoothstep(0.5) = 0.5, so midpoint should equal linear midpoint
        v = interpolate_value(0, 100, 0.5, EasingType.SMOOTHSTEP)
        assert v == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# compute_frame_timings — integer ms contract
# ---------------------------------------------------------------------------


class TestComputeFrameTimings:
    def test_returns_list_of_frame_timings(self):
        events = make_events(
            [(0.0, "navigate", {"value": "http://x"}), (1.0, "click", {"x": 0, "y": 0})]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        assert all(isinstance(t, FrameTiming) for t in timings)

    def test_timestamp_ms_is_int(self):
        events = make_events([(0.0, "navigate", {}), (1.0, "click", {"x": 0, "y": 0})])
        timings = compute_frame_timings(events, DEFAULT_CFG)
        for t in timings:
            assert isinstance(t.timestamp_ms, int), (
                f"timestamp_ms should be int, got {type(t.timestamp_ms)}"
            )

    def test_duration_ms_is_int(self):
        events = make_events([(0.0, "navigate", {}), (1.0, "click", {"x": 0, "y": 0})])
        timings = compute_frame_timings(events, DEFAULT_CFG)
        for t in timings:
            assert isinstance(t.duration_ms, int), (
                f"duration_ms should be int, got {type(t.duration_ms)}"
            )

    def test_count_matches_events(self):
        events = make_events(
            [
                (0.0, "navigate", {}),
                (1.0, "click", {"x": 100, "y": 200}),
                (2.0, "type", {"value": "hi"}),
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        assert len(timings) == 3

    def test_first_timestamp_is_zero(self):
        events = make_events([(5.0, "navigate", {}), (6.0, "click", {"x": 0, "y": 0})])
        timings = compute_frame_timings(events, DEFAULT_CFG)
        assert timings[0].timestamp_ms == 0

    def test_relative_timestamps(self):
        events = make_events([(10.0, "navigate", {}), (11.5, "click", {"x": 0, "y": 0})])
        timings = compute_frame_timings(events, DEFAULT_CFG)
        assert timings[1].timestamp_ms == 1500

    def test_duration_from_gap(self):
        events = make_events([(0.0, "click", {"x": 0, "y": 0}), (2.0, "hover", {"x": 1, "y": 1})])
        timings = compute_frame_timings(events, DEFAULT_CFG)
        assert timings[0].duration_ms == 2000

    def test_min_duration_floored(self):
        cfg = TimingConfig(min_frame_duration_ms=500.0)
        events = make_events([(0.0, "click", {"x": 0, "y": 0}), (0.001, "click", {"x": 1, "y": 1})])
        timings = compute_frame_timings(events, cfg)
        assert timings[0].duration_ms >= 500

    def test_gap_clamped(self):
        cfg = TimingConfig(max_gap_duration_ms=1000.0, min_frame_duration_ms=100.0)
        events = make_events([(0.0, "wait", {}), (60.0, "click", {"x": 0, "y": 0})])
        timings = compute_frame_timings(events, cfg)
        assert timings[0].duration_ms <= 1000

    def test_speed_multiplier(self):
        cfg = TimingConfig(speed_multiplier=2.0, min_frame_duration_ms=50.0)
        events = make_events([(0.0, "click", {"x": 0, "y": 0}), (2.0, "hover", {"x": 1, "y": 1})])
        timings = compute_frame_timings(events, cfg)
        assert timings[0].duration_ms == 1000

    def test_empty_events(self):
        assert compute_frame_timings([], DEFAULT_CFG) == []

    def test_frame_index_sequential(self):
        events = make_events([(i * 1.0, "click", {"x": 0, "y": 0}) for i in range(5)])
        timings = compute_frame_timings(events, DEFAULT_CFG)
        for i, t in enumerate(timings):
            assert t.frame_index == i

    # ------------------------------------------------------------------
    # Last-event capping: max(frame_duration) — NOT action-based dwell
    # ------------------------------------------------------------------

    def test_last_frame_capped_at_max_duration(self):
        """Last frame duration must equal the max of all preceding frames."""
        events = make_events(
            [
                (0.0, "navigate", {"value": "http://x"}),  # gap → 1000 ms
                (1.0, "click", {"x": 0, "y": 0}),  # gap → 500 ms
                (1.5, "hover", {"x": 1, "y": 1}),  # last event
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        preceding_max = max(t.duration_ms for t in timings[:-1])
        assert timings[-1].duration_ms == preceding_max

    def test_last_frame_not_action_dwell(self):
        """navigate_dwell or click_dwell must NOT determine the last frame duration."""
        cfg = TimingConfig(
            navigate_dwell_ms=9999.0,  # would inflate last frame if old logic used
            min_frame_duration_ms=100.0,
        )
        events = make_events(
            [
                (0.0, "click", {"x": 0, "y": 0}),  # gap 500 ms
                (0.5, "click", {"x": 1, "y": 1}),  # gap 500 ms
                (1.0, "navigate", {"value": "http://x"}),  # last — should be 500, not 9999
            ]
        )
        timings = compute_frame_timings(events, cfg)
        # navigate_dwell would give 9999; max of previous durations is 500
        assert timings[-1].duration_ms <= 500

    def test_single_event_uses_min_duration(self):
        cfg = TimingConfig(min_frame_duration_ms=200.0)
        events = make_events([(0.0, "screenshot", {})])
        timings = compute_frame_timings(events, cfg)
        assert timings[0].duration_ms == 200


# ---------------------------------------------------------------------------
# total_duration_ms — returns int
# ---------------------------------------------------------------------------


class TestTotalDuration:
    def test_empty(self):
        assert total_duration_ms([]) == 0

    def test_returns_int(self):
        t = FrameTiming(frame_index=0, timestamp_ms=0, duration_ms=500)
        result = total_duration_ms([t])
        assert isinstance(result, int)

    def test_single(self):
        t = FrameTiming(frame_index=0, timestamp_ms=0, duration_ms=500)
        assert total_duration_ms([t]) == 500

    def test_multiple(self):
        timings = [
            FrameTiming(0, 0, 1000),
            FrameTiming(1, 1000, 500),
            FrameTiming(2, 1500, 300),
        ]
        assert total_duration_ms(timings) == 1800


# ---------------------------------------------------------------------------
# build_cursor_path — segment-relative int timestamps, smoothstep
# ---------------------------------------------------------------------------


class TestBuildCursorPath:
    def test_returns_keyframes(self):
        events = make_events(
            [
                (0.0, "hover", {"x": 100, "y": 100}),
                (1.0, "click", {"x": 200, "y": 200}),
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        kfs = build_cursor_path(events, timings, DEFAULT_CFG)
        assert len(kfs) > 0

    def test_no_coordinates_returns_empty(self):
        events = make_events(
            [
                (0.0, "navigate", {"value": "http://x"}),
                (1.0, "wait", {}),
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        kfs = build_cursor_path(events, timings, DEFAULT_CFG)
        assert kfs == []

    def test_timestamp_ms_is_int(self):
        events = make_events(
            [
                (0.0, "hover", {"x": 10, "y": 20}),
                (1.0, "click", {"x": 100, "y": 100}),
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        kfs = build_cursor_path(events, timings, DEFAULT_CFG)
        for kf in kfs:
            assert isinstance(kf.timestamp_ms, int), (
                f"timestamp_ms should be int, got {type(kf.timestamp_ms)}"
            )

    def test_duration_ms_is_int(self):
        events = make_events(
            [
                (0.0, "hover", {"x": 10, "y": 20}),
                (1.0, "click", {"x": 100, "y": 100}),
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        kfs = build_cursor_path(events, timings, DEFAULT_CFG)
        for kf in kfs:
            assert isinstance(kf.duration_ms, int), (
                f"duration_ms should be int, got {type(kf.duration_ms)}"
            )

    def test_first_keyframe_timestamp_is_zero(self):
        """Segment-relative: first anchor is always at t=0."""
        events = make_events(
            [
                (5.0, "hover", {"x": 10, "y": 20}),
                (6.0, "click", {"x": 100, "y": 100}),
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        kfs = build_cursor_path(events, timings, DEFAULT_CFG)
        assert kfs[0].timestamp_ms == 0

    def test_keyframes_ordered_by_timestamp(self):
        events = make_events(
            [
                (0.0, "hover", {"x": 0, "y": 0}),
                (1.0, "click", {"x": 500, "y": 400}),
                (2.0, "hover", {"x": 100, "y": 100}),
            ]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        kfs = build_cursor_path(events, timings, DEFAULT_CFG)
        ts = [k.timestamp_ms for k in kfs]
        assert ts == sorted(ts)

    def test_interpolation_steps(self):
        cfg = TimingConfig(cursor_interpolation_steps=5, min_frame_duration_ms=50)
        events = make_events(
            [
                (0.0, "hover", {"x": 10, "y": 20}),
                (1.0, "click", {"x": 100, "y": 100}),
            ]
        )
        timings = compute_frame_timings(events, cfg)
        kfs = build_cursor_path(events, timings, cfg)
        # 1 start anchor + 5 interpolated steps = 6 keyframes
        assert len(kfs) == 6

    def test_no_interpolation_steps_one(self):
        cfg = TimingConfig(cursor_interpolation_steps=1, min_frame_duration_ms=50)
        events = make_events(
            [
                (0.0, "hover", {"x": 10, "y": 20}),
                (1.0, "click", {"x": 100, "y": 100}),
            ]
        )
        timings = compute_frame_timings(events, cfg)
        kfs = build_cursor_path(events, timings, cfg)
        assert len(kfs) == 2

    def test_start_position_correct(self):
        events = make_events(
            [(0.0, "hover", {"x": 42.0, "y": 84.0}), (1.0, "click", {"x": 50.0, "y": 90.0})]
        )
        timings = compute_frame_timings(events, DEFAULT_CFG)
        kfs = build_cursor_path(events, timings, DEFAULT_CFG)
        assert kfs[0].x == pytest.approx(42.0)
        assert kfs[0].y == pytest.approx(84.0)

    def test_end_position_correct(self):
        cfg = TimingConfig(cursor_interpolation_steps=1, min_frame_duration_ms=50)
        events = make_events(
            [
                (0.0, "hover", {"x": 5.0, "y": 10.0}),
                (1.0, "click", {"x": 99.0, "y": 77.0}),
            ]
        )
        timings = compute_frame_timings(events, cfg)
        kfs = build_cursor_path(events, timings, cfg)
        assert kfs[-1].x == pytest.approx(99.0)
        assert kfs[-1].y == pytest.approx(77.0)

    def test_smoothstep_is_default_easing(self):
        """Default cursor easing must be smoothstep per schema contract."""
        cfg = TimingConfig(cursor_interpolation_steps=5, min_frame_duration_ms=50)
        events = make_events(
            [
                (0.0, "hover", {"x": 0.0, "y": 0.0}),
                (1.0, "click", {"x": 100.0, "y": 0.0}),
            ]
        )
        timings = compute_frame_timings(events, cfg)
        kfs = build_cursor_path(events, timings, cfg)
        for kf in kfs:
            assert kf.easing == EasingType.SMOOTHSTEP

    def test_smoothstep_midpoint_position(self):
        """At t=0.5, smoothstep(0.5)=0.5 so midpoint x should equal linear midpoint."""
        cfg = TimingConfig(
            cursor_easing=EasingType.SMOOTHSTEP,
            cursor_interpolation_steps=2,  # t=0.5 and t=1.0
            min_frame_duration_ms=50,
        )
        events = make_events(
            [
                (0.0, "hover", {"x": 0.0, "y": 0.0}),
                (1.0, "click", {"x": 100.0, "y": 0.0}),
            ]
        )
        timings = compute_frame_timings(events, cfg)
        kfs = build_cursor_path(events, timings, cfg)
        # kfs[1] corresponds to t=0.5; smoothstep(0.5)=0.5 → x=50
        assert kfs[1].x == pytest.approx(50.0, abs=0.5)

    def test_duplicate_timestamps_deduped(self):
        cfg = TimingConfig(cursor_interpolation_steps=5, min_frame_duration_ms=50)
        events = make_events(
            [
                (0.0, "hover", {"x": 5.0, "y": 10.0}),
                (0.0, "click", {"x": 50.0, "y": 50.0}),
            ]
        )
        timings = compute_frame_timings(events, cfg)
        kfs = build_cursor_path(events, timings, cfg)
        ts_vals = [kf.timestamp_ms for kf in kfs]
        assert len(ts_vals) == len(set(ts_vals))
