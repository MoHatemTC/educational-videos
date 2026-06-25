"""Tests for vlm_render_mapper.mapper"""

from __future__ import annotations

import json
import pytest

from vlm_render_mapper.parser import parse_session_text
from vlm_render_mapper.mapper import MapperConfig, RenderMapper, _make_caption_text
from vlm_render_mapper.schema import (
    ActionType,
    CaptionPosition,
    EasingType,
    RenderPlan,
    RenderTarget,
    TransitionType,
)
from vlm_render_mapper.timing import TimingConfig


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SIMPLE_SESSION = json.dumps(
    [
        {
            "timestamp": 0.0,
            "action": "navigate",
            "value": "https://example.com",
            "screenshot": "s0.png",
        },
        {
            "timestamp": 1.0,
            "action": "click",
            "x": 640,
            "y": 360,
            "target": "#btn",
            "screenshot": "s1.png",
        },
        {
            "timestamp": 2.0,
            "action": "type",
            "value": "hello",
            "target": "input",
            "screenshot": "s2.png",
        },
        {"timestamp": 3.0, "action": "scroll", "deltaY": 200, "screenshot": "s3.png"},
        {"timestamp": 4.0, "action": "screenshot", "screenshot": "s4.png"},
    ]
)

HOVER_SESSION = json.dumps(
    [
        {
            "timestamp": 0.0,
            "action": "hover",
            "x": 100,
            "y": 100,
            "target": ".menu",
            "screenshot": "h0.png",
        },
        {
            "timestamp": 0.5,
            "action": "click",
            "x": 100,
            "y": 150,
            "target": ".item",
            "screenshot": "h1.png",
        },
    ]
)

DRAG_SESSION = json.dumps(
    [
        {
            "timestamp": 0.0,
            "action": "drag",
            "drag_start": {"x": 10, "y": 10},
            "drag_end": {"x": 200, "y": 200},
            "screenshot": "d0.png",
        },
    ]
)


def make_mapper(**kwargs) -> RenderMapper:
    timing = TimingConfig(min_frame_duration_ms=50.0, frame_rate=30)
    cfg = MapperConfig(timing=timing, **kwargs)
    return RenderMapper(cfg)


def map_simple(**kwargs) -> RenderPlan:
    events = parse_session_text(SIMPLE_SESSION)
    return make_mapper(**kwargs).map(events)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestRenderPlanStructure:
    def test_returns_render_plan(self):
        plan = map_simple()
        assert isinstance(plan, RenderPlan)

    def test_frame_count_matches_events(self):
        # Click-like actions (click, double_click, right_click) expand into
        # two frames: a cursor-move frame then a click-highlight frame.
        # SIMPLE_SESSION has 1 click event, so we expect 5 events → 6 frames.
        plan = map_simple()
        events = parse_session_text(SIMPLE_SESSION)
        click_actions = {"click", "double_click", "right_click"}
        click_count = sum(1 for ev in events if ev["action"] in click_actions)
        assert len(plan.frames) == len(events) + click_count

    def test_frame_ids_sequential(self):
        plan = map_simple()
        for i, f in enumerate(plan.frames):
            assert f.frame_id == f"frame_{i:06d}"

    def test_version_set(self):
        plan = map_simple()
        assert plan.version == "1.0.0"

    def test_metadata_session_id(self):
        plan = map_simple(session_id="test-session-42")
        assert plan.metadata.session_id == "test-session-42"

    def test_metadata_frame_rate(self):
        plan = map_simple(frame_rate=60)
        assert plan.metadata.frame_rate == 60

    def test_metadata_resolution(self):
        plan = map_simple(viewport_width=1920, viewport_height=1080)
        assert plan.metadata.resolution.width == 1920
        assert plan.metadata.resolution.height == 1080

    def test_render_target(self):
        plan = map_simple(render_target=RenderTarget.REMOTION)
        assert plan.metadata.render_target == RenderTarget.REMOTION

    def test_total_duration_positive(self):
        plan = map_simple()
        assert plan.metadata.total_duration_ms > 0

    def test_timeline_has_one_segment(self):
        plan = map_simple()
        assert len(plan.timeline.segments) == 1

    def test_timeline_covers_all_frames(self):
        plan = map_simple()
        seg = plan.timeline.segments[0]
        assert len(seg.frame_ids) == len(plan.frames)

    def test_timeline_start_zero(self):
        plan = map_simple()
        assert plan.timeline.segments[0].start_ms == 0

    def test_serialise_round_trip(self):
        plan = map_simple()
        raw = plan.model_dump_json()
        restored = RenderPlan.model_validate_json(raw)
        assert len(restored.frames) == len(plan.frames)


# ---------------------------------------------------------------------------
# Schema contract: recorded_at (not created_at)
# ---------------------------------------------------------------------------


class TestMetadataSchema:
    def test_recorded_at_field_exists(self):
        """metadata must expose `recorded_at`, not `created_at`."""
        plan = map_simple()
        assert hasattr(plan.metadata, "recorded_at"), "RenderMetadata must have `recorded_at` field"

    def test_created_at_field_absent(self):
        """Old `created_at` field must NOT be present."""
        plan = map_simple()
        assert not hasattr(plan.metadata, "created_at"), (
            "RenderMetadata must not have `created_at` (use `recorded_at`)"
        )

    def test_recorded_at_in_serialised_json(self):
        plan = map_simple()
        raw = json.loads(plan.model_dump_json())
        assert "recorded_at" in raw["metadata"]
        assert "created_at" not in raw["metadata"]

    def test_total_duration_ms_is_int(self):
        plan = map_simple()
        assert isinstance(plan.metadata.total_duration_ms, int), (
            "total_duration_ms must be an integer"
        )


# ---------------------------------------------------------------------------
# Integer ms contract
# ---------------------------------------------------------------------------


class TestIntegerMilliseconds:
    def test_frame_timestamp_ms_is_int(self):
        plan = map_simple()
        for f in plan.frames:
            assert isinstance(f.timestamp_ms, int), (
                f"frame {f.frame_id} timestamp_ms should be int, got {type(f.timestamp_ms)}"
            )

    def test_frame_duration_ms_is_int(self):
        plan = map_simple()
        for f in plan.frames:
            assert isinstance(f.duration_ms, int), (
                f"frame {f.frame_id} duration_ms should be int, got {type(f.duration_ms)}"
            )

    def test_caption_start_ms_is_int(self):
        plan = map_simple()
        for c in plan.captions:
            assert isinstance(c.start_ms, int), f"caption {c.caption_id} start_ms should be int"

    def test_caption_end_ms_is_int(self):
        plan = map_simple()
        for c in plan.captions:
            assert isinstance(c.end_ms, int), f"caption {c.caption_id} end_ms should be int"

    def test_timeline_start_ms_is_int(self):
        plan = map_simple()
        for seg in plan.timeline.segments:
            assert isinstance(seg.start_ms, int)
            assert isinstance(seg.end_ms, int)

    def test_cursor_keyframe_timestamp_ms_is_int(self):
        plan = map_simple()
        for kf in plan.cursor_path:
            assert isinstance(kf.timestamp_ms, int), (
                f"cursor keyframe timestamp_ms should be int, got {type(kf.timestamp_ms)}"
            )

    def test_cursor_keyframe_duration_ms_is_int(self):
        plan = map_simple()
        for kf in plan.cursor_path:
            assert isinstance(kf.duration_ms, int)


# ---------------------------------------------------------------------------
# FrameDescriptor structure (renamed from Frame)
# ---------------------------------------------------------------------------


class TestFrameDescriptor:
    def test_screenshot_ref_present(self):
        """FrameDescriptor must expose screenshot_ref."""
        plan = map_simple()
        assert hasattr(plan.frames[0], "screenshot_ref")

    def test_screenshot_ref_value_preserved(self):
        plan = map_simple()
        assert plan.frames[0].screenshot_ref == "s0.png"

    def test_frame_has_timestamp_ms(self):
        plan = map_simple()
        assert hasattr(plan.frames[0], "timestamp_ms")

    def test_frame_has_duration_ms(self):
        plan = map_simple()
        assert hasattr(plan.frames[0], "duration_ms")


# ---------------------------------------------------------------------------
# Cursor path — smoothstep easing
# ---------------------------------------------------------------------------


class TestCursorPathEasing:
    def test_default_easing_is_smoothstep(self):
        plan = map_simple()
        for kf in plan.cursor_path:
            assert kf.easing == EasingType.SMOOTHSTEP, f"Expected SMOOTHSTEP, got {kf.easing}"

    def test_cursor_path_segment_relative_starts_at_zero(self):
        plan = map_simple()
        if plan.cursor_path:
            assert plan.cursor_path[0].timestamp_ms == 0


# ---------------------------------------------------------------------------
# Frame actions
# ---------------------------------------------------------------------------


class TestFrameActions:
    def test_navigate_action_type(self):
        plan = map_simple()
        assert plan.frames[0].action.type == ActionType.NAVIGATE

    def test_click_action_type(self):
        plan = map_simple()
        assert plan.frames[1].action.type == ActionType.CLICK

    def test_type_action_value(self):
        plan = map_simple()
        # click expands into 2 frames, so type is now at index 3
        assert plan.frames[3].action.value == "hello"

    def test_scroll_action_delta(self):
        plan = map_simple()
        # click expands into 2 frames, so scroll is now at index 4
        scroll_frame = plan.frames[4]
        assert scroll_frame.action.type == ActionType.SCROLL
        assert scroll_frame.action.scroll_delta is not None
        assert scroll_frame.action.scroll_delta.y == pytest.approx(200.0)

    def test_click_coordinates(self):
        plan = map_simple()
        coords = plan.frames[1].action.coordinates
        assert coords is not None
        assert coords.x == pytest.approx(640.0)
        assert coords.y == pytest.approx(360.0)

    def test_drag_endpoints(self):
        events = parse_session_text(DRAG_SESSION)
        plan = make_mapper().map(events)
        action = plan.frames[0].action
        assert action.drag_start is not None
        assert action.drag_end is not None
        assert action.drag_start.x == pytest.approx(10.0)
        assert action.drag_end.x == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------


class TestHighlightRegions:
    def test_click_has_highlight(self):
        plan = map_simple()
        # Click expands to 2 frames: frames[1]=cursor-move (no hl), frames[2]=click-highlight
        click_hl_frame = plan.frames[2]
        assert len(click_hl_frame.highlight_regions) == 1

    def test_click_cursor_move_frame_has_no_highlight(self):
        plan = map_simple()
        # The first (cursor-move) frame of a click must have no highlight
        click_move_frame = plan.frames[1]
        assert len(click_move_frame.highlight_regions) == 0

    def test_highlight_centred_on_cursor(self):
        plan = map_simple()
        # Highlight is on the click-highlight (B) frame
        hl = plan.frames[2].highlight_regions[0]
        cfg = MapperConfig()
        half = cfg.highlight_size / 2
        assert hl.x == pytest.approx(640.0 - half)
        assert hl.y == pytest.approx(360.0 - half)

    def test_hover_has_highlight(self):
        events = parse_session_text(HOVER_SESSION)
        plan = make_mapper().map(events)
        assert len(plan.frames[0].highlight_regions) == 1

    def test_navigate_no_highlight(self):
        plan = map_simple()
        assert len(plan.frames[0].highlight_regions) == 0

    def test_scroll_no_highlight(self):
        plan = map_simple()
        # click expands into 2 frames, so scroll is now at index 4
        assert len(plan.frames[4].highlight_regions) == 0

    def test_highlight_color_configurable(self):
        events = parse_session_text(SIMPLE_SESSION)
        cfg = MapperConfig(
            click_highlight_color="#00FF00",
            timing=TimingConfig(min_frame_duration_ms=50),
        )
        plan = RenderMapper(cfg).map(events)
        # Highlight is on the click-highlight (B) frame at index 2
        hl = plan.frames[2].highlight_regions[0]
        assert hl.color == "#00FF00"


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


class TestCaptions:
    def test_captions_generated(self):
        plan = map_simple()
        # Cursor-move frames (the first of each click pair) carry no caption,
        # so caption count equals the number of events, not the number of frames.
        events = parse_session_text(SIMPLE_SESSION)
        assert len(plan.captions) == len(events)

    def test_caption_ids_linked_to_frames(self):
        plan = map_simple()
        caption_ids = {c.caption_id for c in plan.captions}
        for f in plan.frames:
            for cid in f.caption_ids:
                assert cid in caption_ids

    def test_navigate_caption_text(self):
        plan = map_simple()
        cap = plan.captions[0]
        assert "Navigate" in cap.text or "navigate" in cap.text.lower()

    def test_type_caption_includes_value(self):
        plan = map_simple()
        cap = plan.captions[2]
        assert "hello" in cap.text

    def test_caption_timestamps_ordered(self):
        plan = map_simple()
        for cap in plan.captions:
            assert cap.end_ms > cap.start_ms

    def test_no_captions_flag(self):
        events = parse_session_text(SIMPLE_SESSION)
        cfg = MapperConfig(generate_captions=False, timing=TimingConfig(min_frame_duration_ms=50))
        plan = RenderMapper(cfg).map(events)
        assert len(plan.captions) == 0

    def test_caption_position_configurable(self):
        events = parse_session_text(SIMPLE_SESSION)
        cfg = MapperConfig(
            caption_position=CaptionPosition.TOP,
            timing=TimingConfig(min_frame_duration_ms=50),
        )
        plan = RenderMapper(cfg).map(events)
        assert all(c.position == CaptionPosition.TOP for c in plan.captions)


class TestCaptionText:
    def test_click_caption(self):
        ev = {"action": "click", "target": "#btn", "value": None, "scroll_delta_y": None}
        text = _make_caption_text(ev)
        assert "#btn" in text

    def test_type_caption(self):
        ev = {"action": "type", "target": "input", "value": "test text", "scroll_delta_y": None}
        text = _make_caption_text(ev)
        assert "test text" in text

    def test_scroll_down_caption(self):
        ev = {"action": "scroll", "target": None, "value": None, "scroll_delta_y": 100}
        text = _make_caption_text(ev)
        assert "down" in text.lower()

    def test_scroll_up_caption(self):
        ev = {"action": "scroll", "target": None, "value": None, "scroll_delta_y": -100}
        text = _make_caption_text(ev)
        assert "up" in text.lower()


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


class TestTransitions:
    def test_navigate_creates_fade(self):
        plan = map_simple()
        tr_types = [t.type for t in plan.transitions]
        assert TransitionType.FADE in tr_types

    def test_transition_frame_refs_valid(self):
        plan = map_simple()
        frame_ids = {f.frame_id for f in plan.frames}
        for tr in plan.transitions:
            if tr.from_frame_id:
                assert tr.from_frame_id in frame_ids
            if tr.to_frame_id:
                assert tr.to_frame_id in frame_ids

    def test_transition_start_ms_is_int(self):
        plan = map_simple()
        for tr in plan.transitions:
            assert isinstance(tr.start_ms, int)

    def test_transition_duration_ms_is_int(self):
        plan = map_simple()
        for tr in plan.transitions:
            assert isinstance(tr.duration_ms, int)


# ---------------------------------------------------------------------------
# Cursor path
# ---------------------------------------------------------------------------


class TestCursorPath:
    def test_cursor_path_generated(self):
        plan = map_simple()
        assert len(plan.cursor_path) > 0

    def test_cursor_keyframes_ordered(self):
        plan = map_simple()
        ts_list = [kf.timestamp_ms for kf in plan.cursor_path]
        assert ts_list == sorted(ts_list)

    def test_cursor_keyframe_values(self):
        plan = map_simple()
        for kf in plan.cursor_path:
            assert isinstance(kf.x, float)
            assert isinstance(kf.y, float)
            assert kf.timestamp_ms >= 0

    def test_no_cursor_if_no_coordinates(self):
        data = json.dumps(
            [
                {
                    "timestamp": 0.0,
                    "action": "navigate",
                    "value": "https://x.com",
                    "screenshot": "f.png",
                },
                {"timestamp": 1.0, "action": "wait", "screenshot": "f2.png"},
            ]
        )
        events = parse_session_text(data)
        plan = make_mapper().map(events)
        assert plan.cursor_path == []


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_events_raises(self):
        with pytest.raises(ValueError):
            make_mapper().map([])

    def test_single_event_plan(self):
        data = json.dumps([{"timestamp": 0.0, "action": "screenshot", "screenshot": "only.png"}])
        events = parse_session_text(data)
        plan = make_mapper().map(events)
        assert len(plan.frames) == 1

    def test_plan_validates_pydantic(self):
        plan = map_simple()
        raw = json.loads(plan.model_dump_json())
        restored = RenderPlan.model_validate(raw)
        assert restored.metadata.session_id == plan.metadata.session_id
