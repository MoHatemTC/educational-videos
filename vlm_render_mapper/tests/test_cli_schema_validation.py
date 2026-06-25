"""Tests for CLI JSON Schema validation (jsonschema.validate integration)."""

from __future__ import annotations

import json
import pytest

from vlm_render_mapper.cli import validate_against_json_schema
from vlm_render_mapper.parser import parse_session_text
from vlm_render_mapper.mapper import MapperConfig, RenderMapper
from vlm_render_mapper.timing import TimingConfig
import jsonschema


# ---------------------------------------------------------------------------
# Helpers
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
        {"timestamp": 2.0, "action": "wait", "screenshot": "s2.png"},
    ]
)


def build_plan_dict(**kwargs) -> dict:
    events = parse_session_text(SIMPLE_SESSION)
    timing = TimingConfig(min_frame_duration_ms=50.0)
    cfg = MapperConfig(timing=timing, **kwargs)
    plan = RenderMapper(cfg).map(events)
    return json.loads(plan.model_dump_json())


# ---------------------------------------------------------------------------
# Happy-path: generated plan passes JSON Schema validation
# ---------------------------------------------------------------------------


class TestValidateAgainstJsonSchema:
    def test_generated_plan_passes(self):
        """A freshly generated render plan must pass jsonschema.validate."""
        plan_dict = build_plan_dict()
        # Should not raise
        validate_against_json_schema(plan_dict)

    def test_recorded_at_accepted(self):
        """Schema accepts `recorded_at` (ISO datetime string) in metadata."""
        plan_dict = build_plan_dict()
        assert "recorded_at" in plan_dict["metadata"]
        validate_against_json_schema(plan_dict)

    def test_integer_ms_fields_pass(self):
        """Integer ms fields in frames / captions must satisfy `type: integer`."""
        plan_dict = build_plan_dict()
        frame = plan_dict["frames"][0]
        # Field is now `start_ms` (replaces `timestamp_ms` per contract v1.1)
        assert isinstance(frame["start_ms"], int)
        assert isinstance(frame["duration_ms"], int)
        validate_against_json_schema(plan_dict)

    def test_missing_recorded_at_raises(self):
        plan_dict = build_plan_dict()
        del plan_dict["metadata"]["recorded_at"]
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)

    def test_old_created_at_rejected(self):
        """A plan using `created_at` instead of `recorded_at` must fail."""
        plan_dict = build_plan_dict()
        plan_dict["metadata"]["created_at"] = plan_dict["metadata"].pop("recorded_at")
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)

    def test_float_start_ms_in_frame_rejected(self):
        """Frame start_ms must be integer; a float like 0.5 should fail."""
        plan_dict = build_plan_dict()
        plan_dict["frames"][0]["start_ms"] = 0.5  # float, not integer
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)

    def test_float_duration_ms_in_frame_rejected(self):
        plan_dict = build_plan_dict()
        plan_dict["frames"][0]["duration_ms"] = 200.7
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)

    def test_missing_version_raises(self):
        plan_dict = build_plan_dict()
        del plan_dict["version"]
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)

    def test_missing_frames_raises(self):
        plan_dict = build_plan_dict()
        del plan_dict["frames"]
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)

    def test_empty_frames_raises(self):
        plan_dict = build_plan_dict()
        plan_dict["frames"] = []
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)

    def test_invalid_action_type_raises(self):
        plan_dict = build_plan_dict()
        plan_dict["frames"][0]["action"]["type"] = "teleport"
        with pytest.raises(jsonschema.ValidationError):
            validate_against_json_schema(plan_dict)
