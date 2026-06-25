"""Tests for vlm_render_mapper.parser"""

from __future__ import annotations

import json
import textwrap

import pytest

from vlm_render_mapper.parser import (
    SessionParseError,
    parse_session_text,
    parse_session_file,
    _normalise_action,
    _parse_timestamp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ARRAY = json.dumps(
    [
        {
            "timestamp": 1700000000.0,
            "action": "navigate",
            "value": "https://example.com",
            "screenshot": "frame_000000.png",
        },
        {
            "timestamp": 1700000001.5,
            "action": "click",
            "x": 320,
            "y": 240,
            "target": "#submit-btn",
            "screenshot": "frame_000001.png",
        },
        {
            "timestamp": 1700000002.0,
            "action": "type",
            "value": "Hello World",
            "target": "input[name=query]",
            "screenshot": "frame_000002.png",
        },
    ]
)

VALID_JSONL = textwrap.dedent("""\
    {"timestamp": 1700000000.0, "action": "navigate", "value": "https://example.com", "screenshot": "f0.png"}
    {"timestamp": 1700000001.0, "action": "click", "x": 100, "y": 200, "screenshot": "f1.png"}
    {"timestamp": 1700000002.5, "action": "scroll", "deltaY": 300, "screenshot": "f2.png"}
""")


# ---------------------------------------------------------------------------
# _normalise_action
# ---------------------------------------------------------------------------


class TestNormaliseAction:
    def test_passthrough_valid(self):
        assert _normalise_action("click") == "click"
        assert _normalise_action("type") == "type"
        assert _normalise_action("navigate") == "navigate"

    def test_alias_leftclick(self):
        assert _normalise_action("leftclick") == "click"
        assert _normalise_action("left_click") == "click"

    def test_alias_dblclick(self):
        assert _normalise_action("dblclick") == "double_click"
        assert _normalise_action("doubleclick") == "double_click"

    def test_alias_input(self):
        assert _normalise_action("input") == "type"

    def test_alias_goto(self):
        assert _normalise_action("goto") == "navigate"

    def test_unknown_falls_back(self):
        assert _normalise_action("unknown_action_xyz") == "wait"

    def test_case_insensitive(self):
        assert _normalise_action("CLICK") == "click"
        assert _normalise_action("Navigate") == "navigate"

    def test_hyphenated(self):
        assert _normalise_action("page-load") == "page_load"


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    def test_float(self):
        assert _parse_timestamp(1700000000.0) == pytest.approx(1700000000.0)

    def test_int(self):
        assert _parse_timestamp(1700000000) == pytest.approx(1700000000.0)

    def test_float_string(self):
        assert _parse_timestamp("1700000000.5") == pytest.approx(1700000000.5)

    def test_iso8601(self):
        ts = _parse_timestamp("2023-11-14T22:13:20Z")
        assert ts == pytest.approx(1700000000.0, abs=5)

    def test_iso8601_no_z(self):
        ts = _parse_timestamp("2023-11-14T22:13:20")
        assert isinstance(ts, float)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_timestamp("not-a-date")

    def test_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_timestamp(None)


# ---------------------------------------------------------------------------
# SessionParser – parse_text
# ---------------------------------------------------------------------------


class TestParseText:
    def test_json_array(self):
        events = parse_session_text(VALID_ARRAY)
        assert len(events) == 3
        assert events[0]["action"] == "navigate"
        assert events[1]["action"] == "click"
        assert events[2]["action"] == "type"

    def test_jsonl(self):
        events = parse_session_text(VALID_JSONL)
        assert len(events) == 3
        assert events[2]["action"] == "scroll"

    def test_empty_raises(self):
        with pytest.raises(SessionParseError, match="empty"):
            parse_session_text("")

    def test_invalid_json_raises(self):
        with pytest.raises(SessionParseError):
            parse_session_text("{ not valid json }")

    def test_missing_timestamp_raises(self):
        data = json.dumps([{"action": "click", "x": 0, "y": 0, "screenshot": "f.png"}])
        with pytest.raises(SessionParseError, match="timestamp"):
            parse_session_text(data)

    def test_coordinates_extracted(self):
        events = parse_session_text(VALID_ARRAY)
        click = events[1]
        assert click["x"] == pytest.approx(320.0)
        assert click["y"] == pytest.approx(240.0)

    def test_scroll_delta_extracted(self):
        events = parse_session_text(VALID_JSONL)
        scroll = events[2]
        assert scroll["scroll_delta_y"] == pytest.approx(300.0)

    def test_value_extracted(self):
        events = parse_session_text(VALID_ARRAY)
        assert events[2]["value"] == "Hello World"

    def test_target_extracted(self):
        events = parse_session_text(VALID_ARRAY)
        assert events[1]["target"] == "#submit-btn"

    def test_screenshot_falls_back(self):
        data = json.dumps([{"timestamp": 1.0, "action": "wait"}])
        events = parse_session_text(data)
        assert "frame_" in events[0]["screenshot"]

    def test_alias_action_normalised(self):
        data = json.dumps(
            [{"timestamp": 1.0, "action": "leftclick", "x": 0, "y": 0, "screenshot": "f.png"}]
        )
        events = parse_session_text(data)
        assert events[0]["action"] == "click"

    def test_comment_lines_ignored_in_jsonl(self):
        jsonl = (
            "# This is a comment\n"
            '{"timestamp": 1.0, "action": "click", "x": 0, "y": 0, "screenshot": "f.png"}\n'
            "// Another comment\n"
            '{"timestamp": 2.0, "action": "wait", "screenshot": "f2.png"}\n'
        )
        events = parse_session_text(jsonl)
        assert len(events) == 2

    def test_alt_field_names(self):
        """Parser should handle clientX/clientY and event_type."""
        data = json.dumps(
            [
                {
                    "ts": 1000,
                    "event_type": "hover",
                    "clientX": 50,
                    "clientY": 75,
                    "screenshot": "f.png",
                }
            ]
        )
        events = parse_session_text(data)
        assert events[0]["action"] == "hover"
        assert events[0]["x"] == pytest.approx(50.0)
        assert events[0]["y"] == pytest.approx(75.0)

    def test_index_assigned(self):
        events = parse_session_text(VALID_ARRAY)
        for i, ev in enumerate(events):
            assert ev["index"] == i

    def test_timestamps_absolute(self):
        events = parse_session_text(VALID_ARRAY)
        assert events[0]["timestamp"] == pytest.approx(1700000000.0)
        assert events[1]["timestamp"] == pytest.approx(1700000001.5)


# ---------------------------------------------------------------------------
# SessionParser – parse_file
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_json_file(self, tmp_path):
        f = tmp_path / "session.json"
        f.write_text(VALID_ARRAY, encoding="utf-8")
        events = parse_session_file(f)
        assert len(events) == 3

    def test_jsonl_file(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(VALID_JSONL, encoding="utf-8")
        events = parse_session_file(f)
        assert len(events) == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SessionParseError, match="not found"):
            parse_session_file(tmp_path / "no_such_file.json")

    def test_string_path_accepted(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(VALID_ARRAY, encoding="utf-8")
        events = parse_session_file(str(f))
        assert len(events) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_event(self):
        data = json.dumps([{"timestamp": 1.0, "action": "screenshot", "screenshot": "f.png"}])
        events = parse_session_text(data)
        assert len(events) == 1

    def test_drag_fields_preserved(self):
        data = json.dumps(
            [
                {
                    "timestamp": 1.0,
                    "action": "drag",
                    "drag_start": {"x": 10, "y": 20},
                    "drag_end": {"x": 100, "y": 200},
                    "screenshot": "f.png",
                }
            ]
        )
        events = parse_session_text(data)
        assert events[0]["drag_start"] == {"x": 10, "y": 20}
        assert events[0]["drag_end"] == {"x": 100, "y": 200}

    def test_meta_preserved(self):
        data = json.dumps(
            [
                {
                    "timestamp": 1.0,
                    "action": "click",
                    "x": 0,
                    "y": 0,
                    "screenshot": "f.png",
                    "meta": {"modifiers": ["ctrl"]},
                }
            ]
        )
        events = parse_session_text(data)
        assert events[0]["meta"]["modifiers"] == ["ctrl"]
