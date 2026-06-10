"""Tests for Pydantic timeline schemas."""

import json

import pytest
from pydantic import ValidationError

from src.schemas import Timeline


def test_valid_type_event_passes() -> None:
    """Validate a minimal type event."""
    data = {
        "events": [
            {
                "event_type": "type",
                "code": "print('hello')",
            }
        ]
    }

    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "type"


def test_valid_run_event_passes() -> None:
    """Validate a minimal run event."""
    data = {
        "events": [
            {
                "event_type": "run",
                "command": "python main.py",
                "expected_output": "hello",
            }
        ]
    }

    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "run"


def test_valid_highlight_event_passes() -> None:
    """Validate a highlight event with a correct line range."""
    data = {
        "events": [
            {
                "event_type": "highlight",
                "start_line": 2,
                "end_line": 4,
            }
        ]
    }

    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "highlight"


def test_valid_scroll_event_passes() -> None:
    """Validate a scroll event with a positive target line."""
    data = {
        "events": [
            {
                "event_type": "scroll",
                "target_line": 10,
            }
        ]
    }

    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "scroll"


def test_valid_full_timeline_passes() -> None:
    """Validate a timeline containing all supported event types."""
    data = {
        "events": [
            {
                "event_type": "type",
                "code": "def add(a, b):\n    return a + b",
            },
            {
                "event_type": "highlight",
                "start_line": 2,
                "end_line": 2,
            },
            {
                "event_type": "run",
                "command": "add(2, 3)",
                "expected_output": "5",
            },
            {
                "event_type": "scroll",
                "target_line": 1,
            },
        ]
    }

    timeline = Timeline.model_validate(data)

    assert len(timeline.events) == 4


def test_invalid_event_type_fails() -> None:
    """Reject unsupported event types."""
    data = {
        "events": [
            {
                "event_type": "zoom",
                "target_line": 3,
            }
        ]
    }

    with pytest.raises(ValidationError):
        Timeline.model_validate(data)


def test_extra_fields_fail() -> None:
    """Reject fields that are not part of the event schema."""
    data = {
        "events": [
            {
                "event_type": "run",
                "command": "print('hello')",
                "expected_output": "hello",
                "duration": 3,
            }
        ]
    }

    with pytest.raises(ValidationError):
        Timeline.model_validate(data)


def test_highlight_end_before_start_fails() -> None:
    """Reject highlight events where end_line is before start_line."""
    data = {
        "events": [
            {
                "event_type": "highlight",
                "start_line": 5,
                "end_line": 2,
            }
        ]
    }

    with pytest.raises(ValidationError):
        Timeline.model_validate(data)


def test_zero_line_number_fails() -> None:
    """Reject line numbers less than one."""
    data = {
        "events": [
            {
                "event_type": "scroll",
                "target_line": 0,
            }
        ]
    }

    with pytest.raises(ValidationError):
        Timeline.model_validate(data)


def test_empty_events_list_fails() -> None:
    """Reject timelines with no events."""
    data = {"events": []}

    with pytest.raises(ValidationError):
        Timeline.model_validate(data)


def test_raw_json_can_be_validated() -> None:
    """Validate parsed JSON data against the Timeline schema."""
    raw_json = json.dumps(
        {
            "events": [
                {
                    "event_type": "type",
                    "code": "x = 1",
                }
            ]
        }
    )

    data = json.loads(raw_json)
    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "type"