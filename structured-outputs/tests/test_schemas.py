"""Tests for Pydantic timeline schemas."""

import json

import pytest
from pydantic import ValidationError

from src.schemas import Timeline


def test_valid_type_event_passes() -> None:
    """Validate a timed type event."""
    data = {
        "events": [
            {
                "event_type": "type",
                "start_ms": 0,
                "end_ms": 1000,
                "code": "print('hello')",
            }
        ]
    }

    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "type"


def test_valid_run_event_passes() -> None:
    """Validate a timed run event."""
    data = {
        "events": [
            {
                "event_type": "run",
                "start_ms": 1000,
                "end_ms": 1800,
                "command": "python main.py",
                "expected_output": "hello",
            }
        ]
    }

    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "run"


def test_valid_highlight_event_passes() -> None:
    """Validate a timed highlight event with a correct line range."""
    data = {
        "events": [
            {
                "event_type": "highlight",
                "start_ms": 1800,
                "end_ms": 2400,
                "start_line": 2,
                "end_line": 4,
            }
        ]
    }

    timeline = Timeline.model_validate(data)

    assert timeline.events[0].event_type == "highlight"


def test_valid_scroll_event_passes() -> None:
    """Validate a timed scroll event with a positive target line."""
    data = {
        "events": [
            {
                "event_type": "scroll",
                "start_ms": 2400,
                "end_ms": 3000,
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
                "start_ms": 0,
                "end_ms": 1200,
                "code": "def add(a, b):\n    return a + b",
            },
            {
                "event_type": "highlight",
                "start_ms": 1200,
                "end_ms": 1800,
                "start_line": 2,
                "end_line": 2,
            },
            {
                "event_type": "run",
                "start_ms": 1800,
                "end_ms": 2600,
                "command": "add(2, 3)",
                "expected_output": "5",
            },
            {
                "event_type": "scroll",
                "start_ms": 2600,
                "end_ms": 3200,
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
                "start_ms": 0,
                "end_ms": 1000,
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
                "start_ms": 0,
                "end_ms": 1000,
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
                "start_ms": 0,
                "end_ms": 1000,
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
                "start_ms": 0,
                "end_ms": 1000,
                "target_line": 0,
            }
        ]
    }

    with pytest.raises(ValidationError):
        Timeline.model_validate(data)


def test_end_ms_must_be_greater_than_start_ms() -> None:
    """Reject events where end_ms is not greater than start_ms."""
    data = {
        "events": [
            {
                "event_type": "type",
                "start_ms": 1000,
                "end_ms": 1000,
                "code": "x = 1",
            }
        ]
    }

    with pytest.raises(ValidationError):
        Timeline.model_validate(data)


def test_negative_start_ms_fails() -> None:
    """Reject events with negative start_ms."""
    data = {
        "events": [
            {
                "event_type": "type",
                "start_ms": -1,
                "end_ms": 1000,
                "code": "x = 1",
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


def test_raw_json_can_be_validated_directly() -> None:
    """Validate raw JSON directly using Pydantic model_validate_json."""
    raw_json = json.dumps(
        {
            "events": [
                {
                    "event_type": "type",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "code": "x = 1",
                }
            ]
        }
    )

    timeline = Timeline.model_validate_json(raw_json)

    assert timeline.events[0].event_type == "type"
