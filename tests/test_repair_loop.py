"""Tests for timeline validation and repair behavior."""

from typing import Any

import pytest

from app.core.schemas import Timeline
from app.core.validate_repair import TimelineValidationError, validate_or_repair, validate_or_repair_with_stats


class FakeLLMClient:
    """Fake LLM client used to avoid real API calls in tests."""

    def __init__(self, responses: list[str]) -> None:
        """Store fake responses returned by generate_json."""
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        schema_name: str = "structured_output",
    ) -> str:
        """Return the next fake response."""
        _ = schema
        _ = schema_name

        self.calls += 1
        self.prompts.append(prompt)

        if not self.responses:
            return "{}"

        return self.responses.pop(0)


def test_valid_json_returns_without_repair() -> None:
    """Valid timeline JSON should pass without calling the LLM."""
    raw_output = """
    {
      "events": [
        {
          "event_type": "type",
          "start_ms": 0,
          "end_ms": 1000,
          "code": "print('hello')"
        }
      ]
    }
    """

    fake_client = FakeLLMClient(responses=[])

    timeline = validate_or_repair(raw_output, fake_client)

    assert isinstance(timeline, Timeline)
    assert fake_client.calls == 0


def test_malformed_json_triggers_repair() -> None:
    """Malformed JSON should trigger one repair call."""
    bad_output = "{ bad json"

    repaired_output = """
    {
      "events": [
        {
          "event_type": "type",
          "start_ms": 0,
          "end_ms": 1000,
          "code": "print('fixed')"
        }
      ]
    }
    """

    fake_client = FakeLLMClient(responses=[repaired_output])

    timeline = validate_or_repair(
        raw_output=bad_output,
        llm_client=fake_client,
        source_context="Original segment: type a print statement.",
    )

    assert timeline.events[0].event_type == "type"
    assert fake_client.calls == 1
    assert "Original segment: type a print statement." in fake_client.prompts[0]


def test_schema_invalid_json_triggers_repair() -> None:
    """Schema-invalid JSON should trigger repair."""
    bad_output = """
    {
      "events": [
        {
          "event_type": "highlight",
          "start_ms": 1000,
          "end_ms": 1000,
          "start_line": 5,
          "end_line": 2
        }
      ]
    }
    """

    repaired_output = """
    {
      "events": [
        {
          "event_type": "highlight",
          "start_ms": 1000,
          "end_ms": 1600,
          "start_line": 2,
          "end_line": 5
        }
      ]
    }
    """

    fake_client = FakeLLMClient(responses=[repaired_output])

    timeline = validate_or_repair(
        raw_output=bad_output,
        llm_client=fake_client,
        source_context="Original segment: highlight the return block.",
    )

    assert timeline.events[0].event_type == "highlight"
    assert fake_client.calls == 1


def test_repeated_invalid_repairs_raise_error() -> None:
    """Repeated bad repairs should raise TimelineValidationError."""
    bad_output = """
    {
      "events": [
        {
          "event_type": "zoom",
          "start_ms": 0,
          "end_ms": 1000,
          "target_line": 5
        }
      ]
    }
    """

    invalid_repair = """
    {
      "events": [
        {
          "event_type": "zoom",
          "start_ms": 0,
          "end_ms": 1000,
          "target_line": 5
        }
      ]
    }
    """

    fake_client = FakeLLMClient(
        responses=[
            invalid_repair,
            invalid_repair,
        ]
    )

    with pytest.raises(TimelineValidationError):
        validate_or_repair(
            raw_output=bad_output,
            llm_client=fake_client,
            max_repair_attempts=2,
            source_context="Original segment: scroll to line five.",
        )

    assert fake_client.calls == 2


def test_repair_stats_report_repair_rounds() -> None:
    """Repair stats should report how many repair rounds were used."""
    bad_output = "{ bad json"
    repaired_output = """
    {
      "events": [
        {
          "event_type": "type",
          "start_ms": 0,
          "end_ms": 1000,
          "code": "x = 1"
        }
      ]
    }
    """
    fake_client = FakeLLMClient(responses=[repaired_output])

    result = validate_or_repair_with_stats(
        raw_output=bad_output,
        llm_client=fake_client,
        source_context="Original segment: type x equals one.",
    )

    assert result.timeline.events[0].event_type == "type"
    assert result.repair_rounds == 1
    assert "Original segment: type x equals one." in fake_client.prompts[0]
