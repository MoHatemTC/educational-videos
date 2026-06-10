"""Tests for timeline validation and repair behavior."""

import pytest

from src.schemas import Timeline
from src.validate_repair import TimelineValidationError, validate_or_repair


class FakeLLMClient:
    """Fake LLM client used to avoid real API calls in tests."""

    def __init__(self, responses: list[str]) -> None:
        """Store fake responses returned by generate_json."""
        self.responses = responses
        self.calls = 0

    def generate_json(self, prompt: str) -> str:
        """Return the next fake response."""
        self.calls += 1

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
          "code": "print('fixed')"
        }
      ]
    }
    """

    fake_client = FakeLLMClient(responses=[repaired_output])

    timeline = validate_or_repair(bad_output, fake_client)

    assert timeline.events[0].event_type == "type"
    assert fake_client.calls == 1


def test_schema_invalid_json_triggers_repair() -> None:
    """Schema-invalid JSON should trigger repair."""
    bad_output = """
    {
      "events": [
        {
          "event_type": "highlight",
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
          "start_line": 2,
          "end_line": 5
        }
      ]
    }
    """

    fake_client = FakeLLMClient(responses=[repaired_output])

    timeline = validate_or_repair(bad_output, fake_client)

    assert timeline.events[0].event_type == "highlight"
    assert fake_client.calls == 1


def test_repeated_invalid_repairs_raise_error() -> None:
    """Repeated bad repairs should raise TimelineValidationError."""
    bad_output = """
    {
      "events": [
        {
          "event_type": "zoom",
          "target_line": 5
        }
      ]
    }
    """

    fake_client = FakeLLMClient(
        responses=[
            '{"events": [{"event_type": "zoom", "target_line": 5}]}',
            '{"events": [{"event_type": "zoom", "target_line": 5}]}',
        ]
    )

    with pytest.raises(TimelineValidationError):
        validate_or_repair(
            raw_output=bad_output,
            llm_client=fake_client,
            max_repair_attempts=2,
        )

    assert fake_client.calls == 2