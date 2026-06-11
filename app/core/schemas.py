"""Pydantic v2 schemas for validated code-animation timeline events."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt
from pydantic import model_validator


class StrictBaseModel(BaseModel):
    """Base model that rejects extra fields."""

    model_config = ConfigDict(extra="forbid")


class TimedEvent(StrictBaseModel):
    """Shared temporal fields for all animation events."""

    start_ms: NonNegativeInt
    end_ms: PositiveInt

    @model_validator(mode="after")
    def validate_time_range(self):
        """Ensure each event has a positive duration."""
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class TypeEvent(TimedEvent):
    """Represents typing code onto the screen."""

    event_type: Literal["type"]
    code: str


class RunEvent(TimedEvent):
    """Represents running code or showing command output."""

    event_type: Literal["run"]
    command: str
    expected_output: str | None = None


class HighlightEvent(TimedEvent):
    """Represents highlighting one or more code lines."""

    event_type: Literal["highlight"]
    start_line: PositiveInt
    end_line: PositiveInt

    @model_validator(mode="after")
    def validate_line_range(self):
        """Ensure highlight line range is valid."""
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ScrollEvent(TimedEvent):
    """Represents scrolling the code viewport to a line."""

    event_type: Literal["scroll"]
    target_line: PositiveInt


TimelineEvent = Annotated[
    Union[TypeEvent, RunEvent, HighlightEvent, ScrollEvent],
    Field(discriminator="event_type"),
]


class Timeline(StrictBaseModel):
    """Top-level timeline model."""

    events: list[TimelineEvent] = Field(min_length=1)


def timeline_json_schema() -> dict:
    """Return the JSON Schema for the Timeline model."""
    return Timeline.model_json_schema()


def validate_timeline_dict(data: dict) -> Timeline:
    """Validate a Python dictionary as a Timeline."""
    return Timeline.model_validate(data)


def validate_timeline_json(raw_json: str) -> Timeline:
    """Validate a raw JSON string directly as a Timeline."""
    return Timeline.model_validate_json(raw_json)
