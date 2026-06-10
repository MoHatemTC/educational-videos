"""Pydantic schemas for code-animation timeline events."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, ConfigDict, PositiveInt, model_validator


class StrictBaseModel(BaseModel):
    """Base model that rejects unknown fields."""
    model_config = ConfigDict(extra="forbid")


class TypeEvent(StrictBaseModel):
    """Event for typing code into the animation."""
    event_type: Literal["type"]
    code: str


class RunEvent(StrictBaseModel):
    """Event for running a command."""
    event_type: Literal["run"]
    command: str
    expected_output: str | None = None


class HighlightEvent(StrictBaseModel):
    """Event for highlighting a code line range."""
    event_type: Literal["highlight"]
    start_line: PositiveInt
    end_line: PositiveInt

    @model_validator(mode="after")
    def validate_line_range(self):
        """Validate that the ending line is not before the starting line."""
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ScrollEvent(StrictBaseModel):
    """Event for scrolling to a target line."""
    event_type: Literal["scroll"]
    target_line: PositiveInt


TimelineEvent = Annotated[
    Union[TypeEvent, RunEvent, HighlightEvent, ScrollEvent],
    Field(discriminator="event_type"),
]


class Timeline(StrictBaseModel):
    """Validated list of timeline events."""
    events: list[TimelineEvent] = Field(min_length=1)


if __name__ == "__main__":
    from pydantic import ValidationError

    data = {
        "events": [
            {
                "event_type": "type",
                "code": "def add(a, b):\n    return a + b"
            },
            {
                "event_type": "highlight",
                "start_line": 2,
                "end_line": 2
            },
            {
                "event_type": "run",
                "command": "add(2, 3)",
                "expected_output": "5"
            },
            {
                "event_type": "scroll",
                "target_line": 1
            },
            {
                "event_type": "highlight",
                "start_line": 5,
                "end_line": 2
            }
        ]
    }
    try:
        timeline = Timeline.model_validate(data)
        print("Successfully validated Timeline")
    except ValidationError as e:
        print("Failed to validate Timeline")
        print(e)
