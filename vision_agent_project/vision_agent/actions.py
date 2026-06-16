"""vision_agent.actions.

Typed data-classes for every browser action the VLM can produce.

Supported actions
-----------------
  click      — left-click at (x, y)
  type       — keyboard input of a text string
  scroll     — wheel-scroll at (x, y) by delta_y pixels
  navigate   — hard navigate to a URL
  wait       — pause execution for `seconds`
  done       — terminal action; carries the final result text

Parsing
-------
The VLM is instructed to respond with a single JSON block:

    {"action": "click",  "x": 412,  "y": 308}
    {"action": "type",   "text": "Computer vision"}
    {"action": "scroll", "x": 760,  "y": 400, "delta_y": 300}
    {"action": "navigate","url": "https://example.com"}
    {"action": "wait",   "seconds": 2}
    {"action": "done",   "result": "First paragraph text …"}

`Action.from_dict` converts a raw dict into a strongly-typed `Action`.
`Action.from_vlm_text` extracts the JSON block from free-form VLM output.
"""

from __future__ import annotations

import json
import re
from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
from typing import (
    Any,
    Optional,
)


class ActionType(str, Enum):
    """Supported browser actions."""

    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    WAIT = "wait"
    DONE = "done"


@dataclass
class Action:
    """A single browser action produced by the Reason → Plan stages."""

    action_type: ActionType

    # click / scroll
    x: Optional[int] = None
    y: Optional[int] = None

    # scroll
    delta_y: int = 300

    # type
    text: Optional[str] = None

    # navigate
    url: Optional[str] = None

    # wait
    seconds: float = 1.0

    # done
    result: Optional[str] = None

    # raw payload kept for logging / debugging
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        """Build an Action from a parsed JSON dict."""
        try:
            action_type = ActionType(data["action"].lower())
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown action type in payload: {data!r}") from exc

        return cls(
            action_type=action_type,
            x=data.get("x"),
            y=data.get("y"),
            delta_y=int(data.get("delta_y", 300)),
            text=data.get("text"),
            url=data.get("url"),
            seconds=float(data.get("seconds", 1.0)),
            result=data.get("result"),
            raw=data,
        )

    @classmethod
    def from_vlm_text(cls, text: str) -> "Action":
        """Extract the first JSON object from free-form VLM output and parse it.

        The model may wrap the JSON in markdown code fences or precede it with
        reasoning prose – this method handles both cases.

        Raises:
        ------
        ValueError
            If no parseable JSON action block is found.
        """
        # 1. Try a bare JSON object anywhere in the text
        json_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
        for match in json_pattern.finditer(text):
            try:
                data = json.loads(match.group())
                if "action" in data:
                    return cls.from_dict(data)
            except json.JSONDecodeError:
                continue

        # 2. Try fenced code block  ```json … ```
        fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
        for match in fence_pattern.finditer(text):
            try:
                data = json.loads(match.group(1))
                if "action" in data:
                    return cls.from_dict(data)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"No valid action JSON found in VLM output.\n--- Raw output ---\n{text}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def is_terminal(self) -> bool:
        """Return True if this action ends the agent loop."""
        return self.action_type == ActionType.DONE

    def __repr__(self) -> str:
        """Return a debug representation of the action."""
        parts = [f"Action(type={self.action_type.value!r}"]
        if self.x is not None:
            parts.append(f"x={self.x}, y={self.y}")
        if self.text is not None:
            parts.append(f"text={self.text!r}")
        if self.url is not None:
            parts.append(f"url={self.url!r}")
        if self.result is not None:
            parts.append(f"result={self.result[:60]!r}…")
        return ", ".join(parts) + ")"
