"""
Session log parser.

Accepts a raw VLM browser-action session log (JSON Lines or JSON array)
and returns a list of normalised SessionEvent dicts ready for the mapper.

Session log format (each event):
{
  "timestamp": <float | ISO-8601 string>,  # seconds since epoch OR datetime
  "action": "click" | "type" | "scroll" | "navigate" | ...,
  "x": <float>,                            # cursor X (optional)
  "y": <float>,                            # cursor Y (optional)
  "value": <any>,                          # typed text, URL, key name, …
  "target": <str>,                         # CSS selector / element label
  "screenshot": <str>,                     # file path or base64 ref
  "meta": { ... }                          # extra info
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from vlm_render_mapper.schema import ActionType


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

RawEvent = dict[str, Any]
SessionEvent = dict[str, Any]  # normalised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACTION_ALIASES: dict[str, str] = {
    # common VLM / Playwright aliases → ActionType values
    "leftclick": "click",
    "left_click": "click",
    "mouseclick": "click",
    "dblclick": "double_click",
    "doubleclick": "double_click",
    "rightclick": "right_click",
    "mouseover": "hover",
    "mousemove": "hover",
    "input": "type",
    "keyboard": "key_press",
    "keydown": "key_press",
    "keyup": "key_press",
    "goto": "navigate",
    "load": "page_load",
    "pageload": "page_load",
    "snap": "screenshot",
    "capture": "screenshot",
    "sleep": "wait",
    "pause": "wait",
    "delay": "wait",
}

_VALID_ACTIONS: frozenset[str] = frozenset(a.value for a in ActionType)


def _normalise_action(raw: str) -> str:
    lowered = raw.lower().strip().replace("-", "_")
    resolved = _ACTION_ALIASES.get(lowered, lowered)
    if resolved not in _VALID_ACTIONS:
        # best-effort fall-back
        return "wait"
    return resolved


def _parse_timestamp(ts: Any) -> float:
    """Return seconds since epoch as float."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        # try ISO 8601
        try:
            dt = datetime.fromisoformat(ts.rstrip("Z"))
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
        # try plain float string
        try:
            return float(ts)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


def _coerce_coordinate(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class SessionParseError(ValueError):
    """Raised when the session log cannot be parsed."""


class SessionParser:
    """Parse a raw session log file into normalised SessionEvent list."""

    def parse_file(self, path: Union[str, Path]) -> list[SessionEvent]:
        path = Path(path)
        if not path.exists():
            raise SessionParseError(f"Session file not found: {path}")
        text = path.read_text(encoding="utf-8")
        return self.parse_text(text, source=str(path))

    def parse_text(self, text: str, source: str = "<string>") -> list[SessionEvent]:
        text = text.strip()
        if not text:
            raise SessionParseError("Session log is empty")

        raw_events = self._load_json(text, source)
        return [self._normalise(i, ev) for i, ev in enumerate(raw_events)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(text: str, source: str) -> list[RawEvent]:
        """Support JSON array OR JSON Lines (one object per line)."""
        # Try JSON array first
        if text.startswith("["):
            try:
                data = json.loads(text)
                if not isinstance(data, list):
                    raise SessionParseError(f"{source}: top-level JSON must be an array")
                return data
            except json.JSONDecodeError as exc:
                raise SessionParseError(f"{source}: invalid JSON array — {exc}") from exc

        # JSON Lines
        events: list[RawEvent] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise SessionParseError(f"{source}:{lineno}: each line must be a JSON object")
                events.append(obj)
            except json.JSONDecodeError as exc:
                raise SessionParseError(f"{source}:{lineno}: invalid JSON — {exc}") from exc
        if not events:
            raise SessionParseError(f"{source}: no valid events found")
        return events

    def _normalise(self, index: int, raw: RawEvent) -> SessionEvent:
        if not isinstance(raw, dict):
            raise SessionParseError(f"Event #{index} is not a JSON object")

        # --- timestamp ---
        # Use explicit key lookup to avoid falsy 0.0 being skipped by `or`
        ts_raw = next(
            (raw[k] for k in ("timestamp", "ts", "time") if k in raw),
            None,
        )
        if ts_raw is None:
            raise SessionParseError(f"Event #{index} missing 'timestamp' field")
        try:
            timestamp = _parse_timestamp(ts_raw)
        except ValueError as exc:
            raise SessionParseError(f"Event #{index}: {exc}") from exc

        # --- action ---
        action_raw = next(
            (raw[k] for k in ("action", "type", "event_type") if k in raw),
            "wait",
        )
        action = _normalise_action(str(action_raw))

        # --- coordinates (explicit lookup preserves 0.0 values) ---
        x = _coerce_coordinate(next((raw[k] for k in ("x", "clientX", "pageX") if k in raw), None))
        y = _coerce_coordinate(next((raw[k] for k in ("y", "clientY", "pageY") if k in raw), None))

        # --- scroll delta ---
        scroll_delta_x = _coerce_coordinate(
            next((raw[k] for k in ("deltaX", "scroll_x", "scrollX") if k in raw), None)
        )
        scroll_delta_y = _coerce_coordinate(
            next((raw[k] for k in ("deltaY", "scroll_y", "scrollY") if k in raw), None)
        )

        # --- drag endpoints ---
        drag_start = raw.get("drag_start") or raw.get("dragStart")
        drag_end = raw.get("drag_end") or raw.get("dragEnd")

        # --- screenshot ---
        screenshot = (
            raw.get("screenshot")
            or raw.get("screenshot_path")
            or raw.get("screenshotRef")
            or f"frame_{index:06d}.png"
        )

        # --- value / target ---
        value = raw.get("value") or raw.get("text") or raw.get("key") or raw.get("url")
        target = raw.get("target") or raw.get("selector") or raw.get("element")

        return {
            "index": index,
            "timestamp": timestamp,
            "action": action,
            "x": x,
            "y": y,
            "scroll_delta_x": scroll_delta_x,
            "scroll_delta_y": scroll_delta_y,
            "drag_start": drag_start,
            "drag_end": drag_end,
            "value": value,
            "target": target,
            "screenshot": screenshot,
            "meta": raw.get("meta") or raw.get("metadata") or {},
            "_raw": raw,
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def parse_session_file(path: Union[str, Path]) -> list[SessionEvent]:
    return SessionParser().parse_file(path)


def parse_session_text(text: str) -> list[SessionEvent]:
    return SessionParser().parse_text(text)
