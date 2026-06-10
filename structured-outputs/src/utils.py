"""Shared utility functions for the structured-outputs project.

These helpers are intentionally kept small and dependency-light so they can be
used by eval_harness.py, batch_convert.py, and tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schemas import Timeline


def load_json_file(path: Path) -> Any:
    """Load and parse a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, data: Any) -> None:
    """Write data to disk as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def validate_script_item(item: dict[str, Any]) -> tuple[str, str]:
    """Validate one script item from sample_scripts.json.

    Expected format:
        {
            "id": "sample_001",
            "script": "Narration text..."
        }

    Returns:
        (item_id, script)
    """
    if not isinstance(item, dict):
        raise ValueError("Each script item must be a JSON object.")

    if "id" not in item:
        raise ValueError("Script item is missing 'id'.")

    if "script" not in item:
        raise ValueError("Script item is missing 'script'.")

    item_id = item["id"]
    script = item["script"]

    if not isinstance(item_id, str):
        raise ValueError("'id' must be a string.")

    if not isinstance(script, str):
        raise ValueError("'script' must be a string.")

    if not script.strip():
        raise ValueError("'script' must not be empty.")

    return item_id, script


def extract_event_types(timeline: Timeline) -> set[str]:
    """Return all event types used inside a validated Timeline."""
    return {event.event_type for event in timeline.events}