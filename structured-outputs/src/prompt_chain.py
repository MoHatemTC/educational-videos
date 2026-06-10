"""Prompt-chain helpers for converting narration scripts into timelines."""

import json
from pathlib import Path
from typing import Any

from src.schemas import Timeline
from src.validate_repair import validate_or_repair_with_stats

REQUIRED_SEGMENT_KEYS = {"segment_text", "event_type", "notes"}
VALID_EVENT_TYPES = {"type", "run", "highlight", "scroll"}


def load_prompt(path: str | Path) -> str:
    """Load a prompt template from disk."""
    prompt = Path(path).read_text(encoding="utf-8")

    if not prompt.strip():
        raise ValueError(f"Prompt file is empty: {path}")

    return prompt


def validate_segments(segments: Any) -> list[dict[str, str]]:
    """Validate segmentation output shape before timeline synthesis."""
    if not isinstance(segments, list):
        raise ValueError("Segmentation output must be a JSON array.")

    validated_segments: list[dict[str, str]] = []

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {index} must be a JSON object.")

        missing = REQUIRED_SEGMENT_KEYS - set(segment)
        if missing:
            raise ValueError(f"Segment {index} is missing keys: {sorted(missing)}")

        extra = set(segment) - REQUIRED_SEGMENT_KEYS
        if extra:
            raise ValueError(f"Segment {index} has extra keys: {sorted(extra)}")

        event_type = segment["event_type"]
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Segment {index} has invalid event_type: {event_type}")

        validated_segments.append(
            {
                "segment_text": str(segment["segment_text"]),
                "event_type": str(event_type),
                "notes": str(segment["notes"]),
            }
        )

    return validated_segments


def segment_script(script: str, llm_client) -> list[dict[str, str]]:
    """Segment narration into a JSON array of action segments."""
    template = load_prompt("prompts/segment_script_v1.txt")
    prompt = template.replace("{script}", script)

    raw_output = llm_client.generate_json(prompt)
    segments = json.loads(raw_output)

    return validate_segments(segments)


def synthesize_timeline(segments: list[dict[str, str]], llm_client) -> str:
    """Generate timeline JSON from script segments."""
    template = load_prompt("prompts/synthesize_timeline_v1.txt")

    prompt = template.replace(
        "{segments_json}",
        json.dumps(segments, indent=2),
    ).replace(
        "{schema_json}",
        json.dumps(Timeline.model_json_schema(), indent=2),
    )

    return llm_client.generate_json(
        prompt,
        schema=Timeline.model_json_schema(),
        schema_name="timeline",
    )


def build_source_context(script: str, segments: list[dict[str, str]]) -> str:
    """Build source context used by the repair prompt to prevent drift."""
    return (
        "Original narration script:\n"
        f"{script}\n\n"
        "Segmented script actions:\n"
        f"{json.dumps(segments, indent=2)}"
    )


def convert_script_to_timeline_with_stats(
    script: str,
    llm_client,
    max_repair_attempts: int = 2,
) -> tuple[Timeline, int]:
    """Convert a narration script into a timeline and repair-round count."""
    segments = segment_script(script, llm_client)
    raw_timeline = synthesize_timeline(segments, llm_client)

    repair_result = validate_or_repair_with_stats(
        raw_output=raw_timeline,
        llm_client=llm_client,
        max_repair_attempts=max_repair_attempts,
        source_context=build_source_context(script, segments),
    )

    return repair_result.timeline, repair_result.repair_rounds


def convert_script_to_timeline(
    script: str,
    llm_client,
    max_repair_attempts: int = 2,
) -> Timeline:
    """Convert a narration script into a validated Timeline."""
    return convert_script_to_timeline_with_stats(
        script=script,
        llm_client=llm_client,
        max_repair_attempts=max_repair_attempts,
    )[0]
