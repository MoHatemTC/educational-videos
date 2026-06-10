"""Validation and repair helpers for timeline JSON outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src.schemas import Timeline


class TimelineValidationError(Exception):
    """Raised when timeline validation and repair both fail."""


@dataclass(frozen=True)
class RepairResult:
    """Validated timeline plus the number of repair rounds used."""

    timeline: Timeline
    repair_rounds: int


def validate_raw_timeline(raw_output: str) -> Timeline:
    """Validate raw JSON text directly against the Timeline schema."""
    return Timeline.model_validate_json(raw_output)


def load_repair_prompt_template() -> str:
    """Load the versioned repair prompt template from disk."""
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "repair_v1.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing repair prompt file: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8")

    if not prompt.strip():
        raise ValueError(f"Repair prompt file is empty: {prompt_path}")

    return prompt


def build_repair_prompt(
    bad_output: str,
    error_message: str,
    source_context: str = "",
) -> str:
    """Build the repair prompt sent to the LLM."""
    template = load_repair_prompt_template()
    schema_json = json.dumps(Timeline.model_json_schema(), indent=2)

    return (
        template.replace("{schema_json}", schema_json)
        .replace("{bad_output}", bad_output)
        .replace("{error_message}", error_message)
        .replace("{source_context}", source_context)
    )


def validate_or_repair_with_stats(
    raw_output: str,
    llm_client,
    max_repair_attempts: int = 2,
    source_context: str = "",
) -> RepairResult:
    """Validate LLM output and repair it while tracking repair rounds."""
    current_output = raw_output
    last_error = None

    for attempt in range(max_repair_attempts + 1):
        try:
            return RepairResult(
                timeline=validate_raw_timeline(current_output),
                repair_rounds=attempt,
            )

        except ValidationError as error:
            last_error = f"Schema validation failed: {error}"

        if attempt >= max_repair_attempts:
            break

        repair_prompt = build_repair_prompt(
            bad_output=current_output,
            error_message=last_error,
            source_context=source_context,
        )

        current_output = llm_client.generate_json(repair_prompt)

    raise TimelineValidationError(
        f"Timeline could not be validated after "
        f"{max_repair_attempts} repair attempt(s).\n\n"
        f"Last error:\n{last_error}\n\n"
        f"Last output:\n{current_output}"
    )


def validate_or_repair(
    raw_output: str,
    llm_client,
    max_repair_attempts: int = 2,
    source_context: str = "",
) -> Timeline:
    """Validate an LLM timeline output, repairing it if necessary."""
    return validate_or_repair_with_stats(
        raw_output=raw_output,
        llm_client=llm_client,
        max_repair_attempts=max_repair_attempts,
        source_context=source_context,
    ).timeline
