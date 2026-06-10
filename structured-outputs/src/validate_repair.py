"""Validation and repair helpers for timeline JSON outputs."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import Timeline


class TimelineValidationError(Exception):
    """Raised when timeline validation and repair both fail."""


def parse_json(raw_output: str) -> dict[str, Any]:
    """Convert an LLM JSON string into a Python dictionary.

    Raises:
        JSONDecodeError: if the output is not valid JSON.
    """
    return json.loads(raw_output)


def validate_timeline_data(data: dict[str, Any]) -> Timeline:
    """Validate Python dictionary data against the Timeline Pydantic schema.

    Raises:
        ValidationError: if the data does not match the schema.
    """
    return Timeline.model_validate(data)


def validate_raw_timeline(raw_output: str) -> Timeline:
    """Parse raw JSON text and validate it as a Timeline."""
    data = parse_json(raw_output)
    return validate_timeline_data(data)


def load_repair_prompt_template() -> str:
    """Load the repair prompt template.

    Falls back to a built-in repair prompt if the file is missing.
    """
    prompt_path = Path("prompts") / "repair_v1.txt"

    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return """
You are repairing a JSON timeline for code animation events.

The output must be valid JSON only.
Do not include markdown.
Do not include explanations.
Do not include comments.

The JSON must match this schema:
{schema_json}

Original invalid output:
{bad_output}

Validation error:
{error_message}

Return the corrected JSON only.
""".strip()


def build_repair_prompt(
    bad_output: str,
    error_message: str,
) -> str:
    """Build the repair prompt sent to the LLM."""
    template = load_repair_prompt_template()

    schema_json = json.dumps(Timeline.model_json_schema(), indent=2)

    return template.format(
        schema_json=schema_json,
        bad_output=bad_output,
        error_message=error_message,
    )


def validate_or_repair(
    raw_output: str,
    llm_client,
    max_repair_attempts: int = 2,
) -> Timeline:
    """Validate an LLM timeline output.

    If the output is invalid JSON or does not match the Pydantic schema,
    this function asks the LLM to repair it, then validates again.

    Args:
        raw_output: Raw JSON text from the LLM.
        llm_client: Object with a generate_json(prompt: str) -> str method.
        max_repair_attempts: Number of repair attempts before failing.

    Returns:
        A validated Timeline object.

    Raises:
        TimelineValidationError: if validation still fails after repair attempts.
    """
    current_output = raw_output
    last_error = None

    for attempt in range(max_repair_attempts + 1):
        try:
            return validate_raw_timeline(current_output)

        except JSONDecodeError as error:
            last_error = f"Invalid JSON syntax: {error}"

        except ValidationError as error:
            last_error = f"Schema validation failed: {error}"

        if attempt >= max_repair_attempts:
            break

        repair_prompt = build_repair_prompt(
            bad_output=current_output,
            error_message=last_error,
        )

        current_output = llm_client.generate_json(repair_prompt)

    raise TimelineValidationError(
        f"Timeline could not be validated after "
        f"{max_repair_attempts} repair attempt(s).\n\n"
        f"Last error:\n{last_error}\n\n"
        f"Last output:\n{current_output}"
    )
