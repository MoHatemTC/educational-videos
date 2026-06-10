"""Prompt-chain helpers for converting narration scripts into timelines."""

import json
from pathlib import Path

from src.schemas import Timeline
from src.validate_repair import validate_or_repair


def load_prompt(path: str | Path) -> str:
    """Load a prompt template from disk."""
    prompt = Path(path).read_text(encoding="utf-8")

    if not prompt.strip():
        raise ValueError(f"Prompt file is empty: {path}")

    return prompt


def segment_script(script: str, llm_client) -> list[dict]:
    """Segment narration into a JSON array of action segments."""
    template = load_prompt("prompts/segment_script_v1.txt")
    prompt = template.replace("{script}", script)

    raw_output = llm_client.generate_json(prompt)
    segments = json.loads(raw_output)

    if not isinstance(segments, list):
        raise ValueError("Segmentation output must be a JSON array.")

    return segments


def synthesize_timeline(segments: dict, llm_client) -> str:
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


def convert_script_to_timeline(
        script: str,
        llm_client,
        max_repair_attempts: int = 2,
) -> Timeline:
    """Convert a narration script into a validated Timeline.

    The pipeline segments the script, synthesizes timeline JSON with the LLM,
    then validates or repairs the output against the Timeline schema.

    Args:
        script: Original narration script.
        llm_client: Client object with a generate_json method.
        max_repair_attempts: Maximum number of repair retries.

    Returns:
        A validated Timeline object.
    """
    segments = segment_script(script, llm_client)
    raw_timeline = synthesize_timeline(segments, llm_client)

    source_context = (
        "Original narration script:\n"
        f"{script}\n\n"
        "Segmented script actions:\n"
        f"{json.dumps(segments, indent=2)}"
    )

    return validate_or_repair(
        raw_output=raw_timeline,
        llm_client=llm_client,
        max_repair_attempts=max_repair_attempts,
        source_context=source_context,
    )
