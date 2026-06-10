import json
from pathlib import Path

from src.schemas import Timeline
from src.validate_repair import validate_or_repair


def load_prompt(path: str | Path) -> str:
    prompt = Path(path).read_text(encoding="utf-8")

    if not prompt.strip():
        raise ValueError(f"Prompt file is empty: {path}")

    return prompt


def segment_script(script: str, llm_client) -> dict:
    template = load_prompt("prompts/segment_script_v1.txt")

    prompt = template.replace("{script}", script)

    raw_output = llm_client.generate_json(prompt)
    return json.loads(raw_output)


def synthesize_timeline(segments: dict, llm_client) -> str:
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
    segments = segment_script(script, llm_client)
    raw_timeline = synthesize_timeline(segments, llm_client)

    return validate_or_repair(
        raw_output=raw_timeline,
        llm_client=llm_client,
        max_repair_attempts=max_repair_attempts,
    )