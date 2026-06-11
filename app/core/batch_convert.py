"""Batch converter for narration scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.llm_client import LLMClient, LLMClientError
from app.core.prompt_chain import convert_script_to_timeline


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
    """Validate one script item from sample_scripts.json."""
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

    if not isinstance(script, str) or not script.strip():
        raise ValueError("'script' must be a non-empty string.")

    return item_id, script


def batch_convert(
    input_path: Path,
    output_path: Path,
    failures_path: Path,
    max_repair_attempts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a batch of narration scripts into validated timelines."""
    raw_items = load_json_file(input_path)

    if not isinstance(raw_items, list):
        raise ValueError("Input file must contain a JSON list.")

    try:
        llm_client = LLMClient()
    except LLMClientError as error:
        raise RuntimeError(f"Could not initialize LLM client: {error}") from error

    generated_timelines: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for item in raw_items:
        item_id = "unknown_id"

        try:
            item_id, script = validate_script_item(item)

            timeline = convert_script_to_timeline(
                script=script,
                llm_client=llm_client,
                max_repair_attempts=max_repair_attempts,
            )

            generated_timelines.append(
                {
                    "id": item_id,
                    "timeline": timeline.model_dump(),
                }
            )

            print(f"[ok] {item_id}")

        except Exception as error:
            failures.append(
                {
                    "id": item_id,
                    "error": str(error),
                    "input": item,
                }
            )

            print(f"[failed] {item_id}: {error}")

    write_json_file(output_path, generated_timelines)
    write_json_file(failures_path, failures)

    return generated_timelines, failures


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description="Batch convert narration scripts into validated timeline JSON.")

    parser.add_argument(
        "--input",
        default="tests/fixtures/sample_scripts.json",
        help="Input JSON file containing narration scripts.",
    )

    parser.add_argument(
        "--output",
        default="results/generated_timelines.json",
        help="Output JSON file for successfully generated timelines.",
    )

    parser.add_argument(
        "--failures",
        default="results/failures.json",
        help="Output JSON file for failed conversions.",
    )

    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=2,
        help="Maximum repair attempts per script.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    generated, failures = batch_convert(
        input_path=Path(args.input),
        output_path=Path(args.output),
        failures_path=Path(args.failures),
        max_repair_attempts=args.max_repair_attempts,
    )

    print("")
    print("Batch conversion complete.")
    print(f"Successful timelines: {len(generated)}")
    print(f"Failures: {len(failures)}")
    print(f"Output file: {args.output}")
    print(f"Failures file: {args.failures}")


if __name__ == "__main__":
    main()
