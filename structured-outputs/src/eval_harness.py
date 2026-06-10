"""Evaluation harness for the structured-outputs project.

This module measures schema-conformance rate.

It supports two modes:

1. Offline mode:
   - Does NOT call the LLM.
   - Loads fixtures/expected_timelines.json.
   - Validates each timeline against the Pydantic Timeline schema.

   Run:
       python -m src.eval_harness --offline

2. Online mode:
   - Uses LLMClient from src/llm_client.py.
   - Loads narration scripts from fixtures/sample_scripts.json.
   - Converts each script into a timeline through prompt_chain.py.
   - Validates the final returned Timeline object.

   Run:
       python -m src.eval_harness

Important:
    Online mode requires a valid .env file with:
        OPENAI_API_KEY
        OPENAI_MODEL
        OPENAI_BASE_URL
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.llm_client import LLMClient, LLMClientError
from src.prompt_chain import convert_script_to_timeline
from src.schemas import Timeline

REQUIRED_EVENT_TYPES = {"type", "run", "highlight", "scroll"}


@dataclass
class TimelineEvalResult:
    """Result of evaluating one script/timeline item."""

    item_id: str
    is_valid: bool
    error_message: str | None = None
    event_types: set[str] | None = None


@dataclass
class EvaluationSummary:
    """Aggregated evaluation result."""

    mode: str
    total_items: int
    valid_items: int
    failed_items: int
    schema_conformance_rate: float
    event_type_coverage: dict[str, bool]
    item_results: list[TimelineEvalResult]


def load_json_file(path: Path) -> Any:
    """Load and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def extract_event_types(timeline: Timeline) -> set[str]:
    """Return all event types used in a validated Timeline."""
    return {event.event_type for event in timeline.events}


def validate_expected_timeline_item(item: dict[str, Any]) -> TimelineEvalResult:
    # noinspection GrazieInspection
    """Validate one item from expected_timelines.json.

    Expected format:
        {
            "id": "sample_001",
            "timeline": {
                "events": [...]
            }
        }
    """
    item_id = str(item.get("id", "unknown_id"))

    try:
        timeline = Timeline.model_validate(item["timeline"])

        return TimelineEvalResult(
            item_id=item_id,
            is_valid=True,
            event_types=extract_event_types(timeline),
        )

    except KeyError as error:
        return TimelineEvalResult(
            item_id=item_id,
            is_valid=False,
            error_message=f"Missing required key: {error}",
            event_types=set(),
        )

    except ValidationError as error:
        return TimelineEvalResult(
            item_id=item_id,
            is_valid=False,
            error_message=str(error),
            event_types=set(),
        )


def validate_sample_script_item(item: dict[str, Any]) -> tuple[str, str]:
    """Validate one sample script item before sending it to the LLM.

    Expected format:
        {
            "id": "sample_001",
            "script": "..."
        }

    Returns:
        (item_id, script)

    Raises:
        ValueError: if the item is malformed.
    """
    if not isinstance(item, dict):
        raise ValueError("Sample script item must be a JSON object.")

    if "id" not in item:
        raise ValueError("Sample script item is missing 'id'.")

    if "script" not in item:
        raise ValueError("Sample script item is missing 'script'.")

    item_id = item["id"]
    script = item["script"]

    if not isinstance(item_id, str):
        raise ValueError("'id' must be a string.")

    if not isinstance(script, str):
        raise ValueError("'script' must be a string.")

    if not script.strip():
        raise ValueError("'script' must not be empty.")

    return item_id, script


def calculate_event_type_coverage(
    item_results: list[TimelineEvalResult],
) -> dict[str, bool]:
    """Check whether valid timelines covered all required event types.

    Invalid timelines do not count toward coverage.
    """
    seen_event_types: set[str] = set()

    for result in item_results:
        if result.is_valid and result.event_types:
            seen_event_types.update(result.event_types)

    return {event_type: event_type in seen_event_types for event_type in sorted(REQUIRED_EVENT_TYPES)}


def build_summary(
    mode: str,
    item_results: list[TimelineEvalResult],
) -> EvaluationSummary:
    """Build the final evaluation summary from per-item results."""
    total_items = len(item_results)
    valid_items = sum(result.is_valid for result in item_results)
    failed_items = total_items - valid_items

    if total_items == 0:
        schema_conformance_rate = 0.0
    else:
        schema_conformance_rate = (valid_items / total_items) * 100

    return EvaluationSummary(
        mode=mode,
        total_items=total_items,
        valid_items=valid_items,
        failed_items=failed_items,
        schema_conformance_rate=schema_conformance_rate,
        event_type_coverage=calculate_event_type_coverage(item_results),
        item_results=item_results,
    )


def evaluate_offline(expected_path: Path) -> EvaluationSummary:
    """Offline evaluation.

    This validates already-written expected timelines.
    It does not call OpenAI or Puter.
    """
    raw_items = load_json_file(expected_path)

    if not isinstance(raw_items, list):
        raise ValueError("Expected timelines file must contain a JSON list.")

    item_results = [validate_expected_timeline_item(item) for item in raw_items]

    return build_summary(mode="offline", item_results=item_results)


def evaluate_online(
    input_path: Path,
    max_repair_attempts: int,
) -> EvaluationSummary:
    """Online evaluation.

    This loads sample narration scripts, sends them through the LLM pipeline,
    and checks whether the final output conforms to the Timeline schema.

    This requires a working .env/API setup.
    """
    raw_items = load_json_file(input_path)

    if not isinstance(raw_items, list):
        raise ValueError("Sample scripts file must contain a JSON list.")

    try:
        llm_client = LLMClient()
    except LLMClientError as error:
        raise RuntimeError(f"Could not initialize LLM client: {error}") from error

    item_results: list[TimelineEvalResult] = []

    for item in raw_items:
        try:
            item_id, script = validate_sample_script_item(item)

            timeline = convert_script_to_timeline(
                script=script,
                llm_client=llm_client,
                max_repair_attempts=max_repair_attempts,
            )

            item_results.append(
                TimelineEvalResult(
                    item_id=item_id,
                    is_valid=True,
                    event_types=extract_event_types(timeline),
                )
            )

        except Exception as error:
            item_id = str(item.get("id", "unknown_id")) if isinstance(item, dict) else "unknown_id"

            item_results.append(
                TimelineEvalResult(
                    item_id=item_id,
                    is_valid=False,
                    error_message=str(error),
                    event_types=set(),
                )
            )

    return build_summary(mode="online", item_results=item_results)


def format_report(summary: EvaluationSummary) -> str:
    """Convert the evaluation summary into a readable text report."""
    # noinspection PyListCreation
    lines: list[str] = []

    lines.append("Structured Outputs Evaluation Report")
    lines.append("=" * 44)
    lines.append(f"Mode: {summary.mode}")
    lines.append(f"Total items: {summary.total_items}")
    lines.append(f"Valid timelines: {summary.valid_items}")
    lines.append(f"Failed timelines: {summary.failed_items}")
    lines.append(f"Schema conformance rate: {summary.schema_conformance_rate:.2f}%")

    lines.append("")
    lines.append("Event type coverage:")

    for event_type, is_covered in summary.event_type_coverage.items():
        status = "yes" if is_covered else "no"
        lines.append(f"- {event_type}: {status}")

    lines.append("")
    lines.append("Per-item results:")

    for result in summary.item_results:
        if result.is_valid:
            lines.append(f"- {result.item_id}: valid")
        else:
            lines.append(f"- {result.item_id}: invalid")
            lines.append(f"  Error: {result.error_message}")

    return "\n".join(lines)


def write_report(report_text: str, report_path: Path) -> None:
    """Write the evaluation report to disk."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the evaluation harness."""
    parser = argparse.ArgumentParser(description="Evaluate schema conformance of generated timeline JSON.")

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Validate expected timelines without calling the LLM.",
    )

    parser.add_argument(
        "--input",
        default="fixtures/sample_scripts.json",
        help="Path to sample scripts JSON file for online evaluation.",
    )

    parser.add_argument(
        "--expected",
        default="fixtures/expected_timelines.json",
        help="Path to expected timelines JSON file for offline evaluation.",
    )

    parser.add_argument(
        "--report",
        default="results/eval_report.txt",
        help="Path where the report should be written.",
    )

    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=2,
        help="Maximum number of repair attempts during online evaluation.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.offline:
        summary = evaluate_offline(
            expected_path=Path(args.expected),
        )
    else:
        summary = evaluate_online(
            input_path=Path(args.input),
            max_repair_attempts=args.max_repair_attempts,
        )

    report_text = format_report(summary)
    write_report(report_text, Path(args.report))

    print(report_text)


if __name__ == "__main__":
    main()
