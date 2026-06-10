"""Evaluation harness for the structured-outputs project."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.prompt_chain import convert_script_to_timeline_with_stats
from src.schemas import Timeline
from src.validate_repair import validate_or_repair_with_stats

REQUIRED_EVENT_TYPES = {"type", "run", "highlight", "scroll"}


@dataclass
class TimelineEvalResult:
    """Result of evaluating one script/timeline item."""

    item_id: str
    is_valid: bool
    error_message: str | None = None
    event_types: set[str] | None = None
    repair_rounds: int = 0
    sequence_correct: bool | None = None


@dataclass
class EvaluationSummary:
    """Aggregated evaluation result."""

    mode: str
    total_items: int
    valid_items: int
    failed_items: int
    schema_conformance_rate: float
    mean_repair_rounds: float
    sequence_level_accuracy: float | None
    event_type_coverage: dict[str, bool]
    item_results: list[TimelineEvalResult]


def load_json_file(path: Path) -> Any:
    """Load and parse a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


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

    if not isinstance(script, str):
        raise ValueError("'script' must be a string.")

    if not script.strip():
        raise ValueError("'script' must not be empty.")

    return item_id, script


def extract_event_types(timeline: Timeline) -> set[str]:
    """Return all event types used inside a validated Timeline."""
    return {event.event_type for event in timeline.events}


def event_sequence(timeline: Timeline) -> list[str]:
    """Return the ordered event-type sequence for a timeline."""
    return [event.event_type for event in timeline.events]


def load_expected_sequences(expected_path: Path) -> dict[str, list[str]]:
    """Load expected event-type sequences by item id."""
    if not expected_path.exists():
        return {}

    raw_items = load_json_file(expected_path)
    if not isinstance(raw_items, list):
        return {}

    expected_sequences: dict[str, list[str]] = {}

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        try:
            item_id = str(item["id"])
            timeline = Timeline.model_validate(item["timeline"])
        except (KeyError, ValidationError):
            continue

        expected_sequences[item_id] = event_sequence(timeline)

    return expected_sequences


def validate_expected_timeline_item(item: dict[str, Any]) -> TimelineEvalResult:
    """Validate one item from expected_timelines.json."""
    item_id = str(item.get("id", "unknown_id"))

    try:
        timeline = Timeline.model_validate(item["timeline"])

        return TimelineEvalResult(
            item_id=item_id,
            is_valid=True,
            event_types=extract_event_types(timeline),
            repair_rounds=0,
            sequence_correct=True,
        )

    except KeyError as error:
        return TimelineEvalResult(
            item_id=item_id,
            is_valid=False,
            error_message=f"Missing required key: {error}",
            event_types=set(),
            sequence_correct=False,
        )

    except ValidationError as error:
        return TimelineEvalResult(
            item_id=item_id,
            is_valid=False,
            error_message=str(error),
            event_types=set(),
            sequence_correct=False,
        )


def calculate_event_type_coverage(
    item_results: list[TimelineEvalResult],
) -> dict[str, bool]:
    """Check whether valid timelines covered all required event types."""
    seen_event_types: set[str] = set()

    for result in item_results:
        if result.is_valid and result.event_types:
            seen_event_types.update(result.event_types)

    return {
        event_type: event_type in seen_event_types
        for event_type in sorted(REQUIRED_EVENT_TYPES)
    }


def calculate_sequence_level_accuracy(
    item_results: list[TimelineEvalResult],
) -> float | None:
    """Calculate percentage of items with the correct event-type sequence."""
    comparable_results = [
        result for result in item_results if result.sequence_correct is not None
    ]

    if not comparable_results:
        return None

    correct_items = sum(result.sequence_correct for result in comparable_results)
    return (correct_items / len(comparable_results)) * 100


def build_summary(
    mode: str,
    item_results: list[TimelineEvalResult],
) -> EvaluationSummary:
    """Build the final evaluation summary from per-item results."""
    total_items = len(item_results)
    valid_items = sum(result.is_valid for result in item_results)
    failed_items = total_items - valid_items

    schema_conformance_rate = (valid_items / total_items) * 100 if total_items else 0.0
    mean_repair_rounds = (
        sum(result.repair_rounds for result in item_results) / total_items
        if total_items
        else 0.0
    )

    return EvaluationSummary(
        mode=mode,
        total_items=total_items,
        valid_items=valid_items,
        failed_items=failed_items,
        schema_conformance_rate=schema_conformance_rate,
        mean_repair_rounds=mean_repair_rounds,
        sequence_level_accuracy=calculate_sequence_level_accuracy(item_results),
        event_type_coverage=calculate_event_type_coverage(item_results),
        item_results=item_results,
    )


def evaluate_offline(expected_path: Path) -> EvaluationSummary:
    """Validate already-written expected timelines without calling the LLM."""
    raw_items = load_json_file(expected_path)

    if not isinstance(raw_items, list):
        raise ValueError("Expected timelines file must contain a JSON list.")

    item_results = [validate_expected_timeline_item(item) for item in raw_items]

    return build_summary(mode="offline", item_results=item_results)


def evaluate_online(
    input_path: Path,
    max_repair_attempts: int,
    expected_path: Path | None = None,
) -> EvaluationSummary:
    """Evaluate sample scripts through the live LLM pipeline."""
    from src.llm_client import LLMClient, LLMClientError

    raw_items = load_json_file(input_path)

    if not isinstance(raw_items, list):
        raise ValueError("Sample scripts file must contain a JSON list.")

    expected_sequences = load_expected_sequences(expected_path) if expected_path else {}

    try:
        llm_client = LLMClient()
    except LLMClientError as error:
        raise RuntimeError(f"Could not initialize LLM client: {error}") from error

    item_results: list[TimelineEvalResult] = []

    for item in raw_items:
        try:
            item_id, script = validate_script_item(item)
            timeline, repair_rounds = convert_script_to_timeline_with_stats(
                script=script,
                llm_client=llm_client,
                max_repair_attempts=max_repair_attempts,
            )

            predicted_sequence = event_sequence(timeline)
            expected_sequence = expected_sequences.get(item_id)
            sequence_correct = (
                predicted_sequence == expected_sequence
                if expected_sequence is not None
                else None
            )

            item_results.append(
                TimelineEvalResult(
                    item_id=item_id,
                    is_valid=True,
                    event_types=extract_event_types(timeline),
                    repair_rounds=repair_rounds,
                    sequence_correct=sequence_correct,
                )
            )

        except Exception as error:
            item_id = (
                str(item.get("id", "unknown_id"))
                if isinstance(item, dict)
                else "unknown_id"
            )

            item_results.append(
                TimelineEvalResult(
                    item_id=item_id,
                    is_valid=False,
                    error_message=str(error),
                    event_types=set(),
                    sequence_correct=False,
                )
            )

    return build_summary(mode="online", item_results=item_results)


def evaluate_repair_expected(
    expected_path: Path,
    max_repair_attempts: int,
) -> EvaluationSummary:
    """Evaluate the validation-and-repair loop using broken timeline JSON."""
    from src.llm_client import LLMClient, LLMClientError

    raw_items = load_json_file(expected_path)

    if not isinstance(raw_items, list):
        raise ValueError("Expected timelines file must contain a JSON list.")

    try:
        llm_client = LLMClient()
    except LLMClientError as error:
        raise RuntimeError(f"Could not initialize LLM client: {error}") from error

    item_results: list[TimelineEvalResult] = []

    for item in raw_items:
        item_id = (
            str(item.get("id", "unknown_id")) if isinstance(item, dict) else "unknown_id"
        )

        try:
            if not isinstance(item, dict):
                raise ValueError("Timeline item must be a JSON object.")

            if "timeline" not in item:
                raise ValueError("Timeline item is missing 'timeline'.")

            bad_timeline_json = json.dumps(item["timeline"], indent=2)

            repair_result = validate_or_repair_with_stats(
                raw_output=bad_timeline_json,
                llm_client=llm_client,
                max_repair_attempts=max_repair_attempts,
                source_context=f"Repair fixture item id: {item_id}",
            )

            item_results.append(
                TimelineEvalResult(
                    item_id=item_id,
                    is_valid=True,
                    event_types=extract_event_types(repair_result.timeline),
                    repair_rounds=repair_result.repair_rounds,
                )
            )

        except Exception as error:
            item_results.append(
                TimelineEvalResult(
                    item_id=item_id,
                    is_valid=False,
                    error_message=str(error),
                    event_types=set(),
                    sequence_correct=False,
                )
            )

    return build_summary(mode="repair-online", item_results=item_results)


def format_report(summary: EvaluationSummary) -> str:
    """Convert the evaluation summary into a readable text report."""
    lines: list[str] = []

    lines.append("Structured Outputs Evaluation Report")
    lines.append("=" * 44)
    lines.append(f"Mode: {summary.mode}")
    lines.append(f"Total items: {summary.total_items}")
    lines.append(f"Valid timelines: {summary.valid_items}")
    lines.append(f"Failed timelines: {summary.failed_items}")
    lines.append(f"Schema conformance rate: {summary.schema_conformance_rate:.2f}%")
    lines.append(f"Mean repair rounds: {summary.mean_repair_rounds:.2f}")

    if summary.sequence_level_accuracy is None:
        lines.append("Sequence-level accuracy: n/a")
    else:
        lines.append(f"Sequence-level accuracy: {summary.sequence_level_accuracy:.2f}%")

    lines.append("")
    lines.append("Event type coverage:")

    for event_type, is_covered in summary.event_type_coverage.items():
        status = "yes" if is_covered else "no"
        lines.append(f"- {event_type}: {status}")

    lines.append("")
    lines.append("Per-item results:")

    for result in summary.item_results:
        if result.is_valid:
            sequence_status = (
                "n/a" if result.sequence_correct is None else str(result.sequence_correct)
            )
            lines.append(
                f"- {result.item_id}: valid "
                f"(repair_rounds={result.repair_rounds}, "
                f"sequence_correct={sequence_status})"
            )
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
    parser = argparse.ArgumentParser(
        description="Evaluate schema conformance of generated timeline JSON."
    )

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
        help="Path to expected timelines JSON file for evaluation.",
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
        help="Maximum repair attempts during online evaluation.",
    )

    parser.add_argument(
        "--repair-expected",
        action="store_true",
        help="Send expected timeline JSON through the LLM repair loop.",
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

    elif args.repair_expected:
        summary = evaluate_repair_expected(
            expected_path=Path(args.expected),
            max_repair_attempts=args.max_repair_attempts,
        )

    else:
        summary = evaluate_online(
            input_path=Path(args.input),
            max_repair_attempts=args.max_repair_attempts,
            expected_path=Path(args.expected),
        )

    report_text = format_report(summary)
    write_report(report_text, Path(args.report))

    print(report_text)


if __name__ == "__main__":
    main()
