"""Integration tests for fixtures and offline evaluation."""

from pathlib import Path

from src.eval_harness import evaluate_offline, format_report, write_report
from src.schemas import Timeline
from src.utils import load_json_file


def test_expected_timelines_fixture_is_valid() -> None:
    """All expected timeline fixtures should validate successfully."""
    fixture_path = Path("fixtures/expected_timelines.json")
    items = load_json_file(fixture_path)

    for item in items:
        Timeline.model_validate(item["timeline"])


def test_invalid_expected_timelines_fixture_fails_offline() -> None:
    """The intentionally invalid fixture should fail schema validation."""
    fixture_path = Path("fixtures/expected_timelines_invalid.json")
    summary = evaluate_offline(fixture_path)

    assert summary.total_items == 5
    assert summary.valid_items == 0
    assert summary.failed_items == 5
    assert summary.schema_conformance_rate == 0.0


def test_offline_eval_writes_successful_report() -> None:
    """Offline evaluation should produce a readable report file."""
    expected_path = Path("fixtures/expected_timelines.json")
    report_path = Path("results/test_eval_report.txt")

    try:
        summary = evaluate_offline(expected_path)
        report_text = format_report(summary)
        write_report(report_text, report_path)

        saved_report = report_path.read_text(encoding="utf-8")

        assert report_path.exists()
        assert "Structured Outputs Evaluation Report" in saved_report
        assert "Schema conformance rate: 100.00%" in saved_report
        assert "- type: yes" in saved_report
        assert "- run: yes" in saved_report
        assert "- highlight: yes" in saved_report
        assert "- scroll: yes" in saved_report

    finally:
        if report_path.exists():
            report_path.unlink()