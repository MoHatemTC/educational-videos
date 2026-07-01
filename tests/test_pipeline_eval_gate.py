"""Tests for pipeline hallucination evaluation and CI gating."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.services.pipeline import evaluation
from app.services.pipeline.evaluation import (
    PipelineEvalCase,
    evaluate_case,
    evaluate_cases,
    evaluate_pipeline_artifacts,
)
from evals.pipeline_eval import load_cases


def test_grounded_case_passes_local_hallucination_gate() -> None:
    """A grounded output should pass the deterministic local evaluator."""
    case = PipelineEvalCase(
        name="grounded",
        input="Python list comprehension",
        actual_output="A list comprehension uses square brackets to build a new list.",
        retrieval_context=("A list comprehension uses square brackets to build a new list.",),
        expected_terms=("list comprehension", "square brackets"),
        forbidden_terms=("binary search",),
    )

    result = evaluate_case(case, backend="heuristic")

    assert result.passed is True
    assert result.hallucination_score <= 0.05
    assert result.answer_relevancy_score >= 0.60


def test_unsupported_output_fails_local_hallucination_gate() -> None:
    """Unsupported terms in the output should count toward hallucination rate."""
    case = PipelineEvalCase(
        name="hallucinated",
        input="Python bubble sort",
        actual_output="Bubble sort works by doing binary search on the sorted half.",
        retrieval_context=("Bubble sort compares adjacent elements and swaps them when out of order.",),
        expected_terms=("bubble sort",),
        forbidden_terms=("binary search",),
    )

    report = evaluate_cases([case], backend="heuristic", max_hallucination_rate=0.05)

    assert report.passed_gate is False
    assert report.failed_cases == 1
    assert report.hallucination_rate == 1.0


def test_pipeline_cases_fixture_passes_local_gate() -> None:
    """The fixed case set should stay below the 5% gate in local mode."""
    cases = load_cases("evals/pipeline_cases.jsonl")

    report = evaluate_cases(cases, backend="heuristic", max_hallucination_rate=0.05)

    assert report.passed_gate is True
    assert report.failed_cases == 0
    assert report.hallucination_rate == 0.0


def test_evaluate_pipeline_artifacts_records_job_scores() -> None:
    """Generated job artifacts should produce a serializable score artifact."""
    artifact = evaluate_pipeline_artifacts(
        topic="Python for loop",
        code="for item in items:\n    print(item)",
        script="A Python for loop iterates over each item.",
        rag_context={
            "documents": [
                {
                    "citation": "python/docs#chunk-1",
                    "content": "A Python for loop iterates over each item in a sequence.",
                    "score": 0.9,
                }
            ]
        },
    )

    assert artifact["skipped"] is False
    assert "faithfulness_score" in artifact
    assert "answer_relevancy_score" in artifact
    assert "hallucination_score" in artifact


def test_deepeval_backend_uses_deepeval_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DeepEval backend should instantiate and measure DeepEval metrics."""
    measured: list[str] = []

    class _FakeLLMTestCase:
        """Tiny stand-in for DeepEval's LLMTestCase."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _FakeFaithfulnessMetric:
        """Tiny stand-in for DeepEval's FaithfulnessMetric."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.score = 0.99
            self.reason = "faithful to retrieval context"

        def measure(self, test_case: _FakeLLMTestCase) -> None:
            measured.append(f"faithfulness:{test_case.kwargs['input']}")

    class _FakeAnswerRelevancyMetric:
        """Tiny stand-in for DeepEval's AnswerRelevancyMetric."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.score = 0.95
            self.reason = "relevant to input"

        def measure(self, test_case: _FakeLLMTestCase) -> None:
            measured.append(f"relevancy:{test_case.kwargs['input']}")

    def _fake_import_module(name: str) -> Any:
        if name == "deepeval.metrics":
            return SimpleNamespace(
                FaithfulnessMetric=_FakeFaithfulnessMetric,
                AnswerRelevancyMetric=_FakeAnswerRelevancyMetric,
            )
        if name == "deepeval.test_case":
            return SimpleNamespace(LLMTestCase=_FakeLLMTestCase)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(evaluation, "import_module", _fake_import_module)
    case = PipelineEvalCase(
        name="deepeval_mock",
        input="Python range loop",
        actual_output="range(10) counts from zero up to but not including ten.",
        retrieval_context=("range(10) counts from zero up to but not including ten.",),
    )

    result = evaluate_case(case, backend="deepeval")

    assert result.backend == "deepeval"
    assert result.passed is True
    assert result.hallucination_score <= 0.05
    assert measured == ["faithfulness:Python range loop", "relevancy:Python range loop"]
