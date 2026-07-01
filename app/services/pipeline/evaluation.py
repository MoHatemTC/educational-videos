"""Evaluate pipeline outputs with DeepEval-backed hallucination metrics.

Evaluation is a normal pipeline step: generated code and narration are scored
against the RAG context and stored on the job artifact. The evaluator backend is
configurable, but CI uses the DeepEval backend to satisfy the PRD gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any, Iterable, Literal
from importlib import import_module
from typing import cast

from app.core.config import settings
from app.core.logging import logger

# deepeval import
metrics_module = cast(Any, import_module("deepeval.metrics"))
test_case_module = cast(Any, import_module("deepeval.test_case"))
faithfulness_cls = metrics_module.FaithfulnessMetric
answer_relevancy_cls = metrics_module.AnswerRelevancyMetric
test_case_cls = test_case_module.LLMTestCase

EvalBackend = Literal["deepeval", "heuristic"]

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_DEFAULT_MAX_HALLUCINATION_RATE = 0.05
_DEFAULT_FAITHFULNESS_THRESHOLD = 0.95
_DEFAULT_RELEVANCY_THRESHOLD = 0.60


@dataclass(frozen=True)
class PipelineEvalCase:
    """One fixed or job-derived pipeline evaluation case."""

    name: str
    input: str
    actual_output: str
    retrieval_context: tuple[str, ...]
    expected_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PipelineEvalCase":
        """Build an evaluation case from JSON-compatible data."""
        topic = str(data.get("topic") or data.get("input") or "").strip()
        code = str(data.get("code") or "").strip()
        narration = str(data.get("narration") or data.get("script") or "").strip()
        actual_output = str(
            data.get("actual_output") or "\n\n".join(part for part in (code, narration) if part)
        ).strip()
        retrieval_context = _tuple_of_strings(data.get("retrieval_context") or data.get("context") or [])

        return cls(
            name=str(data.get("name") or topic or "pipeline_eval_case"),
            input=topic,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            expected_terms=_tuple_of_strings(data.get("expected_terms") or []),
            forbidden_terms=_tuple_of_strings(data.get("forbidden_terms") or []),
        )


@dataclass(frozen=True)
class PipelineEvalResult:
    """Metric scores for a single pipeline evaluation case."""

    name: str
    backend: EvalBackend
    faithfulness_score: float
    answer_relevancy_score: float
    hallucination_score: float
    passed: bool
    skipped: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-serializable result artifact."""
        return {
            "name": self.name,
            "backend": self.backend,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "answer_relevancy_score": round(self.answer_relevancy_score, 4),
            "hallucination_score": round(self.hallucination_score, 4),
            "passed": self.passed,
            "skipped": self.skipped,
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class PipelineEvalReport:
    """Aggregate hallucination gate report for a fixed case set."""

    backend: EvalBackend
    max_hallucination_rate: float
    total_cases: int
    failed_cases: int
    hallucination_rate: float
    passed_gate: bool
    results: tuple[PipelineEvalResult, ...]

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-serializable report artifact."""
        return {
            "backend": self.backend,
            "max_hallucination_rate": self.max_hallucination_rate,
            "total_cases": self.total_cases,
            "failed_cases": self.failed_cases,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "passed_gate": self.passed_gate,
            "results": [result.to_artifact() for result in self.results],
        }


def evaluate_pipeline_artifacts(
    *,
    topic: str,
    code: str,
    script: str,
    rag_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate one generated job and return scores suitable for artifacts."""
    context = _context_from_rag_artifact(rag_context)
    backend = _pipeline_eval_backend()
    if not context:
        return {"skipped": True, "backend": backend, "reason": "missing_rag_context"}

    case = PipelineEvalCase(
        name="job_pipeline_output",
        input=topic,
        actual_output="\n\n".join(part for part in (code.strip(), script.strip()) if part),
        retrieval_context=context,
    )

    try:
        result = evaluate_case(case, backend=backend)
    except Exception as exc:  # noqa: BLE001 - generation should record eval failure, not crash silently
        logger.exception("pipeline_eval_failed", backend=backend, error=str(exc))
        return {
            "name": case.name,
            "backend": backend,
            "skipped": False,
            "passed": False,
            "faithfulness_score": 0.0,
            "answer_relevancy_score": 0.0,
            "hallucination_score": 1.0,
            "reasons": [str(exc)],
        }

    return {"skipped": False, **result.to_artifact()}


def evaluate_case(
    case: PipelineEvalCase,
    *,
    backend: EvalBackend | None = None,
    faithfulness_threshold: float | None = None,
    relevancy_threshold: float | None = None,
) -> PipelineEvalResult:
    """Evaluate a single case with the selected backend."""
    selected_backend = backend or _pipeline_eval_backend()
    threshold = faithfulness_threshold if faithfulness_threshold is not None else _faithfulness_threshold()
    answer_threshold = relevancy_threshold if relevancy_threshold is not None else _relevancy_threshold()

    if not case.retrieval_context:
        return PipelineEvalResult(
            name=case.name,
            backend=selected_backend,
            faithfulness_score=0.0,
            answer_relevancy_score=0.0,
            hallucination_score=1.0,
            passed=False,
            skipped=True,
            reasons=["missing retrieval_context"],
        )

    if selected_backend == "deepeval":
        return _evaluate_with_deepeval(case, threshold, answer_threshold)

    return _evaluate_with_heuristics(case, threshold, answer_threshold)


def evaluate_cases(
    cases: Iterable[PipelineEvalCase],
    *,
    backend: EvalBackend | None = None,
    max_hallucination_rate: float | None = None,
) -> PipelineEvalReport:
    """Evaluate a fixed case set and compute the merge-blocking rate."""
    selected_backend = backend or _pipeline_eval_backend()
    max_rate = max_hallucination_rate if max_hallucination_rate is not None else _max_hallucination_rate()
    results = tuple(evaluate_case(case, backend=selected_backend) for case in cases)
    total_cases = len(results)
    failed_cases = sum(1 for result in results if not result.passed)
    hallucination_rate = failed_cases / total_cases if total_cases else 1.0

    return PipelineEvalReport(
        backend=selected_backend,
        max_hallucination_rate=max_rate,
        total_cases=total_cases,
        failed_cases=failed_cases,
        hallucination_rate=hallucination_rate,
        passed_gate=hallucination_rate <= max_rate,
        results=results,
    )


def _evaluate_with_deepeval(
    case: PipelineEvalCase,
    faithfulness_threshold: float,
    relevancy_threshold: float,
) -> PipelineEvalResult:
    """Evaluate a case with DeepEval faithfulness and answer relevancy metrics."""
    metrics_module = cast(Any, import_module("deepeval.metrics"))
    test_case_module = cast(Any, import_module("deepeval.test_case"))

    faithfulness_cls = metrics_module.FaithfulnessMetric
    answer_relevancy_cls = metrics_module.AnswerRelevancyMetric
    test_case_cls = test_case_module.LLMTestCase
    test_case = test_case_cls(
        input=case.input,
        actual_output=case.actual_output,
        retrieval_context=list(case.retrieval_context),
    )
    model = _deepeval_model()
    common_kwargs: dict[str, Any] = {"include_reason": True}
    faithfulness_kwargs = {**common_kwargs, "threshold": faithfulness_threshold}
    answer_kwargs = {**common_kwargs, "threshold": relevancy_threshold}
    if model:
        faithfulness_kwargs["model"] = model
        answer_kwargs["model"] = model

    faithfulness_metric = faithfulness_cls(**faithfulness_kwargs)
    answer_relevancy_metric = answer_relevancy_cls(**answer_kwargs)
    faithfulness_metric.measure(test_case)
    answer_relevancy_metric.measure(test_case)

    faithfulness_score = _score_from_metric(faithfulness_metric)
    answer_relevancy_score = _score_from_metric(answer_relevancy_metric)
    hallucination_score = max(0.0, min(1.0, 1.0 - faithfulness_score))
    passed = faithfulness_score >= faithfulness_threshold and answer_relevancy_score >= relevancy_threshold
    reasons = _metric_reasons(faithfulness_metric, answer_relevancy_metric)

    return PipelineEvalResult(
        name=case.name,
        backend="deepeval",
        faithfulness_score=faithfulness_score,
        answer_relevancy_score=answer_relevancy_score,
        hallucination_score=hallucination_score,
        passed=passed,
        reasons=reasons,
    )


def _evaluate_with_heuristics(
    case: PipelineEvalCase,
    faithfulness_threshold: float,
    relevancy_threshold: float,
) -> PipelineEvalResult:
    """Evaluate groundedness with deterministic offline checks for local tests."""
    output = _normalize(case.actual_output)
    context = _normalize("\n".join(case.retrieval_context))
    reasons: list[str] = []

    unsupported_terms = [
        term for term in case.forbidden_terms if _contains_phrase(output, term) and not _contains_phrase(context, term)
    ]
    missing_terms = [term for term in case.expected_terms if not _contains_phrase(output, term)]

    faithfulness_penalty = min(1.0, (0.5 * len(unsupported_terms)) + (0.1 * len(missing_terms)))
    faithfulness_score = max(0.0, 1.0 - faithfulness_penalty)
    if unsupported_terms:
        reasons.append(f"unsupported terms: {', '.join(unsupported_terms)}")
    if missing_terms:
        reasons.append(f"missing expected terms: {', '.join(missing_terms)}")

    answer_relevancy_score = _token_overlap_score(_normalize(case.input), output)
    if case.expected_terms:
        matched_expected = sum(1 for term in case.expected_terms if _contains_phrase(output, term))
        expected_score = matched_expected / len(case.expected_terms)
        answer_relevancy_score = max(answer_relevancy_score, expected_score)

    hallucination_score = max(0.0, min(1.0, 1.0 - faithfulness_score))
    passed = hallucination_score <= (1.0 - faithfulness_threshold) and answer_relevancy_score >= relevancy_threshold
    if not reasons:
        reasons.append("output stayed grounded in the provided retrieval context")
    if answer_relevancy_score < relevancy_threshold:
        reasons.append("answer relevancy below threshold")

    return PipelineEvalResult(
        name=case.name,
        backend="heuristic",
        faithfulness_score=faithfulness_score,
        answer_relevancy_score=answer_relevancy_score,
        hallucination_score=hallucination_score,
        passed=passed,
        reasons=reasons,
    )


def _score_from_metric(metric: Any) -> float:
    """Read a bounded numeric score from a DeepEval metric instance."""
    value = getattr(metric, "score", 0.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _metric_reasons(*metrics: Any) -> list[str]:
    """Collect human-readable reasons from DeepEval metric instances."""
    reasons: list[str] = []
    for metric in metrics:
        reason = str(getattr(metric, "reason", "") or "").strip()
        if reason:
            reasons.append(reason)
    return reasons or ["DeepEval metrics completed without a reason string"]


def _context_from_rag_artifact(rag_context: dict[str, Any] | None) -> tuple[str, ...]:
    """Extract retrieved context strings from the RAG artifact shape."""
    if not rag_context:
        return ()

    documents = rag_context.get("documents") or rag_context.get("chunks") or []
    context: list[str] = []
    for item in documents:
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("text") or "").strip()
            citation = str(item.get("citation") or item.get("source") or "").strip()
            if content and citation:
                context.append(f"[{citation}] {content}")
            elif content:
                context.append(content)
        elif item:
            context.append(str(item))
    return tuple(context)


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    """Normalize JSON values into a tuple of non-empty strings."""
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Iterable):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _pipeline_eval_backend() -> EvalBackend:
    """Return the configured evaluator backend."""
    raw = str(getattr(settings, "PIPELINE_EVAL_BACKEND", os.getenv("PIPELINE_EVAL_BACKEND", "deepeval"))).lower()
    if raw == "heuristic":
        return "heuristic"
    return "deepeval"


def _max_hallucination_rate() -> float:
    """Return the maximum allowed hallucination rate."""
    return float(getattr(settings, "PIPELINE_EVAL_MAX_HALLUCINATION_RATE", _DEFAULT_MAX_HALLUCINATION_RATE))


def _faithfulness_threshold() -> float:
    """Return the minimum acceptable faithfulness score."""
    return float(getattr(settings, "PIPELINE_EVAL_FAITHFULNESS_THRESHOLD", _DEFAULT_FAITHFULNESS_THRESHOLD))


def _relevancy_threshold() -> float:
    """Return the minimum acceptable answer relevancy score."""
    return float(getattr(settings, "PIPELINE_EVAL_RELEVANCY_THRESHOLD", _DEFAULT_RELEVANCY_THRESHOLD))


def _deepeval_model() -> str:
    """Return an optional DeepEval judge model name."""
    return str(getattr(settings, "PIPELINE_EVAL_MODEL", os.getenv("PIPELINE_EVAL_MODEL", ""))).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return whether a normalized phrase appears in normalized text."""
    return _normalize(phrase) in text


def _normalize(text: str) -> str:
    """Lowercase and compact text for deterministic local checks."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _token_overlap_score(input_text: str, output_text: str) -> float:
    """Return simple token overlap between input and output for local checks."""
    input_tokens = set(_WORD_RE.findall(input_text))
    output_tokens = set(_WORD_RE.findall(output_text))
    if not input_tokens:
        return 0.0
    return len(input_tokens & output_tokens) / len(input_tokens)
