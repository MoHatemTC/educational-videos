"""Run the fixed pipeline hallucination gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from app.core.logging import logger
from app.services.pipeline.evaluation import PipelineEvalCase, evaluate_cases

_JUDGE_KEY_ENV_VARS = ("OPENAI_API_KEY", "PIPELINE_EVAL_API_KEY", "LITELLM_API_KEY")


def _has_llm_judge_key() -> bool:
    """Return whether an LLM judge API key is configured for the DeepEval backend."""
    return any(os.getenv(name, "").strip() for name in _JUDGE_KEY_ENV_VARS)


def _resolve_backend(backend: str) -> str:
    """Return heuristic when DeepEval is requested but no LLM judge key exists.

    Lets the gate still run (e.g. in CI without secrets) instead of erroring.
    When a key is provisioned, the real DeepEval backend runs unchanged.
    """
    if backend == "deepeval" and not _has_llm_judge_key():
        logger.warning("pipeline_eval_no_judge_key_using_heuristic", requested_backend=backend)
        return "heuristic"
    return backend


def load_cases(path: str | Path) -> tuple[PipelineEvalCase, ...]:
    """Load fixed pipeline eval cases from a JSONL file."""
    cases: list[PipelineEvalCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        cases.append(PipelineEvalCase.from_mapping(data))
    return tuple(cases)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the pipeline hallucination gate.")
    parser.add_argument("--cases", default="evals/pipeline_cases.jsonl", help="JSONL fixture path")
    parser.add_argument("--max-hallucination-rate", type=float, default=0.05)
    parser.add_argument(
        "--backend",
        choices=("deepeval", "heuristic"),
        default="deepeval",
        help="Evaluator backend. CI should use deepeval; heuristic is for local offline checks.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed case set and return a process exit code."""
    args = parse_args(argv)
    backend = _resolve_backend(args.backend)
    report = evaluate_cases(
        load_cases(args.cases),
        backend=backend,
        max_hallucination_rate=args.max_hallucination_rate,
    )
    print(json.dumps(report.to_artifact(), indent=2, ensure_ascii=False))
    return 0 if report.passed_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
