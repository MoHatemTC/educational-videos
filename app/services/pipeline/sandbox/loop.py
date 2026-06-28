"""Self-healing loop: run -> parse -> repair (Kimi) -> rerun, until clean.

Only parsed traceback fields are injected into the repair prompt (never raw
stdout). Exits immediately on a clean run; gives up after ``max_retries`` repair
attempts, returning the last code + a structured per-iteration log.
"""

import re
from dataclasses import dataclass, field

from app.core.logging import logger
from app.services.pipeline.llm import PipelineLLM
from app.services.pipeline.sandbox.parser import parse_traceback
from app.services.pipeline.sandbox.runner import ExecutionResult, run_code

_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE)
_SYSTEM = (
    "You are a Python debugging assistant. You fix the given code so it runs without error and prints output. "
    "Return ONLY the corrected, complete Python code."
)


def _strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence if present."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def _correction_prompt(code: str, traceback_fields: dict | None, iteration: int) -> str:
    """Build a repair prompt from parsed traceback fields (no raw stdout)."""
    tb = traceback_fields or {}
    return (
        f"Attempt {iteration}: the following Python code failed when executed.\n\n"
        f"Exception type: {tb.get('exception_type')}\n"
        f"Exception message: {tb.get('exception_message')}\n"
        f"Failing line number: {tb.get('line')}\n"
        f"Offending source line: {tb.get('innermost_frame')}\n\n"
        f"Original code:\n{code}\n\n"
        "Return the corrected, complete Python code only — no fences, no commentary. "
        "Keep it minimal and standard-library only, and make sure it prints illustrative output."
    )


@dataclass
class SelfHealResult:
    """Outcome of the self-healing loop."""

    code: str
    validated: bool
    result: ExecutionResult
    log: list[dict] = field(default_factory=list)


def self_heal_code(
    code: str,
    llm: PipelineLLM,
    *,
    max_retries: int = 3,
    job_id: str | None = None,
) -> SelfHealResult:
    """Run the code, repairing it with Kimi until it executes cleanly.

    Args:
        code: Initial Python snippet.
        llm: Kimi client (repair calls are traced when it carries a job_id).
        max_retries: Maximum repair attempts after the initial run.
        job_id: For log correlation.

    Returns:
        A ``SelfHealResult`` with the final code, whether it validated, the last
        execution result, and a per-iteration log.
    """
    current = code
    result: ExecutionResult | None = None
    log: list[dict] = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            tb = parse_traceback(result.stderr) if result else None
            corrected = _strip_fences(
                llm.complete(stage="sandbox_repair", system=_SYSTEM, user=_correction_prompt(current, tb, attempt))
            )
            if corrected:
                current = corrected

        result = run_code(current)
        failure = None if result.ok else parse_traceback(result.stderr)
        log.append(
            {
                "iteration": attempt,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "exception_type": (failure or {}).get("exception_type"),
                "correction_attempted": attempt > 0,
            }
        )

        if result.ok:
            logger.info("sandbox_self_heal_ok", job_id=job_id, iteration=attempt)
            return SelfHealResult(code=current, validated=True, result=result, log=log)

    logger.warning("sandbox_self_heal_exhausted", job_id=job_id, attempts=max_retries + 1)
    return SelfHealResult(code=current, validated=False, result=result, log=log)  # type: ignore[arg-type]
