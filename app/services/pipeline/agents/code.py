"""Code agent — generates a single self-contained Python teaching example."""

import re

from app.services.pipeline.llm import PipelineLLM

_SYSTEM = "You write minimal, correct, self-contained Python teaching examples using only the standard library."

_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence if present."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def generate_code(llm: PipelineLLM, topic: str, research_notes: str) -> str:
    """Return one runnable Python snippet (<=25 lines) demonstrating the topic."""
    user = (
        f"Topic: {topic}\n\n"
        f"Key teaching points:\n{research_notes}\n\n"
        "Write ONE self-contained Python snippet (at most 25 lines) that clearly demonstrates the topic "
        "and prints illustrative output so a learner can see the result. Use only the standard library. "
        "Return ONLY the Python code — no markdown fences, no comments-as-explanation, no prose."
    )
    code = llm.complete(stage="code", system=_SYSTEM, user=user)
    return _strip_fences(code)
