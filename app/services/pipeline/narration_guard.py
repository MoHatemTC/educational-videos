"""Guards that keep model analysis out of spoken narration.

The pipeline should only store and synthesize final narration text. These helpers
detect common reasoning/drafting leaks, deterministically extract the best final
narration candidate, and optionally ask the LLM for a cleanup pass only when a
leak is detected.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.logging import logger
from app.services.pipeline.llm import PipelineLLM


_META_MARKERS: tuple[str, ...] = (
    "the user wants",
    "the user asked",
    "key teaching points",
    "the code provided",
    "i need to",
    "i should",
    "i'll",
    "let me draft",
    "let me check",
    "let me revise",
    "wait,",
    "actually,",
    "word count",
    "count:",
    "that's about",
    "this seems good",
    "covered.",
    "return only",
    "final answer",
    "analysis",
    "reasoning",
    "prompt:",
    "system prompt",
    "developer instruction",
)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_FENCE_RE = re.compile(r"```(?:text|markdown|arabic|python)?\s*\n(?P<body>.*?)\n```", re.IGNORECASE | re.DOTALL)
_QUOTED_RE = re.compile(r'["\u201c](?P<body>[^"\u201d]{40,})["\u201d]', re.DOTALL)

_REPAIR_SYSTEM = (
    "You clean text for a text-to-speech narration pipeline. "
    "Return only the final spoken narration. Remove prompt restatement, analysis, reasoning, "
    "drafting notes, self-corrections, markdown, checklists, word counts, and comments about what "
    "the assistant is doing."
)


def contains_ai_monologue(text: str) -> bool:
    """Return whether text looks like model analysis instead of narration."""
    lowered = text.lower()
    return any(marker in lowered for marker in _META_MARKERS)


def _strip_fences(text: str) -> str:
    """Replace markdown code fences with their body text."""
    return _FENCE_RE.sub(lambda match: match.group("body").strip(), text).strip()


def _strip_wrapping_quotes(text: str) -> str:
    """Remove one layer of wrapping quotes."""
    stripped = text.strip()
    pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("«", "»"))
    for left, right in pairs:
        if stripped.startswith(left) and stripped.endswith(right):
            return stripped[1:-1].strip()
    return stripped


def _normalize_narration(text: str) -> str:
    """Normalize narration text without changing its meaning."""
    cleaned = _strip_wrapping_quotes(_strip_fences(text))
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _candidate_blocks(text: str) -> Sequence[str]:
    """Return likely final-narration candidate blocks."""
    normalized = _normalize_narration(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    quoted = [match.group("body").strip() for match in _QUOTED_RE.finditer(normalized)]
    lines = [line.strip() for line in normalized.splitlines() if len(line.strip()) >= 40]
    return [*quoted, *paragraphs, *lines]


def _candidate_score(candidate: str, language: str) -> int:
    """Score a candidate narration block."""
    if not candidate:
        return -1000

    lowered = candidate.lower()
    if contains_ai_monologue(candidate):
        return -500

    if lowered.startswith(("#", "-", "*")):
        return -50

    words = re.findall(r"\S+", candidate)
    score = len(words)

    if language == "egyptian_arabic" and _ARABIC_RE.search(candidate):
        score += 100

    if 60 <= len(words) <= 240:
        score += 25

    if any(term in lowered for term in ("function", "code", "output", "print", "return")):
        score += 10

    return score


def _best_candidate(text: str, language: str) -> str:
    """Extract the best final narration candidate from leaked output."""
    candidates = list(_candidate_blocks(text))
    if not candidates:
        return _normalize_narration(text)

    best = max(candidates, key=lambda candidate: _candidate_score(candidate, language))
    return _normalize_narration(best)


def clean_narration_text(text: str, language: str) -> str:
    """Return narration text with obvious AI monologue removed."""
    cleaned = _normalize_narration(text)

    if not contains_ai_monologue(cleaned):
        return cleaned

    candidate = _best_candidate(cleaned, language)
    return candidate or cleaned


def ensure_clean_narration(
    llm: PipelineLLM,
    *,
    raw_text: str,
    language: str,
    stage: str,
    context: str,
) -> str:
    """Clean narration, using one repair LLM call only when deterministic cleanup is not enough."""
    cleaned = clean_narration_text(raw_text, language)

    if cleaned and not contains_ai_monologue(cleaned):
        return cleaned

    logger.warning("narration_ai_monologue_detected_repairing", stage=stage, context=context)

    try:
        repaired = llm.complete(
            stage=f"{stage}_repair",
            system=_REPAIR_SYSTEM,
            user=(
                "Clean this model output for spoken narration. "
                "Return only the final narration text that should be read aloud.\n\n"
                f"{raw_text}"
            ),
            temperature=0.0,
            max_tokens=1200,
        )
    except Exception as exc:  # noqa: BLE001 - generation should not crash if cleanup repair fails
        logger.warning("narration_repair_failed", stage=stage, context=context, error=str(exc))
        return cleaned or _normalize_narration(raw_text)

    repaired_clean = clean_narration_text(repaired, language)
    if repaired_clean and not contains_ai_monologue(repaired_clean):
        return repaired_clean

    logger.warning("narration_repair_still_suspicious_using_best_candidate", stage=stage, context=context)
    return _best_candidate(f"{raw_text}\n\n{repaired}", language).strip()
