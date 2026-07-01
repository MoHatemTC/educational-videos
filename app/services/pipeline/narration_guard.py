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
    "let me read",
    "let me write",
    "let me count",
    "clean version",
    "revised draft",
    "revised:",
    "draft:",
    "read it aloud",
    "read aloud mentally",
    "does the narration flow",
    "narration flow",
    "wait,",
    "actually,",
    "word count",
    "count:",
    "within range",
    "words. good",
    "words. perfect",
    "good, within range",
    "that's about",
    "this seems good",
    "all good",
    "covered.",
    "return only",
    "final answer",
    "analysis:",
    "reasoning:",
    "prompt:",
    "system prompt",
    "developer instruction",
    "state that",
    "technical term",
    "arabic transliteration",
    "latin script",
    "-> yes",
)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
# No whitespace allowed before "[" so code refs (arr[j]) are rewritten but
# prose brackets (item [1], citations [1]) are left untouched.
_INDEXED_REFERENCE_RE = re.compile(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[\s*(?P<index>[^\[\]\n]+?)\s*\]")
_LEN_CALL_RE = re.compile(r"\blen\s*\(\s*(?P<arg>[^()\n]+?)\s*\)")
_COPY_CALL_RE = re.compile(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*copy\s*\(\s*\)")
_FENCE_RE = re.compile(r"```(?:text|markdown|arabic|python)?\s*\n(?P<body>.*?)\n```", re.IGNORECASE | re.DOTALL)
_QUOTED_RE = re.compile(r'["\u201c](?P<body>[^"\u201d]{40,})["\u201d]', re.DOTALL)
_NUMBERED_WORD_LINE_RE = re.compile(r"^\s*\d+\.\s+\S+(?:\s+\S+){0,3}\s*$")
_DRAFT_HEADING_RE = re.compile(
    r"^\s*(?:(?:clean|final|revised)\s+)?(?:draft|version|narration|script)\s*:\s*$|^\s*revised\s*:\s*$",
    re.IGNORECASE,
)
_WORD_COUNT_NOTE_RE = re.compile(
    r"^\s*(?:about\s+)?\d+\s+words?\b.*$|^.*\b\d+\s+words?\b.*(?:good|perfect|range).*$",
    re.IGNORECASE,
)
_CHECKLIST_NOTE_RE = re.compile(r"^\s*-\s+.*(?:->|yes,|no,).*$", re.IGNORECASE)
# Only parenthesized counters "word (1) word (2)" — not bare decimals like
# "3.14", which are legitimate in English narration.
_COUNTED_TOKEN_RE = re.compile(r"\(\d+\)")

_REPAIR_SYSTEM = (
    "You clean text for a text-to-speech narration pipeline. "
    "Return only the final spoken narration. Remove prompt restatement, analysis, reasoning, "
    "drafting notes, self-corrections, markdown, checklists, word counts, and comments about what "
    "the assistant is doing."
)


def contains_ai_monologue(text: str) -> bool:
    """Return whether text looks like model analysis instead of narration."""
    lowered = text.lower()
    if any(marker in lowered for marker in _META_MARKERS):
        return True

    numbered_word_lines = sum(1 for line in text.splitlines() if _NUMBERED_WORD_LINE_RE.match(line))
    if numbered_word_lines >= 5:
        return True

    return any(_looks_like_counting_line(line) for line in text.splitlines())


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


def _looks_like_counting_line(line: str) -> bool:
    """Return whether a line is a token-by-token word-count dump."""
    return len(_COUNTED_TOKEN_RE.findall(line)) >= 6


def _dedupe_repeated_lines(text: str) -> str:
    """Remove repeated narration lines emitted by draft/revision scaffolding."""
    seen: set[str] = set()
    kept: list[str] = []

    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(line)

    return "\n".join(kept).strip()


def _spoken_identifier(identifier: str) -> str:
    """Return a readable name for a simple Python identifier."""
    aliases = {
        "arr": "array",
        "idx": "index",
        "lst": "list",
        "num": "number",
        "nums": "numbers",
    }
    normalized = identifier.strip()
    return aliases.get(normalized, normalized.replace("_", " "))


def _spoken_index(index: str) -> str:
    """Return a readable index expression for narration."""
    spoken = index.strip()
    replacements = (
        ("+", " plus "),
        ("-", " minus "),
        ("*", " times "),
        ("/", " divided by "),
    )
    for symbol, words in replacements:
        spoken = spoken.replace(symbol, words)
    return re.sub(r"\s+", " ", spoken).strip()


def _replace_indexed_reference(match: re.Match[str]) -> str:
    """Rewrite a Python index reference as spoken narration."""
    name = match.group("name")
    index = _spoken_index(match.group("index"))
    if name == "arr":
        return f"element {index}"
    return f"{_spoken_identifier(name)} element {index}"


def _spoken_code_expression(expression: str) -> str:
    """Return a safe spoken form for a small Python expression."""
    spoken = _INDEXED_REFERENCE_RE.sub(_replace_indexed_reference, expression.strip())
    return _spoken_identifier(spoken)


def _replace_len_call(match: re.Match[str]) -> str:
    """Rewrite a len(...) call as spoken narration."""
    arg = _spoken_code_expression(match.group("arg"))
    return f"length of {arg}"


def _replace_copy_call(match: re.Match[str]) -> str:
    """Rewrite a .copy() call as spoken narration."""
    return f"copy of {_spoken_identifier(match.group('name'))}"


def _normalize_code_references_for_speech(text: str) -> str:
    """Rewrite common raw Python references into TTS-friendly speech."""
    spoken = _COPY_CALL_RE.sub(_replace_copy_call, text)
    spoken = _LEN_CALL_RE.sub(_replace_len_call, spoken)
    spoken = _INDEXED_REFERENCE_RE.sub(_replace_indexed_reference, spoken)
    return spoken


def _normalize_narration(text: str) -> str:
    """Normalize narration text without changing its meaning."""
    cleaned = _strip_wrapping_quotes(_strip_fences(text))
    cleaned = cleaned.replace("`", "")
    cleaned = _normalize_code_references_for_speech(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _drop_non_narration_lines(text: str) -> str:
    """Remove obvious non-narration lines while preserving Arabic/code narration."""
    kept: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lowered = stripped.lower()
        has_arabic = _ARABIC_RE.search(stripped) is not None

        if _DRAFT_HEADING_RE.match(stripped):
            continue

        if _NUMBERED_WORD_LINE_RE.match(stripped):
            continue

        if _WORD_COUNT_NOTE_RE.match(stripped):
            continue

        if _CHECKLIST_NOTE_RE.match(stripped):
            continue

        if _looks_like_counting_line(stripped):
            continue

        if contains_ai_monologue(stripped) and not has_arabic:
            continue

        if lowered.startswith(("-", "*")) and not has_arabic:
            continue

        kept.append(stripped)

    cleaned = "\n".join(kept).strip()
    deduped = _dedupe_repeated_lines(cleaned)
    return deduped.lstrip("\"'“").rstrip("\"'”").strip()


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

    line_cleaned = _drop_non_narration_lines(cleaned)
    if line_cleaned and not contains_ai_monologue(line_cleaned):
        return line_cleaned

    candidate = _best_candidate(line_cleaned or cleaned, language)
    return candidate or line_cleaned or cleaned


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
