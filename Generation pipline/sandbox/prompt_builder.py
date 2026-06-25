"""sandbox/prompt_builder.py — Construct LLM prompts for code self-correction.

Converts a structured :class:`ExecutionResult` into a focused system +
user prompt pair that the correction LLM can act on precisely.
"""

from __future__ import annotations

from typing import (
    List,
    Tuple,
)

from .parser import (
    ErrorRecord,
    ExecutionResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (static)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python debugger embedded in an automated video-generation \
pipeline. Your sole task is to correct broken Python code so that it executes \
cleanly with zero errors.

Rules:
1. Return ONLY the corrected Python code — no markdown fences, no explanation.
2. Preserve the original intent and logic of the code.
3. Fix every error described; do not introduce new imports that are unavailable \
   in a standard CPython 3.11 environment unless they were already present.
4. If the code requires external data files that do not exist, mock them \
   inline with minimal synthetic content so the code can run end-to-end.
5. Never truncate the code; output must be complete and runnable.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_error_block(errors: List[ErrorRecord], attempt: int) -> str:
    """Render errors into a readable section for the prompt."""
    if not errors:
        return "(no structured errors captured)"

    parts: List[str] = [
        f"=== Execution Failure — Attempt {attempt} ===",
    ]
    for idx, err in enumerate(errors, 1):
        parts.append(f"\n--- Error {idx}: {err.error_type} ---")
        parts.append(f"Message : {err.error_message}")
        if err.first_failing_line:
            parts.append(f"Line    : {err.first_failing_line}")
        if err.traceback_frames:
            parts.append("Frames (innermost last):")
            for frame in err.traceback_frames[-4:]:  # cap to last 4 frames
                parts.append(f"  {frame.filename}:{frame.lineno} in {frame.function}")
                if frame.source_line:
                    parts.append(f"    >>> {frame.source_line}")
        if err.raw_traceback and len(err.raw_traceback) < 2000:
            parts.append("\nFull traceback:")
            parts.append(err.raw_traceback)

    return "\n".join(parts)


def _annotate_code(code: str, errors: List[ErrorRecord]) -> str:
    """Return the code with inline comments marking known bad lines.

    Lines beyond the source are left untouched.
    """
    bad_lines = {err.first_failing_line for err in errors if err.first_failing_line is not None}
    if not bad_lines:
        return code

    annotated: List[str] = []
    for lineno, line in enumerate(code.splitlines(), 1):
        if lineno in bad_lines:
            annotated.append(f"{line}  # ← ERROR HERE (line {lineno})")
        else:
            annotated.append(line)
    return "\n".join(annotated)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_correction_prompt(
    result: ExecutionResult,
    attempt: int,
    history: List[Tuple[str, ExecutionResult]] | None = None,
) -> Tuple[str, str]:
    """Build ``(system_prompt, user_prompt)`` for a correction LLM call.

    Parameters
    ----------
    result:
        The failed :class:`ExecutionResult` to correct.
    attempt:
        1-based correction attempt number (shown in the prompt).
    history:
        Optional list of ``(code, result)`` pairs from prior attempts so the
        LLM knows what was already tried.
    """
    error_block = _format_error_block(result.errors, attempt)
    annotated_code = _annotate_code(result.code_snapshot, result.errors)

    history_section = ""
    if history:
        items: List[str] = ["\n=== Prior Correction Attempts (do NOT repeat these) ==="]
        for i, (prev_code, prev_result) in enumerate(history[-3:], 1):  # last 3
            summary = (
                "✓ succeeded"
                if prev_result.success
                else f"✗ {prev_result.errors[0].error_type if prev_result.errors else 'unknown'}"
            )
            items.append(f"\n[Attempt {i}] {summary}")
            items.append("--- code snippet (first 20 lines) ---")
            items.append("\n".join(prev_code.splitlines()[:20]))
        history_section = "\n".join(items)

    stdout_section = ""
    if result.stdout.strip():
        preview = result.stdout[:500]
        stdout_section = f"\n=== stdout (preview) ===\n{preview}"

    user_prompt = f"""\
The following Python code failed to execute. Fix it so it runs cleanly.

{error_block}
{stdout_section}
{history_section}

=== Broken Code ===
{annotated_code}

Return ONLY the corrected Python source code, nothing else.
"""

    return SYSTEM_PROMPT, user_prompt.strip()
