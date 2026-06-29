"""Tests for narration AI-monologue safeguards."""

from app.services.pipeline.narration_guard import clean_narration_text, contains_ai_monologue


def test_contains_ai_monologue_detects_drafting_notes() -> None:
    """Drafting language should be treated as a narration leak."""
    text = "The user wants a spoken narration script. Let me draft it first."
    assert contains_ai_monologue(text)


def test_clean_narration_keeps_clean_egyptian_script() -> None:
    """Clean narration should not be changed."""
    text = "هنكتب function اسمها bubble_sort، وبعد كده نستخدم print عشان نشوف output."
    assert clean_narration_text(text, "egyptian_arabic") == text


def test_clean_narration_extracts_final_arabic_candidate() -> None:
    """Reasoning notes should be removed before script storage or TTS."""
    leaked = """
The user wants a spoken narration script for a short educational coding video.

Let me draft it:

"هنكتب function اسمها bubble_sort، وبعد كده نشرح ascending و descending باستخدام نفس nested loops."

Wait, I need to make sure all technical terms stay in English.
Word count is still too short.
"""
    cleaned = clean_narration_text(leaked, "egyptian_arabic")

    assert "The user wants" not in cleaned
    assert "Let me draft" not in cleaned
    assert "Wait" not in cleaned
    assert "Word count" not in cleaned
    assert cleaned.startswith("هنكتب function")


def test_clean_narration_removes_inline_markdown_ticks() -> None:
    """Inline markdown ticks should not reach TTS."""
    text = "هنكتب `def bubble_sort(arr):` ونشوف `output`."
    assert clean_narration_text(text, "egyptian_arabic") == "هنكتب def bubble_sort(arr): ونشوف output."
