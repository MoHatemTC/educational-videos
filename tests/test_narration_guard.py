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


def test_clean_narration_removes_revised_draft_word_breakdown() -> None:
    """Draft headings, token dumps, and rubric notes should not reach TTS."""
    leaked = """
Revised draft:
هنبدأ بالطريقة التقليدية. هنعمل variable اسمه numbers ونخلي قيمته range of ten.
بعدين هنعمل empty list اسمها multi_line. لما ن run الكود، الـ values match هتطلع True.
1. هنبدأ
2. بالطريقة
3. التقليدية.
4. هنعمل
5. variable
6. اسمه
124 words. Good, within range.
- State that a list comprehension creates a new list in a single line. -> Yes,
"""

    cleaned = clean_narration_text(leaked, "egyptian_arabic")

    assert cleaned.startswith("هنبدأ بالطريقة التقليدية")
    assert "Revised draft" not in cleaned
    assert "1. هنبدأ" not in cleaned
    assert "124 words" not in cleaned
    assert "-> Yes" not in cleaned
    assert "values match" in cleaned


def test_clean_narration_removes_inline_markdown_ticks() -> None:
    """Inline markdown ticks should not reach TTS."""
    text = "هنكتب `def bubble_sort(arr):` ونشوف `output`."
    assert clean_narration_text(text, "egyptian_arabic") == "هنكتب def bubble_sort(arr): ونشوف output."


def test_clean_narration_rewrites_raw_python_references_for_tts() -> None:
    """Raw Python references should become readable spoken phrases."""
    text = "نستخدم len(arr)، ونقارن arr[j] مع arr[j+1]، وبعدها نجرب data.copy()."

    cleaned = clean_narration_text(text, "egyptian_arabic")

    assert "len(arr)" not in cleaned
    assert "arr[j]" not in cleaned
    assert "arr[j+1]" not in cleaned
    assert "data.copy()" not in cleaned
    assert "length of array" in cleaned
    assert "element j" in cleaned
    assert "element j plus 1" in cleaned
    assert "copy of data" in cleaned


def test_clean_narration_removes_meta_intro_revised_and_inline_count() -> None:
    """Section-level drafting notes and inline word counts should be stripped."""
    narration = (
        "هنبدأ بـ numbers equals list فيها الأرقام من 1 لـ 6. الهدف نعمل list جديدة فيها "
        "مربعات الأرقام الزوجية بس. في Python، list comprehension بتخلينا نعمل list جديدة "
        "في سطر واحد. بنستخدم square brackets، وكل البنية بتكون جواهم. أول حاجة جواهم "
        "بنحط x to the power of 2 عشان نحدد إن كل عنصر في النتيجة هيكون المربع. "
        "بعد كده بنكتب for x in numbers عشان نلف على كل عنصر في الـ list الأصلية من "
        "غير ما نكتب for block منفصل. وأخيراً بنضيف if x modulo 2 equals 0 عشان "
        "نفلتر وناخد الأرقام الزوجية بس قبل التحويل. السطر كامل بيبقى squares equals "
        "square brackets x to the power of 2 for x in numbers if x modulo 2 equals 0 "
        "square brackets. لما نعمل run للكود، print squares هيطلع 4 و 16 و 36."
    )
    leaked = f"""
So list is a built-in type name, thus a technical term. It should be in English Latin script.
Let me write a clean version:
{narration}
Revised:
{narration}
Let me count again:
هنبدأ (1) بـ (2) numbers (3) equals (4) list (5) فيها (6) الأرقام (7) من (8) 1 (9)
About 133 words. Perfect.
All good. No Arabic transliteration like
"""

    cleaned = clean_narration_text(leaked, "egyptian_arabic")

    assert cleaned == narration
    assert "technical term" not in cleaned
    assert "Let me write" not in cleaned
    assert "Revised:" not in cleaned
    assert "Let me count" not in cleaned
    assert "(1)" not in cleaned
    assert "About 133 words" not in cleaned
