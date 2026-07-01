"""Script agent — writes the spoken narration synced to the code.

Supports English and Egyptian-Arabic narration. For Arabic, technical terms stay
in English (per the project's dialect requirement) so TTS pronounces code
identifiers correctly.
"""

from app.services.pipeline.llm import PipelineLLM
from app.services.pipeline.narration_guard import ensure_clean_narration

_SYSTEM = (
    "You are a scriptwriter for short educational coding videos. "
    "The narration is read aloud as a voiceover synced to code being typed and run on screen. "
    "Return only the final spoken narration. Never include analysis, reasoning, drafts, checks, "
    "self-corrections, word counts, token-by-token breakdowns, evaluation checklists, "
    "prompt restatement, headings, stage directions, or markdown."
)


def _language_rule(language: str) -> str:
    """Return the narration-language instruction for the prompt."""
    if language == "egyptian_arabic":
        return (
            "Write the narration in Egyptian Arabic dialect (العامية المصرية). "
            "Keep ALL technical terms — keywords, function names, library names, and code — in English, "
            "written in Latin script. Do NOT transliterate technical terms into Arabic letters."
        )
    return "Write the narration in clear, friendly English."


def generate_script(
    llm: PipelineLLM,
    topic: str,
    research_notes: str,
    code: str,
    language: str = "en",
    grounding_context: str | None = None,
) -> str:
    """Return a 120-200 word spoken narration that walks through the code."""
    context_block = grounding_context or "No retrieved documentation context is available."
    user = (
        f"Topic: {topic}\n\n"
        f"Key teaching points:\n{research_notes}\n\n"
        f"The on-screen code is:\n```python\n{code}\n```\n\n"
        "Retrieved documentation context with citations:\n"
        f"{context_block}\n\n"
        "Keep explanations consistent with the retrieved context and do not state unsupported technical "
        "claims. Do NOT read citations or source numbers aloud in the narration. "
        "Write a spoken narration (120-200 words) that walks the viewer through typing and running this "
        "code, explaining what each part does and what the output means. "
        "For TTS clarity, explain raw Python references in natural English instead of reading symbols: "
        "say 'length of array' instead of 'len(arr)', 'element j' instead of 'arr[j]', "
        "'element j plus 1' instead of 'arr[j+1]', and 'copy of data' instead of 'data.copy()'. "
        f"{_language_rule(language)} "
        "Return only the narration text. Do not include headings, stage directions, markdown, analysis, "
        "reasoning, drafts, revised-draft headings, checks, self-corrections, word counts, "
        "numbered token-by-token breakdowns, evaluation checklists, or prompt restatement."
    )
    raw_script = llm.complete(stage="script", system=_SYSTEM, user=user)
    return ensure_clean_narration(
        llm,
        raw_text=raw_script,
        language=language,
        stage="script",
        context=topic,
    )
