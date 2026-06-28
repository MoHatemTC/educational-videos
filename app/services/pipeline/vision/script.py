"""Generate the spoken narration that explains a web page.

Reuses the code-tutorial scriptwriter's language rule so Egyptian-Arabic output
keeps brand/UI terms in Latin script (correct TTS pronunciation).
"""

from app.services.pipeline.agents.script import _language_rule
from app.services.pipeline.llm import PipelineLLM

_SYSTEM = (
    "You are a scriptwriter for short web-page explainer videos. The narration is read aloud as a voiceover while "
    "the page's screenshot scrolls on screen."
)


def generate_web_script(llm: PipelineLLM, url: str, description: str, language: str = "egyptian_arabic") -> str:
    """Return a 120-220 word spoken tour of the page from its description."""
    user = (
        f"Web page: {url}\n\n"
        f"Visible content (from screenshot analysis):\n{description}\n\n"
        "Write a spoken narration (120-220 words) giving a guided tour of this page: what the site is, the main "
        "sections, the navigation, and the most notable items (include prices when shown). "
        f"{_language_rule(language)} "
        "Keep brand names and UI labels (site name, button text) in their original Latin script so they are "
        "pronounced correctly. Return only the narration text — no headings, no stage directions."
    )
    return llm.complete(stage="script", system=_SYSTEM, user=user)
