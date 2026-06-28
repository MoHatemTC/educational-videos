"""Research agent — distils a topic into key teaching points.

For the MVP this is a direct Kimi call. Step 4 grounds it in retrieved docs
(RAG over Qdrant) and the optional web-search tool.
"""

from app.services.pipeline.llm import PipelineLLM

_SYSTEM = (
    "You are a precise technical curriculum researcher for short coding tutorial videos. "
    "You only state facts you are confident are correct."
)


def research_topic(llm: PipelineLLM, topic: str, language: str = "en") -> str:
    """Return 5-7 concrete teaching points for the topic as markdown bullets."""
    user = (
        f"Topic: {topic}\n\n"
        "List 5-7 key teaching points a 60-90 second educational coding video should cover. "
        "Each point must be concrete, accurate, and build toward a single runnable example. "
        "Return only markdown bullet points (one per line), no preamble or conclusion."
    )
    return llm.complete(stage="research", system=_SYSTEM, user=user)
