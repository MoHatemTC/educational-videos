"""Research agent — distils a topic into cited key teaching points."""

from app.services.pipeline.llm import PipelineLLM

_SYSTEM = (
    "You are a precise technical curriculum researcher for short coding tutorial videos. "
    "You only state facts you are confident are correct."
)


def research_topic(
    llm: PipelineLLM,
    topic: str,
    language: str = "en",
    grounding_context: str | None = None,
) -> str:
    """Return 5-7 concrete teaching points for the topic as markdown bullets."""
    context_block = grounding_context or "No retrieved documentation context is available."
    user = (
        f"Topic: {topic}\n"
        f"Narration language: {language}\n\n"
        "Retrieved documentation context with citations:\n"
        f"{context_block}\n\n"
        "List 5-7 key teaching points a 60-90 second educational coding video should cover. "
        "Each point must be concrete, accurate, and build toward a single runnable example. "
        "Ground the points in the retrieved context when context is available. "
        "Add the relevant citation at the end of any point that uses retrieved context. "
        "If no context is available, do not invent citations. "
        "Return only markdown bullet points (one per line), no preamble or conclusion."
    )
    return llm.complete(stage="research", system=_SYSTEM, user=user)
