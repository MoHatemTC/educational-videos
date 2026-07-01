"""Pipeline adapter for retrieving cited RAG grounding context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.logging import logger


@dataclass(frozen=True)
class GroundingDocument:
    """One retrieved source chunk used to ground pipeline prompts."""

    citation: str
    content: str
    score: float

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact representation."""
        return {
            "citation": self.citation,
            "content": self.content,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class GroundingContext:
    """Retrieved RAG context prepared for prompt injection and artifacts."""

    query: str
    documents: tuple[GroundingDocument, ...]
    filters: dict[str, str]
    unavailable_reason: str | None = None

    @property
    def citations(self) -> list[str]:
        """Return citations for retrieved documents in prompt order."""
        return [document.citation for document in self.documents]

    @property
    def has_documents(self) -> bool:
        """Return whether any grounding documents were retrieved."""
        return bool(self.documents)

    def format_for_prompt(self) -> str:
        """Format retrieved context as a cited prompt block."""
        if not self.documents:
            return "No retrieved documentation context is available. Do not invent citations."

        sections: list[str] = []
        for index, document in enumerate(self.documents, start=1):
            sections.extend(
                [
                    f"Source {index}: {document.citation}",
                    document.content.strip(),
                    "",
                ]
            )

        return "\n".join(sections).strip()

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact representation."""
        return {
            "query": self.query,
            "filters": self.filters,
            "citations": self.citations,
            "documents": [document.to_artifact() for document in self.documents],
            "unavailable_reason": self.unavailable_reason,
        }


def _setting_bool(name: str, default: bool) -> bool:
    """Read a bool setting without forcing config migrations for tests."""
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "on"}


def _setting_int(name: str, default: int) -> int:
    """Read an int setting with a safe fallback."""
    try:
        return int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def _setting_float(name: str, default: float) -> float:
    """Read a float setting with a safe fallback."""
    try:
        return float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def _setting_str_or_none(name: str) -> str | None:
    """Read a stripped optional string setting."""
    value = getattr(settings, name, None)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def empty_grounding_context(query: str, reason: str | None = None) -> GroundingContext:
    """Build an empty context for graceful RAG degradation."""
    return GroundingContext(query=query, documents=(), filters={}, unavailable_reason=reason)


def retrieve_grounding_context(topic: str, language: str = "en") -> GroundingContext:
    """Retrieve cited documentation context for a video topic.

    The function intentionally imports the Chroma/SentenceTransformer-backed RAG
    stack lazily so the API can still boot when the optional vector-store runtime
    is unavailable. Empty stores and retrieval failures degrade to an empty
    context instead of failing the generation job.
    """
    query = f"{topic.strip()} coding tutorial concepts and runnable Python example"

    if not _setting_bool("RAG_ENABLED", True):
        logger.info("rag_context_disabled", topic=topic, language=language)
        return empty_grounding_context(query, reason="disabled")

    try:
        from app.services.rag.tool.retriever import retrieve_chunks
        from app.services.rag.tool.schema import RetrievalQuery
    except Exception as exc:  # noqa: BLE001 - optional RAG deps must not crash the API
        logger.warning("rag_context_unavailable", topic=topic, error=str(exc))
        return empty_grounding_context(query, reason="import_failed")

    filters = {
        key: value
        for key, value in {
            "source": _setting_str_or_none("RAG_SOURCE"),
            "version": _setting_str_or_none("RAG_VERSION"),
            "doc_type": _setting_str_or_none("RAG_DOC_TYPE"),
        }.items()
        if value is not None
    }

    try:
        request = RetrievalQuery(
            query=query,
            source=filters.get("source"),
            version=filters.get("version"),
            doc_type=filters.get("doc_type"),
            top_k=_setting_int("RAG_TOP_K", 5),
            similarity_threshold=_setting_float("RAG_SIMILARITY_THRESHOLD", 0.35),
        )
        output = retrieve_chunks(
            request,
            persist_dir=_setting_str_or_none("RAG_CHROMA_PERSIST_DIR"),
            collection_name=_setting_str_or_none("RAG_CHROMA_COLLECTION"),
        )
    except Exception as exc:  # noqa: BLE001 - generation must degrade gracefully
        logger.warning("rag_context_failed", topic=topic, error=str(exc))
        return GroundingContext(query=query, documents=(), filters=filters, unavailable_reason="retrieval_failed")

    documents = tuple(
        GroundingDocument(
            citation=result.citation,
            content=result.content,
            score=result.score,
        )
        for result in output.results
        if result.content.strip()
    )

    if not documents:
        logger.info("rag_context_empty", topic=topic, filters=filters)
        return GroundingContext(query=output.query, documents=(), filters=output.filters_applied)

    logger.info(
        "rag_context_retrieved",
        topic=topic,
        result_count=len(documents),
        citations=[document.citation for document in documents],
    )
    return GroundingContext(query=output.query, documents=documents, filters=output.filters_applied)
