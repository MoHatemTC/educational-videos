"""Tests for wiring RAG grounding context into pipeline prompts."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing_extensions import override

from app.core.config import settings
from app.services.pipeline.agents.code import generate_code
from app.services.pipeline.agents.research import research_topic
from app.services.pipeline.rag import retrieve_grounding_context
from app.services.rag.ingestion.chunker import chunk_documents
from app.services.rag.ingestion.vector_store import upsert_documents


class FakeEmbeddings(Embeddings):
    """Deterministic embedding function for seeded RAG tests."""

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with simple keyword features."""
        return [self.embed_query(text) for text in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        """Embed text with simple keyword features."""
        lowered_text = text.lower()
        return [
            float("vector" in lowered_text),
            float("search" in lowered_text),
            float("qdrant" in lowered_text),
            float("metadata" in lowered_text),
        ]


class RecordingLLM:
    """Minimal LLM spy that records pipeline prompt payloads."""

    def __init__(self) -> None:
        """Initialize the call recorder."""
        self.calls: list[dict[str, str]] = []

    def complete(self, *, stage: str, system: str, user: str) -> str:
        """Record the prompt and return deterministic stage output."""
        self.calls.append({"stage": stage, "system": system, "user": user})
        if stage == "code":
            return "print('vector search demo')"
        return "- Use vector search with metadata filters. [qdrant/master/md] intro.md#chunk-0"


def _collection_name() -> str:
    """Return a unique valid Chroma collection name."""
    return f"test_rag_{uuid4().hex}"


def test_seeded_rag_context_reaches_research_and_code_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed a RAG store and assert retrieved context reaches both prompts."""
    pytest.importorskip("chromadb")

    collection_name = _collection_name()
    document = Document(
        page_content="Qdrant vector search supports metadata filtering for retrieval.",
        metadata={
            "source": "qdrant",
            "version": "master",
            "doc_type": "md",
            "path": "intro.md",
        },
    )
    chunks = chunk_documents([document], chunk_size=90, chunk_overlap=10)
    upsert_documents(
        documents=chunks,
        embedding_function=FakeEmbeddings(),
        persist_dir=tmp_path,
        collection_name=collection_name,
    )

    monkeypatch.setattr("app.services.rag.tool.retriever.get_embedding_function", lambda: FakeEmbeddings())
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "RAG_CHROMA_COLLECTION", collection_name)
    monkeypatch.setattr(settings, "RAG_TOP_K", 1)
    monkeypatch.setattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "RAG_SOURCE", "qdrant")
    monkeypatch.setattr(settings, "RAG_VERSION", "master")
    monkeypatch.setattr(settings, "RAG_DOC_TYPE", "md")

    grounding = retrieve_grounding_context("vector search", "en")
    prompt_context = grounding.format_for_prompt()

    llm = RecordingLLM()
    research_notes = research_topic(llm, "vector search", "en", grounding_context=prompt_context)  # type: ignore[arg-type]
    code = generate_code(llm, "vector search", research_notes, grounding_context=prompt_context)  # type: ignore[arg-type]

    combined_prompts = "\n".join(call["user"] for call in llm.calls)

    assert grounding.citations == ["[qdrant/master/md] intro.md#chunk-0"]
    assert "Qdrant vector search supports metadata filtering" in combined_prompts
    assert "[qdrant/master/md] intro.md#chunk-0" in combined_prompts
    assert code == "print('vector search demo')"


def test_rag_failure_degrades_to_empty_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable vector stores should not block generation."""

    def fail_retrieval(*args: Any, **kwargs: Any) -> object:
        """Raise a vector-store failure."""
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr("app.services.rag.tool.retriever.retrieve_chunks", fail_retrieval)

    grounding = retrieve_grounding_context("bubble sort", "en")

    assert not grounding.has_documents
    assert grounding.unavailable_reason == "retrieval_failed"
    assert "No retrieved documentation context" in grounding.format_for_prompt()
