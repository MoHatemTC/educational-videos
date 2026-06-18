"""Tests for the Sprint 2 RAG foundation retrieval stack."""

from rag_tool.tool_definition import retrieve_technical_docs
from agent.research_agent import (
    build_research_graph,
    call_retrieval_tool_node,
    prepare_context_node,
    prepare_tool_input_node,
)
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from ingestion.chunker import build_chunk_id, chunk_documents
from ingestion.loader import load_documents
from ingestion.vector_store import count_vectors, get_collection, upsert_documents
from rag_tool.citation import build_citation_from_metadata
from rag_tool.metadata import DocumentMetadata, build_metadata_filter
from rag_tool.retriever import retrieve_chunks
from rag_tool.schema import RetrievalQuery


class FakeEmbeddings(Embeddings):
    """Small deterministic embedding function for fast local tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with simple keyword features.

        Args:
            texts: Text values to embed.

        Returns:
            Deterministic embedding vectors.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with simple keyword features.

        Args:
            text: Text value to embed.

        Returns:
            Deterministic embedding vector.
        """
        lowered_text = text.lower()

        return [
            float("vector" in lowered_text),
            float("search" in lowered_text),
            float("qdrant" in lowered_text),
            float("metadata" in lowered_text),
        ]


def make_collection_name() -> str:
    """Create a valid unique Chroma collection name.

    Returns:
        Unique Chroma collection name.
    """
    return f"test_rag_{uuid4().hex}"


def test_document_metadata_requires_shared_contract() -> None:
    """Validate the required source/version/doc_type metadata contract."""
    metadata = DocumentMetadata(
        source="qdrant",
        version="master",
        doc_type="md",
    )

    assert metadata.source == "qdrant"
    assert metadata.version == "master"
    assert metadata.doc_type == "md"


def test_document_metadata_rejects_empty_values() -> None:
    """Reject empty metadata values."""
    with pytest.raises(ValidationError):
        DocumentMetadata(source="", version="master", doc_type="md")


def test_metadata_filter_builder_uses_shared_fields() -> None:
    """Build Chroma-compatible filters from the shared metadata fields."""
    metadata_filter = build_metadata_filter(
        source="qdrant",
        version="master",
        doc_type="md",
    )

    assert metadata_filter == {
        "$and": [
            {"source": "qdrant"},
            {"version": "master"},
            {"doc_type": "md"},
        ]
    }


def test_loader_loads_supported_files_with_metadata(tmp_path: Path) -> None:
    """Load supported documentation files and attach metadata."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "intro.md").write_text("Qdrant vector search docs.", encoding="utf-8")
    (docs_dir / "image.png").write_bytes(b"not a real image")

    documents = load_documents(docs_dir)

    assert len(documents) == 1
    assert documents[0].page_content == "Qdrant vector search docs."
    assert documents[0].metadata == {
        "source": "qdrant",
        "version": "master",
        "doc_type": "md",
        "path": "intro.md",
    }


def test_chunk_id_is_deterministic() -> None:
    """Generate the same chunk ID for the same source/version/path/index."""
    first_id = build_chunk_id("qdrant", "master", "intro.md", 0)
    second_id = build_chunk_id("qdrant", "master", "intro.md", 0)

    assert first_id == second_id


def test_chunker_preserves_metadata_and_adds_chunk_fields() -> None:
    """Chunk documents while preserving shared metadata."""
    document = Document(
        page_content="Qdrant supports vector search with metadata filters.",
        metadata={
            "source": "qdrant",
            "version": "master",
            "doc_type": "md",
            "path": "intro.md",
        },
    )

    chunks = chunk_documents([document], chunk_size=40, chunk_overlap=5)

    assert chunks
    assert chunks[0].metadata["source"] == "qdrant"
    assert chunks[0].metadata["version"] == "master"
    assert chunks[0].metadata["doc_type"] == "md"
    assert chunks[0].metadata["path"] == "intro.md"
    assert "chunk_id" in chunks[0].metadata
    assert "content_hash" in chunks[0].metadata


def test_vector_store_upsert_is_idempotent(tmp_path: Path) -> None:
    """Re-ingesting the same chunks must not duplicate vectors."""
    document = Document(
        page_content="Qdrant vector search metadata.",
        metadata={
            "source": "qdrant",
            "version": "master",
            "doc_type": "md",
            "path": "intro.md",
        },
    )
    chunks = chunk_documents([document], chunk_size=80, chunk_overlap=10)
    collection_name = make_collection_name()

    first_stats = upsert_documents(
        documents=chunks,
        embedding_function=FakeEmbeddings(),
        persist_dir=tmp_path,
        collection_name=collection_name,
    )
    second_stats = upsert_documents(
        documents=chunks,
        embedding_function=FakeEmbeddings(),
        persist_dir=tmp_path,
        collection_name=collection_name,
    )
    collection = get_collection(
        persist_dir=tmp_path,
        collection_name=collection_name,
    )

    assert first_stats.count_after == len(chunks)
    assert second_stats.count_after == first_stats.count_after
    assert second_stats.net_new_vectors == 0
    assert count_vectors(collection) == len(chunks)


def test_citation_formatter_includes_required_fields() -> None:
    """Build citations using source, version, doc_type, path, and chunk index."""
    citation = build_citation_from_metadata(
        {
            "source": "qdrant",
            "version": "master",
            "doc_type": "md",
            "path": "intro.md",
            "chunk_index": 2,
            "chunk_id": "abc",
            "content_hash": "def",
        }
    )

    assert citation == "[qdrant/master/md] intro.md#chunk-2"


def test_retrieval_query_rejects_invalid_input() -> None:
    """Reject invalid retrieval tool input."""
    with pytest.raises(ValidationError):
        RetrievalQuery(query="", top_k=0)


def test_retriever_returns_cited_filtered_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieve cited chunks using metadata filters."""
    document = Document(
        page_content="Qdrant vector search supports metadata filtering.",
        metadata={
            "source": "qdrant",
            "version": "master",
            "doc_type": "md",
            "path": "intro.md",
        },
    )
    chunks = chunk_documents([document], chunk_size=80, chunk_overlap=10)
    collection_name = make_collection_name()

    upsert_documents(
        documents=chunks,
        embedding_function=FakeEmbeddings(),
        persist_dir=tmp_path,
        collection_name=collection_name,
    )

    monkeypatch.setattr(
        "rag_tool.retriever.get_embedding_function",
        lambda: FakeEmbeddings(),
    )

    output = retrieve_chunks(
        RetrievalQuery(
            query="vector search",
            source="qdrant",
            version="master",
            doc_type="md",
            top_k=1,
            similarity_threshold=0.0,
        ),
        persist_dir=tmp_path,
        collection_name=collection_name,
    )

    assert output.result_count == 1
    assert output.results[0].citation.startswith("[qdrant/master/md]")
    assert output.results[0].metadata.source == "qdrant"


def test_structured_tool_is_registered() -> None:
    """Confirm the retrieval tool is registered as a StructuredTool."""
    assert retrieve_technical_docs.name == "retrieve_technical_docs"
    assert retrieve_technical_docs.args_schema is RetrievalQuery
    assert "source" in retrieve_technical_docs.description
    assert "version" in retrieve_technical_docs.description
    assert "doc_type" in retrieve_technical_docs.description


def test_agent_graph_compiles() -> None:
    """Confirm the LangGraph StateGraph research agent compiles."""
    graph = build_research_graph()

    assert graph is not None


def test_prepare_tool_input_node_uses_metadata_contract() -> None:
    """Confirm the agent prepares tool input with shared metadata filters."""
    state = prepare_tool_input_node(
        {
            "query": "vector search",
            "source": "qdrant",
            "version": "master",
            "doc_type": "md",
            "top_k": 3,
            "similarity_threshold": 0.1,
        }
    )

    assert state["tool_input"] == {
        "query": "vector search",
        "source": "qdrant",
        "version": "master",
        "doc_type": "md",
        "top_k": 3,
        "similarity_threshold": 0.1,
    }


def test_prepare_context_node_formats_citations() -> None:
    """Confirm the agent formats retrieved chunks into cited context."""
    state = prepare_context_node(
        {
            "query": "vector search",
            "retrieval": {
                "results": [
                    {
                        "citation": "[qdrant/master/md] intro.md#chunk-0",
                        "content": "Qdrant supports vector search.",
                    }
                ]
            },
        }
    )

    assert "Citation: [qdrant/master/md] intro.md#chunk-0" in state["answer_context"]
