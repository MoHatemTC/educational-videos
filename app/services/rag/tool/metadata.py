"""Shared metadata schema and filtering helpers for the RAG foundation."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Required metadata contract shared by ingestion and retrieval.

    These fields are written by the ingestion pipeline and used directly by
    the retrieval tool for structured metadata filtering.
    """

    source: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    doc_type: str = Field(..., min_length=1)


class ChunkMetadata(DocumentMetadata):
    """Metadata stored with each embedded document chunk.

    The inherited fields form the required shared contract. The extra fields
    support deterministic re-ingestion, debugging, and citation formatting.
    """

    path: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    chunk_id: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=1)


def build_metadata_filter(
    source: str | None = None,
    version: str | None = None,
    doc_type: str | None = None,
) -> dict[str, Any]:
    """Build a Chroma-compatible metadata filter.

    Args:
        source: Optional documentation source filter.
        version: Optional documentation version filter.
        doc_type: Optional document type filter.

    Returns:
        A Chroma-compatible metadata filter dictionary.
    """
    filters: list[dict[str, str]] = []

    if source is not None:
        filters.append({"source": source})
    if version is not None:
        filters.append({"version": version})
    if doc_type is not None:
        filters.append({"doc_type": doc_type})

    if not filters:
        return {}

    if len(filters) == 1:
        return filters[0]

    return {"$and": filters}
