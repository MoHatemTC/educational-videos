"""Pydantic schemas for the RAG retrieval tool."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_tool.metadata import ChunkMetadata


class RetrievalQuery(BaseModel):
    """Input schema for the technical-document retrieval tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    source: str | None = Field(default=None, min_length=1)
    version: str | None = Field(default=None, min_length=1)
    doc_type: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> str:
        """Strip and validate the retrieval query.

        Args:
            value: Raw query value.

        Returns:
            Normalized query string.

        Raises:
            ValueError: If the query is not a non-empty string.
        """
        if not isinstance(value, str):
            msg = "query must be a string."
            raise ValueError(msg)

        normalized_value = value.strip()

        if not normalized_value:
            msg = "query must be non-empty."
            raise ValueError(msg)

        return normalized_value

    @field_validator("source", "version", "doc_type", mode="before")
    @classmethod
    def normalize_optional_filter(cls, value: Any) -> str | None:
        """Normalize optional metadata filter values.

        Args:
            value: Raw metadata filter value.

        Returns:
            Stripped string value, or None when empty.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            return value

        normalized_value = value.strip()
        return normalized_value or None

    def active_filters(self) -> dict[str, str]:
        """Return only metadata filters that were provided.

        Returns:
            Dictionary containing active source, version, and doc_type filters.
        """
        filters: dict[str, str] = {}

        if self.source is not None:
            filters["source"] = self.source
        if self.version is not None:
            filters["version"] = self.version
        if self.doc_type is not None:
            filters["doc_type"] = self.doc_type

        return filters


class RetrievedChunk(BaseModel):
    """One grounded retrieval result returned by the RAG tool."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    citation: str = Field(..., min_length=1)
    metadata: ChunkMetadata


class RetrievalOutput(BaseModel):
    """Structured response returned by the technical-document retrieval tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    filters_applied: dict[str, str] = Field(default_factory=dict)
    results: list[RetrievedChunk] = Field(default_factory=list)

    @property
    def result_count(self) -> int:
        """Return the number of retrieved chunks.

        Returns:
            Number of retrieval results.
        """
        return len(self.results)
