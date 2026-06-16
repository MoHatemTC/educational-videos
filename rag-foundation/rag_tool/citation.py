"""Citation formatting helpers for retrieved RAG chunks."""

from typing import Any

from rag_tool.metadata import ChunkMetadata


def build_chunk_citation(metadata: ChunkMetadata) -> str:
    """Build a stable citation string for one retrieved chunk.

    Args:
        metadata: Validated chunk metadata.

    Returns:
        Citation string containing source, version, doc type, path, and chunk index.
    """
    return f"[{metadata.source}/{metadata.version}/{metadata.doc_type}] {metadata.path}#chunk-{metadata.chunk_index}"


def build_citation_from_metadata(metadata: dict[str, Any]) -> str:
    """Build a citation from a raw metadata dictionary.

    Args:
        metadata: Raw chunk metadata dictionary.

    Returns:
        Citation string.

    Raises:
        pydantic.ValidationError: If metadata does not match ``ChunkMetadata``.
    """
    chunk_metadata = ChunkMetadata.model_validate(metadata)
    return build_chunk_citation(chunk_metadata)
