"""Metadata-aware Chroma retriever for the RAG tool."""

from pathlib import Path
from typing import Any, Final
from pydantic import ValidationError

from rag_tool.citation import build_chunk_citation
from ingestion.embedder import get_embedding_function
from ingestion.vector_store import (
    get_collection,
)
from rag_tool.metadata import ChunkMetadata, build_metadata_filter
from rag_tool.schema import RetrievalOutput, RetrievalQuery, RetrievedChunk

DEFAULT_FETCH_MULTIPLIER: Final[int] = 3
MAX_DISTANCE_FOR_SCORE: Final[float] = 2.0


def distance_to_score(distance: float) -> float:
    """Convert a Chroma distance value into a bounded similarity score.

    Args:
        distance: Raw distance returned by the vector database.

    Returns:
        Similarity score between 0.0 and 1.0, where higher is better.
    """
    bounded_distance = min(max(distance, 0.0), MAX_DISTANCE_FOR_SCORE)
    return 1.0 - (bounded_distance / MAX_DISTANCE_FOR_SCORE)


def get_first_result_list(raw_results: dict[str, Any], key: str) -> list[Any]:
    """Extract the first result list from a Chroma query response.

    Args:
        raw_results: Raw Chroma query response.
        key: Response key to extract.

    Returns:
        First nested result list, or an empty list.
    """
    value = raw_results.get(key)

    if not isinstance(value, list) or not value:
        return []

    first_value = value[0]

    if not isinstance(first_value, list):
        return []

    return first_value


def build_query_kwargs(
    query_embedding: list[float],
    request: RetrievalQuery,
    available_vectors: int,
) -> dict[str, Any]:
    """Build keyword arguments for a Chroma collection query.

    Args:
        query_embedding: Embedded query vector.
        request: Validated retrieval request.
        available_vectors: Number of vectors available in the collection.

    Returns:
        Keyword arguments for ``collection.query``.
    """
    requested_results = max(
        request.top_k,
        request.top_k * DEFAULT_FETCH_MULTIPLIER,
    )

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(requested_results, available_vectors),
    }

    metadata_filter = build_metadata_filter(
        source=request.source,
        version=request.version,
        doc_type=request.doc_type,
    )

    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    return query_kwargs


def retrieve_chunks(
    request: RetrievalQuery,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
) -> RetrievalOutput:
    """Retrieve cited chunks from ChromaDB using query and metadata filters.

    Args:
        request: Validated retrieval request.
        persist_dir: Chroma persistence directory.
        collection_name: Chroma collection name.

    Returns:
        Structured retrieval output.
    """
    collection = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    available_vectors = int(collection.count())

    if available_vectors == 0:
        return RetrievalOutput(
            query=request.query,
            filters_applied=request.active_filters(),
            results=[],
        )

    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(request.query)

    raw_results = collection.query(
        **build_query_kwargs(
            query_embedding=query_embedding,
            request=request,
            available_vectors=available_vectors,
        )
    )

    documents = get_first_result_list(raw_results, "documents")
    metadatas = get_first_result_list(raw_results, "metadatas")
    distances = get_first_result_list(raw_results, "distances")

    retrieved_chunks: list[RetrievedChunk] = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
        strict=False,
    ):
        score = distance_to_score(float(distance))

        if score < request.similarity_threshold:
            continue

        try:
            chunk_metadata = ChunkMetadata.model_validate(metadata)
        except ValidationError:
            continue

        retrieved_chunks.append(
            RetrievedChunk(
                content=str(document),
                score=score,
                citation=build_chunk_citation(chunk_metadata),
                metadata=chunk_metadata,
            )
        )

        if len(retrieved_chunks) >= request.top_k:
            break

    return RetrievalOutput(
        query=request.query,
        filters_applied=request.active_filters(),
        results=retrieved_chunks,
    )
