"""ChromaDB vector-store utilities for the RAG ingestion pipeline."""

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import chromadb
from chromadb.api import ClientAPI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_CHROMA_PERSIST_DIR: Final[str] = ".chroma"
DEFAULT_COLLECTION_NAME: Final[str] = "technical_docs"
DEFAULT_UPSERT_BATCH_SIZE: Final[int] = 64

MetadataValue = str | int | float | bool
CleanMetadata = dict[str, MetadataValue]


@dataclass(frozen=True)
class UpsertStats:
    """Summary statistics returned after a vector-store upsert."""

    submitted: int
    count_before: int
    count_after: int

    @property
    def net_new_vectors(self) -> int:
        """Return the number of new vector IDs added to the collection."""
        return self.count_after - self.count_before


def resolve_persist_dir(persist_dir: str | Path | None = None) -> Path:
    """Resolve the Chroma persistence directory.

    Args:
        persist_dir: Optional explicit persistence directory.

    Returns:
        Absolute path to the persistence directory.
    """
    raw_persist_dir = (
        str(persist_dir) if persist_dir is not None else os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_PERSIST_DIR)
    )

    return Path(raw_persist_dir).expanduser().resolve()


def resolve_collection_name(collection_name: str | None = None) -> str:
    """Resolve the Chroma collection name.

    Args:
        collection_name: Optional explicit collection name.

    Returns:
        Chroma collection name.

    Raises:
        ValueError: If the resolved collection name is empty.
    """
    resolved_name = collection_name or os.getenv("CHROMA_COLLECTION") or DEFAULT_COLLECTION_NAME

    if not resolved_name.strip():
        msg = "Chroma collection name must be non-empty."
        raise ValueError(msg)

    return resolved_name


def create_chroma_client(persist_dir: str | Path | None = None) -> ClientAPI:
    """Create a persistent ChromaDB client.

    Args:
        persist_dir: Optional explicit persistence directory.

    Returns:
        Persistent ChromaDB client.
    """
    resolved_dir = resolve_persist_dir(persist_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(resolved_dir))


def get_collection(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
) -> Any:
    """Create or load the configured ChromaDB collection.

    Args:
        persist_dir: Optional explicit persistence directory.
        collection_name: Optional explicit collection name.

    Returns:
        ChromaDB collection object.
    """
    client = create_chroma_client(persist_dir)
    resolved_name = resolve_collection_name(collection_name)

    return client.get_or_create_collection(name=resolved_name)


def clean_metadata(metadata: dict[str, Any]) -> CleanMetadata:
    """Convert document metadata to Chroma-compatible scalar values.

    Args:
        metadata: Raw metadata dictionary.

    Returns:
        Metadata dictionary containing only Chroma-compatible scalar values.
    """
    clean: CleanMetadata = {}

    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        elif value is not None:
            clean[key] = str(value)

    return clean


def get_chunk_id(document: Document) -> str:
    """Extract the deterministic chunk ID from a document.

    Args:
        document: Chunked LangChain document.

    Returns:
        Chunk ID string.

    Raises:
        ValueError: If the document has no valid chunk ID.
    """
    chunk_id = document.metadata.get("chunk_id")

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        msg = "Each chunk document must include a non-empty 'chunk_id'."
        raise ValueError(msg)

    return chunk_id


def iter_batches[T](items: list[T], batch_size: int) -> Iterator[list[T]]:
    """Yield list batches.

    Args:
        items: Items to batch.
        batch_size: Number of items per batch.

    Yields:
        Batches of items.

    Raises:
        ValueError: If batch size is invalid.
    """
    if batch_size <= 0:
        msg = "batch_size must be greater than 0."
        raise ValueError(msg)

    for start_index in range(0, len(items), batch_size):
        yield items[start_index : start_index + batch_size]


def count_vectors(collection: Any) -> int:
    """Return the number of vectors in a Chroma collection.

    Args:
        collection: ChromaDB collection object.

    Returns:
        Number of vectors in the collection.
    """
    return int(collection.count())


def upsert_documents(
    documents: Iterable[Document],
    embedding_function: Embeddings,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
) -> UpsertStats:
    """Embed and upsert chunk documents into ChromaDB.

    Args:
        documents: Chunked LangChain documents.
        embedding_function: Shared embedding function.
        persist_dir: Optional explicit Chroma persistence directory.
        collection_name: Optional explicit Chroma collection name.
        batch_size: Number of chunks to upsert per batch.

    Returns:
        Upsert summary statistics.
    """
    collection = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    document_list = list(documents)
    count_before = count_vectors(collection)

    if not document_list:
        return UpsertStats(
            submitted=0,
            count_before=count_before,
            count_after=count_before,
        )

    for batch in iter_batches(document_list, batch_size):
        ids = [get_chunk_id(document) for document in batch]
        texts = [document.page_content for document in batch]
        metadatas = [clean_metadata(document.metadata) for document in batch]
        embeddings = embedding_function.embed_documents(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    count_after = count_vectors(collection)

    return UpsertStats(
        submitted=len(document_list),
        count_before=count_before,
        count_after=count_after,
    )
