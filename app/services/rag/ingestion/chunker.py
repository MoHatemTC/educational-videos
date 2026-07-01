"""Document chunking utilities for the RAG ingestion pipeline."""

from collections.abc import Iterable
from hashlib import sha256
from typing import Any, Final

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.rag.tool.metadata import ChunkMetadata, DocumentMetadata

DEFAULT_CHUNK_SIZE: Final[int] = 900
DEFAULT_CHUNK_OVERLAP: Final[int] = 150

DEFAULT_SEPARATORS: Final[list[str]] = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    " ",
    "",
]


def hash_text(text: str) -> str:
    """Return a deterministic SHA-256 hash for text content.

    Args:
        text: Text content to hash.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return sha256(text.encode("utf-8")).hexdigest()


def build_chunk_id(
    source: str,
    version: str,
    path: str,
    chunk_index: int,
) -> str:
    """Build a stable deterministic chunk ID.

    Args:
        source: Documentation source name.
        version: Documentation version.
        path: Relative source file path.
        chunk_index: Zero-based chunk index inside the source file.

    Returns:
        Stable chunk ID suitable for idempotent vector-store upserts.
    """
    raw_id = f"{source}:{version}:{path}:{chunk_index}"
    return hash_text(raw_id)


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk size and overlap settings.

    Args:
        chunk_size: Maximum chunk size.
        chunk_overlap: Number of overlapping characters between chunks.

    Raises:
        ValueError: If chunking settings are invalid.
    """
    if chunk_size <= 0:
        msg = "chunk_size must be greater than 0."
        raise ValueError(msg)

    if chunk_overlap < 0:
        msg = "chunk_overlap cannot be negative."
        raise ValueError(msg)

    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size."
        raise ValueError(msg)


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create the recursive text splitter used by ingestion.

    Args:
        chunk_size: Maximum chunk size.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        Configured recursive character text splitter.
    """
    validate_chunk_settings(chunk_size, chunk_overlap)

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        keep_separator=True,
    )


def get_document_path(metadata: dict[str, Any]) -> str:
    """Extract and validate the relative source path from metadata.

    Args:
        metadata: Document metadata dictionary.

    Returns:
        Relative source path.

    Raises:
        ValueError: If the path is missing or invalid.
    """
    path = metadata.get("path")

    if not isinstance(path, str) or not path:
        msg = "Document metadata must include a non-empty 'path' field."
        raise ValueError(msg)

    return path


def build_chunk_metadata(
    document_metadata: dict[str, Any],
    chunk_content: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Build validated metadata for one chunk.

    Args:
        document_metadata: Metadata from the source document.
        chunk_content: Chunk text content.
        chunk_index: Zero-based chunk index.

    Returns:
        Metadata dictionary for the chunk.
    """
    base_metadata = DocumentMetadata.model_validate(document_metadata)
    path = get_document_path(document_metadata)

    chunk_id = build_chunk_id(
        source=base_metadata.source,
        version=base_metadata.version,
        path=path,
        chunk_index=chunk_index,
    )

    return ChunkMetadata(
        source=base_metadata.source,
        version=base_metadata.version,
        doc_type=base_metadata.doc_type,
        path=path,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        content_hash=hash_text(chunk_content),
    ).model_dump()


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split loaded documents into metadata-rich chunks.

    Args:
        documents: Loaded LangChain documents.
        chunk_size: Maximum chunk size.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked LangChain documents.
    """
    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []

    for document in documents:
        split_texts = [text.strip() for text in splitter.split_text(document.page_content) if text.strip()]

        for chunk_index, chunk_content in enumerate(split_texts):
            chunks.append(
                Document(
                    page_content=chunk_content,
                    metadata=build_chunk_metadata(
                        document_metadata=document.metadata,
                        chunk_content=chunk_content,
                        chunk_index=chunk_index,
                    ),
                )
            )

    return chunks
