"""Document loading utilities for the RAG ingestion pipeline."""

from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document

from rag_tool.metadata import DocumentMetadata

SUPPORTED_SUFFIXES = frozenset(
    {
        ".md",
        ".mdx",
        ".txt",
        ".rst",
        ".json",
        ".yaml",
        ".yml",
    }
)

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "dist",
        "build",
        ".chroma",
    }
)


def normalize_suffixes(suffixes: Iterable[str] | None = None) -> frozenset[str]:
    """Normalize supported file suffixes to lowercase dot-prefixed values.

    Args:
        suffixes: Optional iterable of suffixes such as ``md`` or ``.md``.

    Returns:
        A frozen set of normalized suffix strings.
    """
    if suffixes is None:
        return SUPPORTED_SUFFIXES

    return frozenset(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in suffixes
    )


def is_ignored_path(path: Path, root: Path) -> bool:
    """Check whether a path is inside an ignored directory.

    Args:
        path: File or directory path to check.
        root: Root directory used to calculate relative path parts.

    Returns:
        True if the path should be ignored, otherwise False.
    """
    relative_parts = path.relative_to(root).parts
    return any(part in IGNORED_DIRS for part in relative_parts)


def iter_document_paths(
    docs_path: str | Path,
    suffixes: Iterable[str] | None = None,
) -> list[Path]:
    """Return supported documentation file paths under a directory.

    Args:
        docs_path: Root documentation directory.
        suffixes: Optional iterable of supported suffixes.

    Returns:
        Sorted list of supported file paths.

    Raises:
        FileNotFoundError: If the docs path does not exist.
        NotADirectoryError: If the docs path is not a directory.
    """
    root = Path(docs_path).expanduser().resolve()
    supported_suffixes = normalize_suffixes(suffixes)

    if not root.exists():
        msg = f"Documentation path does not exist: {root}"
        raise FileNotFoundError(msg)

    if not root.is_dir():
        msg = f"Documentation path is not a directory: {root}"
        raise NotADirectoryError(msg)

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_suffixes
        and not is_ignored_path(path, root)
    )


def read_text_file(path: Path) -> str:
    """Read a text file safely as UTF-8.

    Args:
        path: Text file path.

    Returns:
        File contents with undecodable characters replaced.
    """
    return path.read_text(encoding="utf-8", errors="replace").strip()


def build_document_metadata(
    file_path: Path,
    docs_root: Path,
    source: str,
    version: str,
) -> dict[str, str]:
    """Build metadata for a loaded documentation file.

    Args:
        file_path: Path to the loaded file.
        docs_root: Root documentation directory.
        source: Documentation source name.
        version: Documentation version.

    Returns:
        Metadata dictionary containing the shared contract fields plus path.
    """
    relative_path = file_path.relative_to(docs_root).as_posix()
    doc_type = file_path.suffix.lower().lstrip(".")

    metadata = DocumentMetadata(
        source=source,
        version=version,
        doc_type=doc_type,
    ).model_dump()

    metadata["path"] = relative_path
    return metadata


def load_documents(
    docs_path: str | Path,
    source: str = "qdrant",
    version: str = "master",
    suffixes: Iterable[str] | None = None,
) -> list[Document]:
    """Load supported documentation files as LangChain documents.

    Args:
        docs_path: Root documentation directory.
        source: Documentation source name.
        version: Documentation version.
        suffixes: Optional iterable of supported file suffixes.

    Returns:
        List of loaded LangChain ``Document`` objects.
    """
    docs_root = Path(docs_path).expanduser().resolve()
    file_paths = iter_document_paths(docs_root, suffixes)

    documents: list[Document] = []

    for file_path in file_paths:
        content = read_text_file(file_path)

        if not content:
            continue

        metadata = build_document_metadata(
            file_path=file_path,
            docs_root=docs_root,
            source=source,
            version=version,
        )

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents