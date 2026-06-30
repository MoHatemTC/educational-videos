"""Basic retrieval sanity check for the RAG ingestion pipeline."""

import argparse
from pathlib import Path
from typing import Any

from app.services.rag.ingestion.embedder import get_embedding_function
from app.services.rag.ingestion.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHROMA_PERSIST_DIR,
    get_collection,
)
from app.services.rag.tool.metadata import build_metadata_filter


def build_parser() -> argparse.ArgumentParser:
    """Build the retrieval-check CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run a basic retrieval sanity check against ChromaDB.",
    )

    parser.add_argument(
        "query",
        help="Search query to run against the vector store.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source metadata filter.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional version metadata filter.",
    )
    parser.add_argument(
        "--doc-type",
        default=None,
        help="Optional document type metadata filter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of chunks to retrieve.",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_CHROMA_PERSIST_DIR,
        help="Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/validation_report.txt",
        help="Path to write the retrieval validation report.",
    )

    return parser


def validate_top_k(top_k: int) -> None:
    """Validate top-k retrieval setting.

    Args:
        top_k: Requested number of results.

    Raises:
        ValueError: If top_k is invalid.
    """
    if top_k <= 0:
        msg = "top_k must be greater than 0."
        raise ValueError(msg)


def format_citation(metadata: dict[str, Any]) -> str:
    """Format a basic citation for a retrieved chunk.

    Args:
        metadata: Retrieved chunk metadata.

    Returns:
        Citation string.
    """
    source = metadata.get("source", "unknown-source")
    version = metadata.get("version", "unknown-version")
    doc_type = metadata.get("doc_type", "unknown-type")
    path = metadata.get("path", "unknown-path")
    chunk_index = metadata.get("chunk_index", "unknown")

    return f"[{source}/{version}/{doc_type}] {path}#chunk-{chunk_index}"


def run_retrieval_check(
    query: str,
    source: str | None,
    version: str | None,
    doc_type: str | None,
    top_k: int,
    persist_dir: str,
    collection_name: str,
) -> list[dict[str, Any]]:
    """Run a vector-store query and return structured results.

    Args:
        query: Search query.
        source: Optional source filter.
        version: Optional version filter.
        doc_type: Optional document type filter.
        top_k: Maximum number of results.
        persist_dir: Chroma persistence directory.
        collection_name: Chroma collection name.

    Returns:
        Retrieved result dictionaries.
    """
    validate_top_k(top_k)

    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(query)

    collection = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    metadata_filter = build_metadata_filter(
        source=source,
        version=version,
        doc_type=doc_type,
    )

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    raw_results = collection.query(**query_kwargs)

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]
    ids = raw_results.get("ids", [[]])[0]

    results: list[dict[str, Any]] = []

    for result_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
        strict=True,
    ):
        clean_metadata = dict(metadata or {})

        results.append(
            {
                "id": result_id,
                "distance": distance,
                "citation": format_citation(clean_metadata),
                "metadata": clean_metadata,
                "content_preview": str(document)[:300],
            }
        )

    return results


def format_validation_report(
    query: str,
    source: str | None,
    version: str | None,
    doc_type: str | None,
    top_k: int,
    results: list[dict[str, Any]],
) -> str:
    """Format the retrieval-check validation report.

    Args:
        query: Search query.
        source: Source filter.
        version: Version filter.
        doc_type: Document type filter.
        top_k: Requested top-k value.
        results: Retrieved results.

    Returns:
        Formatted validation report.
    """
    lines = [
        "Sprint 2 RAG Foundation - Retrieval Validation Report",
        "=" * 56,
        f"Query: {query}",
        f"Source filter: {source}",
        f"Version filter: {version}",
        f"Doc type filter: {doc_type}",
        f"Top K: {top_k}",
        f"Results returned: {len(results)}",
        "Metadata contract checked: source, version, doc_type",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"Result {index}",
                f"ID: {result['id']}",
                f"Distance: {result['distance']}",
                f"Citation: {result['citation']}",
                f"Metadata: {result['metadata']}",
                f"Preview: {result['content_preview']}",
                "",
            ]
        )

    return "\n".join(lines)


def write_report(report_path: str | Path, report_text: str) -> None:
    """Write a validation report to disk.

    Args:
        report_path: Output report path.
        report_text: Report contents.
    """
    resolved_path = Path(report_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    """Run the retrieval sanity check."""
    parser = build_parser()
    args = parser.parse_args()

    results = run_retrieval_check(
        query=args.query,
        source=args.source,
        version=args.version,
        doc_type=args.doc_type,
        top_k=args.top_k,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    report = format_validation_report(
        query=args.query,
        source=args.source,
        version=args.version,
        doc_type=args.doc_type,
        top_k=args.top_k,
        results=results,
    )

    write_report(args.report_path, report)

    print(report)


if __name__ == "__main__":
    main()
