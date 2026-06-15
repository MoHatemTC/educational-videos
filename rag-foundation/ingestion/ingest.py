"""CLI entrypoint for loading, chunking, embedding, and upserting documents."""

import argparse
from pathlib import Path

from ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from ingestion.embedder import get_embedding_function
from ingestion.loader import load_documents
from ingestion.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHROMA_PERSIST_DIR,
    UpsertStats,
    upsert_documents,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Ingest technical documentation into ChromaDB.",
    )

    parser.add_argument(
        "--docs-path",
        required=True,
        help="Path to the technical documentation corpus.",
    )
    parser.add_argument(
        "--source",
        default="qdrant",
        help="Documentation source metadata value.",
    )
    parser.add_argument(
        "--version",
        default="master",
        help="Documentation version metadata value.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap size.",
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
        default="reports/ingestion_report.txt",
        help="Path to write the ingestion report.",
    )

    return parser


def format_ingestion_report(
    docs_path: str,
    source: str,
    version: str,
    chunk_size: int,
    chunk_overlap: int,
    documents_loaded: int,
    chunks_created: int,
    stats: UpsertStats,
    collection_name: str,
    persist_dir: str,
) -> str:
    """Format a plain-text ingestion report.

    Args:
        docs_path: Documentation corpus path.
        source: Source metadata value.
        version: Version metadata value.
        chunk_size: Chunk size used.
        chunk_overlap: Chunk overlap used.
        documents_loaded: Number of loaded documents.
        chunks_created: Number of generated chunks.
        stats: Vector-store upsert statistics.
        collection_name: Chroma collection name.
        persist_dir: Chroma persistence directory.

    Returns:
        Formatted ingestion report text.
    """
    duplicate_prevented = max(0, stats.submitted - stats.net_new_vectors)

    return "\n".join(
        [
            "Sprint 2 RAG Foundation - Ingestion Report",
            "=" * 48,
            f"Corpus path: {docs_path}",
            f"Source: {source}",
            f"Version: {version}",
            "Vector database: ChromaDB",
            f"Collection name: {collection_name}",
            f"Persist directory: {persist_dir}",
            "Embedding model: sentence-transformers/all-MiniLM-L6-v2",
            f"Chunk size: {chunk_size}",
            f"Chunk overlap: {chunk_overlap}",
            f"Documents loaded: {documents_loaded}",
            f"Chunks created: {chunks_created}",
            f"Vectors submitted: {stats.submitted}",
            f"Vector count before: {stats.count_before}",
            f"Vector count after: {stats.count_after}",
            f"Net new vectors: {stats.net_new_vectors}",
            f"Duplicate vectors prevented: {duplicate_prevented}",
            "Metadata contract: source, version, doc_type",
            "",
        ]
    )


def write_report(report_path: str | Path, report_text: str) -> None:
    """Write a report to disk.

    Args:
        report_path: Output report path.
        report_text: Report contents.
    """
    resolved_path = Path(report_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    """Run the ingestion pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    documents = load_documents(
        docs_path=args.docs_path,
        source=args.source,
        version=args.version,
    )

    chunks = chunk_documents(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    embedding_function = get_embedding_function()

    stats = upsert_documents(
        documents=chunks,
        embedding_function=embedding_function,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    report = format_ingestion_report(
        docs_path=args.docs_path,
        source=args.source,
        version=args.version,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        stats=stats,
        collection_name=args.collection_name,
        persist_dir=args.persist_dir,
    )

    write_report(args.report_path, report)

    print(report)


if __name__ == "__main__":
    main()
