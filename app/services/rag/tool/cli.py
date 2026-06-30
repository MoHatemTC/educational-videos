"""Command-line interface for the RAG technical-document retriever."""

import argparse
import json
from typing import Any

from pydantic import ValidationError

from app.services.rag.ingestion.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_CHROMA_PERSIST_DIR
from app.services.rag.tool.retriever import retrieve_chunks
from app.services.rag.tool.schema import RetrievalOutput, RetrievalQuery


def build_parser() -> argparse.ArgumentParser:
    """Build the retrieval CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Retrieve cited technical-document chunks from ChromaDB.",
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
        default=5,
        help="Maximum number of chunks to return.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum similarity score required.",
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
        "--json",
        action="store_true",
        help="Print the full structured result as JSON.",
    )

    return parser


def build_request(args: argparse.Namespace) -> RetrievalQuery:
    """Build a validated retrieval request from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Validated retrieval query.
    """
    return RetrievalQuery(
        query=args.query,
        source=args.source,
        version=args.version,
        doc_type=args.doc_type,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )


def output_as_json(output: RetrievalOutput) -> str:
    """Serialize retrieval output as formatted JSON.

    Args:
        output: Structured retrieval output.

    Returns:
        Pretty-printed JSON string.
    """
    return json.dumps(
        output.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )


def output_as_text(output: RetrievalOutput) -> str:
    """Format retrieval output as readable CLI text.

    Args:
        output: Structured retrieval output.

    Returns:
        Human-readable retrieval summary.
    """
    lines = [
        "RAG Retrieval Results",
        "=" * 40,
        f"Query: {output.query}",
        f"Filters: {output.filters_applied}",
        f"Results: {output.result_count}",
        "",
    ]

    for index, result in enumerate(output.results, start=1):
        preview = result.content.replace("\n", " ")[:300]

        lines.extend(
            [
                f"Result {index}",
                f"Score: {result.score:.4f}",
                f"Citation: {result.citation}",
                f"Preview: {preview}",
                "",
            ]
        )

    return "\n".join(lines)


def run_cli(args: argparse.Namespace) -> str:
    """Run the retrieval CLI workflow.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Text to print to stdout.
    """
    request = build_request(args)

    output = retrieve_chunks(
        request=request,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    if args.json:
        return output_as_json(output)

    return output_as_text(output)


def main() -> None:
    """Run the command-line interface."""
    parser = build_parser()

    try:
        output = run_cli(parser.parse_args())
    except ValidationError as error:
        error_payload: dict[str, Any] = {"errors": error.errors()}
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        raise SystemExit(2) from error

    print(output)


if __name__ == "__main__":
    main()
