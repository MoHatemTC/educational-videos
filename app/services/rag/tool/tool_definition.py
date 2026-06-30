"""LangChain StructuredTool registration for technical-document retrieval."""

from typing import Any

from langchain_core.tools import StructuredTool

from app.services.rag.tool.config import get_settings
from app.services.rag.tool.retriever import retrieve_chunks
from app.services.rag.tool.schema import RetrievalQuery


def retrieve_technical_docs_function(
    query: str,
    source: str | None = None,
    version: str | None = None,
    doc_type: str | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
    """Retrieve grounded technical-document chunks with citations.

    Args:
        query: Search query.
        source: Optional source metadata filter.
        version: Optional version metadata filter.
        doc_type: Optional document type metadata filter.
        top_k: Optional maximum number of chunks to return.
        similarity_threshold: Optional minimum similarity score.

    Returns:
        Structured retrieval output as a dictionary.
    """
    settings = get_settings()

    request = RetrievalQuery(
        query=query,
        source=source,
        version=version,
        doc_type=doc_type,
        top_k=top_k or settings.default_top_k,
        similarity_threshold=(
            similarity_threshold if similarity_threshold is not None else settings.default_similarity_threshold
        ),
    )

    return retrieve_chunks(request).model_dump(mode="json")


retrieve_technical_docs = StructuredTool.from_function(
    func=retrieve_technical_docs_function,
    name="retrieve_technical_docs",
    description=(
        "Retrieve grounded technical documentation chunks from the vector store. "
        "Use this for research questions that need cited technical context. "
        "Supports metadata filters using the shared ingestion contract: "
        "source, version, and doc_type. Returns cited chunks only."
    ),
    args_schema=RetrievalQuery,
)
