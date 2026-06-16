"""LangChain StructuredTool definition for technical-document retrieval."""

from typing import Any

from langchain_core.tools import StructuredTool

from rag_tool.retriever import retrieve_chunks
from rag_tool.schema import RetrievalQuery


def retrieve_technical_docs_function(
    query: str,
    source: str | None = None,
    version: str | None = None,
    doc_type: str | None = None,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
) -> dict[str, Any]:
    """Retrieve grounded technical-document chunks with citations.

    Args:
        query: Search query.
        source: Optional source metadata filter.
        version: Optional version metadata filter.
        doc_type: Optional document type metadata filter.
        top_k: Maximum number of chunks to return.
        similarity_threshold: Minimum similarity score required.

    Returns:
        Structured retrieval output as a dictionary.
    """
    request = RetrievalQuery(
        query=query,
        source=source,
        version=version,
        doc_type=doc_type,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    return retrieve_chunks(request).model_dump()


retrieve_technical_docs = StructuredTool.from_function(
    func=retrieve_technical_docs_function,
    name="retrieve_technical_docs",
    description=(
        "Retrieve grounded technical documentation chunks from the vector store. "
        "Supports metadata filters for source, version, and doc_type. "
        "Use this when the research agent needs cited technical context."
    ),
    args_schema=RetrievalQuery,
)
