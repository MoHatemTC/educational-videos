"""Minimal LangGraph-compatible research agent for RAG retrieval."""

from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from rag_tool.tool_definition import retrieve_technical_docs


class ResearchAgentState(TypedDict):
    """State passed through the research-agent graph."""

    query: str
    source: NotRequired[str | None]
    version: NotRequired[str | None]
    doc_type: NotRequired[str | None]
    top_k: NotRequired[int]
    similarity_threshold: NotRequired[float]
    retrieval: NotRequired[dict[str, Any]]
    answer_context: NotRequired[str]


def build_tool_input(state: ResearchAgentState) -> dict[str, Any]:
    """Build the retrieval tool input from graph state.

    Args:
        state: Current research-agent state.

    Returns:
        Tool input dictionary.

    Raises:
        ValueError: If the state does not contain a valid query.
    """
    query = state.get("query")

    if not isinstance(query, str) or not query.strip():
        msg = "ResearchAgentState must include a non-empty query."
        raise ValueError(msg)

    tool_input: dict[str, Any] = {"query": query}

    for key in ("source", "version", "doc_type"):
        value = state.get(key)
        if value is not None:
            tool_input[key] = value

    if "top_k" in state:
        tool_input["top_k"] = state["top_k"]

    if "similarity_threshold" in state:
        tool_input["similarity_threshold"] = state["similarity_threshold"]

    return tool_input


def format_answer_context(retrieval: dict[str, Any]) -> str:
    """Format retrieved chunks as grounded context for downstream nodes.

    Args:
        retrieval: Structured retrieval dictionary returned by the tool.

    Returns:
        Readable cited context block.
    """
    results = retrieval.get("results")

    if not isinstance(results, list) or not results:
        return "No grounded context retrieved."

    lines: list[str] = []

    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue

        citation = str(result.get("citation", "unknown citation"))
        content = str(result.get("content", "")).strip()

        if not content:
            continue

        lines.extend(
            [
                f"Context {index}",
                f"Citation: {citation}",
                content,
                "",
            ]
        )

    return "\n".join(lines).strip() or "No grounded context retrieved."


def retrieve_context_node(state: ResearchAgentState) -> ResearchAgentState:
    """Retrieve grounded context for the research-agent graph.

    Args:
        state: Current graph state.

    Returns:
        Updated graph state containing retrieval output and answer context.
    """
    retrieval_output = retrieve_technical_docs.invoke(build_tool_input(state))

    if not isinstance(retrieval_output, dict):
        retrieval_output = {"raw_output": retrieval_output}

    return {
        "retrieval": retrieval_output,
        "answer_context": format_answer_context(retrieval_output),
    }


def build_research_graph() -> Any:
    """Build and compile the minimal research-agent graph.

    Returns:
        Compiled LangGraph graph.
    """
    graph = StateGraph(ResearchAgentState)

    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", END)

    return graph.compile()


def run_research_agent(
    query: str,
    source: str | None = None,
    version: str | None = None,
    doc_type: str | None = None,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
) -> ResearchAgentState:
    """Run the minimal research agent once.

    Args:
        query: Research query.
        source: Optional source metadata filter.
        version: Optional version metadata filter.
        doc_type: Optional document type metadata filter.
        top_k: Maximum number of chunks to retrieve.
        similarity_threshold: Minimum similarity score required.

    Returns:
        Final graph state.
    """
    graph = build_research_graph()

    initial_state: ResearchAgentState = {
        "query": query,
        "source": source,
        "version": version,
        "doc_type": doc_type,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
    }

    return cast(ResearchAgentState, graph.invoke(initial_state))


RESEARCH_AGENT_TOOLS = [retrieve_technical_docs]
