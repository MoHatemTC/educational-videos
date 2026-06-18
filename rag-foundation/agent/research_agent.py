"""LangGraph StateGraph agent loop for technical-document research."""

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
    tool_input: NotRequired[dict[str, Any]]
    retrieval: NotRequired[dict[str, Any]]
    answer_context: NotRequired[str]


def prepare_tool_input_node(state: ResearchAgentState) -> ResearchAgentState:
    """Prepare StructuredTool input from the graph state.

    Args:
        state: Current graph state.

    Returns:
        Updated graph state containing tool input.

    Raises:
        ValueError: If query is missing.
    """
    query = state.get("query")

    if not isinstance(query, str) or not query.strip():
        msg = "Research agent state must include a non-empty query."
        raise ValueError(msg)

    tool_input: dict[str, Any] = {"query": query}

    for key in ("source", "version", "doc_type", "top_k", "similarity_threshold"):
        value = state.get(key)
        if value is not None:
            tool_input[key] = value

    return {
        **state,
        "tool_input": tool_input,
    }


def call_retrieval_tool_node(state: ResearchAgentState) -> ResearchAgentState:
    """Call the registered StructuredTool inside the graph loop.

    Args:
        state: Current graph state.

    Returns:
        Updated graph state containing retrieval output.

    Raises:
        ValueError: If tool input is missing.
    """
    tool_input = state.get("tool_input")

    if not isinstance(tool_input, dict):
        msg = "Research agent state must include prepared tool_input."
        raise ValueError(msg)

    retrieval_output = retrieve_technical_docs.invoke(tool_input)

    if not isinstance(retrieval_output, dict):
        retrieval_output = {"raw_output": retrieval_output}

    return {
        **state,
        "retrieval": retrieval_output,
    }


def format_answer_context(retrieval: dict[str, Any]) -> str:
    """Format retrieved chunks as cited context.

    Args:
        retrieval: Structured retrieval dictionary.

    Returns:
        Cited context block for downstream agent nodes.
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


def prepare_context_node(state: ResearchAgentState) -> ResearchAgentState:
    """Prepare final cited context from retrieval results.

    Args:
        state: Current graph state.

    Returns:
        Updated graph state containing answer context.
    """
    retrieval = state.get("retrieval")

    if not isinstance(retrieval, dict):
        retrieval = {}

    return {
        **state,
        "answer_context": format_answer_context(retrieval),
    }


def build_research_graph() -> Any:
    """Build and compile the explicit LangGraph StateGraph loop.

    Returns:
        Compiled LangGraph graph.
    """
    graph = StateGraph(ResearchAgentState)

    graph.add_node("prepare_tool_input", prepare_tool_input_node)
    graph.add_node("call_retrieval_tool", call_retrieval_tool_node)
    graph.add_node("prepare_context", prepare_context_node)

    graph.add_edge(START, "prepare_tool_input")
    graph.add_edge("prepare_tool_input", "call_retrieval_tool")
    graph.add_edge("call_retrieval_tool", "prepare_context")
    graph.add_edge("prepare_context", END)

    return graph.compile()


def run_research_agent(
    query: str,
    source: str | None = None,
    version: str | None = None,
    doc_type: str | None = None,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
) -> ResearchAgentState:
    """Run the research agent once.

    Args:
        query: Research query.
        source: Optional source metadata filter.
        version: Optional version metadata filter.
        doc_type: Optional document type metadata filter.
        top_k: Maximum number of chunks to retrieve.
        similarity_threshold: Minimum similarity score.

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
