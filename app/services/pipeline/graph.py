"""LangGraph workflow for the educational-video generation pipeline.

This module owns the generation half of the video pipeline. Rendering stays in
``orchestrator.py`` because it is approval-dependent output assembly, while this
state graph covers research, code generation, sandbox validation/repair,
narration, visual planning, web-explainer vision nodes, and the human approval
interrupt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Command, CompiledStateGraph
from langgraph.types import interrupt

from app.core.config import settings
from app.core.llm_client import LLMClient
from app.core.logging import logger
from app.core.prompt_chain import convert_script_to_timeline
from app.models.video_job import VideoJob
from app.services.pipeline.agents import generate_code, generate_script, research_topic
from app.services.pipeline.evaluation import evaluate_pipeline_artifacts
from app.services.pipeline.llm import PipelineLLM
from app.services.pipeline.rag import retrieve_grounding_context
from app.services.pipeline.sandbox.parser import parse_traceback
from app.services.pipeline.sandbox.runner import run_code
from app.services.pipeline.vision import (
    capture_page,
    capture_page_with_actions,
    describe_screenshots,
    generate_web_script,
)
from app.services.video_store import video_store

_MAX_SANDBOX_REPAIRS = 3
_REPAIR_SYSTEM = (
    "You are a Python debugging assistant. You fix the given code so it runs without error and prints output. "
    "Return ONLY the corrected, complete Python code."
)
_CHECKPOINTER = MemorySaver()
_GRAPH: CompiledStateGraph | None = None


class PipelineState(TypedDict, total=False):
    """Typed LangGraph state shared by all educational-video nodes."""

    job_id: str
    mode: str
    topic: str
    language: str
    url: str | None
    grounding_context: str
    rag_context: dict[str, Any]
    citations: list[str]
    research_notes: str
    code: str
    code_validated: bool
    code_output: str
    sandbox_log: list[dict[str, Any]]
    sandbox_attempt: int
    last_sandbox_error: dict[str, Any] | None
    script: str
    pipeline_eval_scores: dict[str, Any]
    timeline: dict[str, Any] | None
    timeline_error: str | None
    screenshots: list[str]
    description: str
    visual_plan: dict[str, Any]
    raw_script: str
    vision_actions: list[dict[str, Any]]
    review_status: str


ReviewDecision = dict[str, Any]
Route = Literal["research", "web_capture", "visual_planning", "sandbox_repair", "script"]
WebCaptureRoute = Literal["web_describe", "web_visual_planning"]
StateUpdate = dict[str, Any]


def _required_value(state: PipelineState, key: str) -> Any:
    """Return a required graph-state value or fail with a clear pipeline error."""
    value = state.get(key)
    if value is None:
        raise RuntimeError(f"pipeline state missing required key: {key}")
    return value


def _required_str(state: PipelineState, key: str) -> str:
    """Return a required string graph-state value."""
    value = _required_value(state, key)
    if not isinstance(value, str):
        raise RuntimeError(f"pipeline state key {key!r} must be a string")
    return value


def _required_string_list(state: PipelineState, key: str) -> list[str]:
    """Return a required list-of-strings graph-state value."""
    value = _required_value(state, key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"pipeline state key {key!r} must be a list of strings")
    return cast(list[str], value)


def _optional_str(state: PipelineState, key: str) -> str | None:
    """Return an optional string graph-state value."""
    value = state.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"pipeline state key {key!r} must be a string when present")
    return value


def _graph_config(job_id: str) -> RunnableConfig:
    """Return the deterministic LangGraph config for one video job thread."""
    return {
        "configurable": {"thread_id": job_id},
        "metadata": {
            "job_id": job_id,
            "component": "video_pipeline",
            "langfuse_tags": ["video-pipeline", "langgraph", settings.ENVIRONMENT.value],
        },
    }


def _llm(state: PipelineState) -> PipelineLLM:
    """Create a traced pipeline LLM client for the current job."""
    return PipelineLLM(job_id=_required_str(state, "job_id"))


def _job_or_raise(job_id: str) -> VideoJob:
    """Fetch a video job or raise a hard pipeline error."""
    job = video_store.get_job(job_id)
    if job is None:
        raise RuntimeError(f"video job {job_id} not found")
    return job


def _strip_fences(text: str) -> str:
    """Remove a simple surrounding Python Markdown fence from model output."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _correction_prompt(code: str, traceback_fields: dict[str, Any] | None, attempt: int) -> str:
    """Build the sandbox repair prompt from parsed traceback fields only."""
    tb = traceback_fields or {}
    return (
        f"Attempt {attempt}: the following Python code failed when executed.\n\n"
        f"Exception type: {tb.get('exception_type')}\n"
        f"Exception message: {tb.get('exception_message')}\n"
        f"Failing line number: {tb.get('line')}\n"
        f"Offending source line: {tb.get('innermost_frame')}\n\n"
        f"Original code:\n{code}\n\n"
        "Return the corrected, complete Python code only — no fences, no commentary. "
        "Keep it minimal and standard-library only, and make sure it prints illustrative output."
    )


def _load_job(state: PipelineState) -> StateUpdate:
    """Load the latest job row and seed graph state with immutable job inputs."""
    job = _job_or_raise(_required_str(state, "job_id"))
    artifacts = job.artifacts or {}
    raw_script = artifacts.get("raw_script")
    vision_actions = artifacts.get("vision_actions")
    update: StateUpdate = {
        "mode": job.mode,
        "topic": job.topic,
        "language": job.language,
        "url": job.url,
        "sandbox_attempt": 0,
        "sandbox_log": [],
    }
    if isinstance(raw_script, str) and raw_script.strip():
        update["raw_script"] = raw_script.strip()
        update["script"] = raw_script.strip()
    if isinstance(vision_actions, list):
        update["vision_actions"] = vision_actions

    video_store.update_job(job.id, status="running", current_step="routing")
    return update


def _route_mode(state: PipelineState) -> Route:
    """Route jobs into the code tutorial, raw-script, or web explainer branch."""
    if state.get("mode") == "web_explainer":
        return "web_capture"
    if isinstance(state.get("raw_script"), str) and str(state.get("raw_script")).strip():
        return "visual_planning"
    return "research"


def _research_node(state: PipelineState) -> StateUpdate:
    """Retrieve grounding context and produce research notes."""
    job_id = _required_str(state, "job_id")
    video_store.update_job(job_id, status="running", current_step="research")

    grounding = retrieve_grounding_context(_required_str(state, "topic"), _required_str(state, "language"))
    prompt_context = grounding.format_for_prompt()
    research_notes = research_topic(
        _llm(state),
        _required_str(state, "topic"),
        _required_str(state, "language"),
        grounding_context=prompt_context,
    )
    rag_context = grounding.to_artifact()

    video_store.update_job(
        job_id,
        current_step="code",
        artifacts_merge={
            "research": research_notes,
            "rag_context": rag_context,
            "citations": grounding.citations,
        },
    )
    return {
        "grounding_context": prompt_context,
        "research_notes": research_notes,
        "rag_context": rag_context,
        "citations": grounding.citations,
    }


def _code_node(state: PipelineState) -> StateUpdate:
    """Generate the first runnable code draft from research notes."""
    code = generate_code(
        _llm(state),
        _required_str(state, "topic"),
        _required_str(state, "research_notes"),
        grounding_context=state.get("grounding_context"),
    )
    video_store.update_job(_required_str(state, "job_id"), current_step="sandbox", artifacts_merge={"code": code})
    return {"code": code, "sandbox_attempt": 0, "sandbox_log": []}


def _sandbox_node(state: PipelineState) -> StateUpdate:
    """Run the current code draft and record whether it validates."""
    code = _required_str(state, "code")
    result = run_code(code)
    failure = None if result.ok else parse_traceback(result.stderr)
    attempt = int(state.get("sandbox_attempt", 0))
    sandbox_log = list(state.get("sandbox_log", []))
    sandbox_log.append(
        {
            "iteration": attempt,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "exception_type": (failure or {}).get("exception_type"),
            "correction_attempted": attempt > 0,
        }
    )

    artifacts: dict[str, Any] = {
        "code": code,
        "code_validated": result.ok,
        "code_output": result.stdout[:2000],
        "sandbox_log": sandbox_log,
    }
    job_id = _required_str(state, "job_id")
    video_store.update_job(job_id, current_step="sandbox", artifacts_merge=artifacts)

    if result.ok:
        logger.info("pipeline_graph_sandbox_ok", job_id=job_id, iteration=attempt)
    elif attempt >= _MAX_SANDBOX_REPAIRS:
        logger.warning("pipeline_graph_sandbox_exhausted", job_id=job_id, attempts=attempt + 1)

    return {
        "code_validated": result.ok,
        "code_output": result.stdout[:2000],
        "sandbox_log": sandbox_log,
        "last_sandbox_error": failure,
    }


def _route_sandbox(state: PipelineState) -> Route:
    """Loop through repair while attempts remain; otherwise continue to script."""
    if state.get("code_validated"):
        return "script"
    if int(state.get("sandbox_attempt", 0)) >= _MAX_SANDBOX_REPAIRS:
        return "script"
    return "sandbox_repair"


def _sandbox_repair_node(state: PipelineState) -> StateUpdate:
    """Ask the LLM to repair failed code before routing back to sandbox."""
    next_attempt = int(state.get("sandbox_attempt", 0)) + 1
    corrected = _strip_fences(
        _llm(state).complete(
            stage="sandbox_repair",
            system=_REPAIR_SYSTEM,
            user=_correction_prompt(_required_str(state, "code"), state.get("last_sandbox_error"), next_attempt),
        )
    )
    code = corrected or _required_str(state, "code")
    video_store.update_job(
        _required_str(state, "job_id"),
        current_step="sandbox_repair",
        artifacts_merge={"code": code},
    )
    return {"code": code, "sandbox_attempt": next_attempt}


def _script_node(state: PipelineState) -> StateUpdate:
    """Generate narration and evaluate the produced code/script artifacts."""
    job_id = _required_str(state, "job_id")
    video_store.update_job(job_id, current_step="script")
    script = generate_script(
        _llm(state),
        _required_str(state, "topic"),
        _required_str(state, "research_notes"),
        _required_str(state, "code"),
        _required_str(state, "language"),
        grounding_context=state.get("grounding_context"),
    )
    pipeline_eval_scores = evaluate_pipeline_artifacts(
        topic=_required_str(state, "topic"),
        code=_required_str(state, "code"),
        script=script,
        rag_context=state.get("rag_context", {}),
    )
    video_store.update_job(
        job_id,
        current_step="visual_planning",
        artifacts_merge={"script": script, "pipeline_eval_scores": pipeline_eval_scores},
    )
    return {"script": script, "pipeline_eval_scores": pipeline_eval_scores}


def _visual_planning_node(state: PipelineState) -> StateUpdate:
    """Convert narration into a code-typing visual timeline as a graph node."""
    timeline: dict[str, Any] | None
    timeline_error: str | None
    try:
        timeline = cast(
            dict[str, Any],
            convert_script_to_timeline(_required_str(state, "script"), LLMClient()).model_dump(),
        )
        timeline_error = None
    except Exception as exc:  # noqa: BLE001 - timeline failure should not kill generation
        logger.warning("timeline_generation_failed", job_id=_required_str(state, "job_id"), error=str(exc))
        timeline = None
        timeline_error = str(exc)

    video_store.update_job(
        _required_str(state, "job_id"),
        current_step="awaiting_approval",
        artifacts_merge={"timeline": timeline, "timeline_error": timeline_error},
    )
    return {"timeline": timeline, "timeline_error": timeline_error}


def _web_capture_node(state: PipelineState) -> StateUpdate:
    """Capture screenshots for a web explainer job, optionally after coordinate actions."""
    job_id = _required_str(state, "job_id")
    video_store.update_job(job_id, status="running", current_step="web_capture")
    shots_dir = Path(settings.VIDEO_DATA_DIR) / "screenshots" / job_id
    actions = state.get("vision_actions")
    if isinstance(actions, list) and actions:
        screenshots = capture_page_with_actions(_optional_str(state, "url") or "", shots_dir, actions)
        visual_driver: dict[str, Any] = {"kind": "coordinate_action_loop", "actions": actions}
    else:
        screenshots = capture_page(_optional_str(state, "url") or "", shots_dir)
        visual_driver = {"kind": "screenshot_capture"}
    video_store.update_job(
        job_id,
        current_step="web_describe",
        artifacts_merge={"screenshots": screenshots, "vision_driver": visual_driver},
    )
    return {"screenshots": screenshots}


def _route_web_capture(state: PipelineState) -> WebCaptureRoute:
    """Skip vision description/script generation when a raw web script was supplied."""
    if isinstance(state.get("raw_script"), str) and str(state.get("raw_script")).strip():
        return "web_visual_planning"
    return "web_describe"


def _web_describe_node(state: PipelineState) -> StateUpdate:
    """Describe captured screenshots with the vision model."""
    description = describe_screenshots(
        _required_string_list(state, "screenshots"),
        _optional_str(state, "url") or "",
        job_id=_required_str(state, "job_id"),
    )
    video_store.update_job(
        _required_str(state, "job_id"),
        current_step="web_script",
        artifacts_merge={"research": description, "code": None},
    )
    return {"description": description}


def _web_script_node(state: PipelineState) -> StateUpdate:
    """Generate narration for the screenshot-driven web explainer branch."""
    script = generate_web_script(
        _llm(state),
        _optional_str(state, "url") or "",
        _required_str(state, "description"),
        _required_str(state, "language"),
    )
    video_store.update_job(
        _required_str(state, "job_id"),
        current_step="web_visual_planning",
        artifacts_merge={"script": script},
    )
    return {"script": script}


def _web_visual_planning_node(state: PipelineState) -> StateUpdate:
    """Create a visual plan for the Ken-Burns screenshot renderer."""
    visual_plan: dict[str, Any] = {
        "kind": "screenshot_ken_burns",
        "screenshots": state.get("screenshots", []),
        "timeline": None,
    }
    if isinstance(state.get("vision_actions"), list) and state.get("vision_actions"):
        visual_plan["driver"] = {"kind": "coordinate_action_loop", "actions": state.get("vision_actions", [])}
    video_store.update_job(
        _required_str(state, "job_id"),
        current_step="awaiting_approval",
        artifacts_merge={"timeline": None, "timeline_error": None, "visual_plan": visual_plan},
    )
    return {"timeline": None, "timeline_error": None, "visual_plan": visual_plan}


def _apply_reviewer_edits(job_id: str, decision: ReviewDecision) -> None:
    """Merge reviewer edits supplied by the resume command."""
    edits = decision.get("artifacts")
    if not isinstance(edits, dict):
        return

    allowed = {"script", "code", "timeline", "tts_settings", "tts_segments"}
    filtered_edits = {key: value for key, value in edits.items() if key in allowed}
    if filtered_edits:
        video_store.update_job(job_id, artifacts_merge=filtered_edits)


def _approval_gate_node(state: PipelineState) -> Command:
    """Pause generation for human approval, then resume with the API decision."""
    job_id = _required_str(state, "job_id")
    video_store.update_job(
        job_id,
        status="awaiting_approval",
        current_step="awaiting_approval",
        awaiting_approval=True,
        review_status="pending",
    )
    decision = interrupt(
        {
            "job_id": job_id,
            "mode": state.get("mode"),
            "status": "awaiting_approval",
            "message": "Review the generated artifacts, then approve or reject the job.",
        }
    )
    normalized = decision if isinstance(decision, dict) else {"approved": bool(decision)}
    if not normalized.get("approved", True):
        reason = str(normalized.get("reason") or "rejected by reviewer")
        video_store.update_job(
            job_id,
            status="rejected",
            current_step="rejected",
            awaiting_approval=False,
            review_status="rejected",
            error_message=reason,
        )
        return Command(update={"review_status": "rejected"}, goto=END)

    _apply_reviewer_edits(job_id, normalized)
    video_store.update_job(
        job_id,
        status="approved",
        current_step="approved",
        awaiting_approval=False,
        review_status="approved",
        error_message=None,
    )
    return Command(update={"review_status": "approved"}, goto=END)


def _build_graph() -> CompiledStateGraph:
    """Compile the stateful video pipeline graph with its checkpointer."""
    graph = StateGraph(PipelineState)
    graph.add_node("load_job", _load_job)
    graph.add_node("research", _research_node)
    graph.add_node("code", _code_node)
    graph.add_node("sandbox", _sandbox_node)
    graph.add_node("sandbox_repair", _sandbox_repair_node)
    graph.add_node("script", _script_node)
    graph.add_node("visual_planning", _visual_planning_node)
    graph.add_node("web_capture", _web_capture_node)
    graph.add_node("web_describe", _web_describe_node)
    graph.add_node("web_script", _web_script_node)
    graph.add_node("web_visual_planning", _web_visual_planning_node)
    graph.add_node("approval_gate", _approval_gate_node, destinations=(END,))

    graph.set_entry_point("load_job")
    graph.add_conditional_edges(
        "load_job",
        _route_mode,
        {"research": "research", "web_capture": "web_capture", "visual_planning": "visual_planning"},
    )
    graph.add_edge("research", "code")
    graph.add_edge("code", "sandbox")
    graph.add_conditional_edges("sandbox", _route_sandbox, {"sandbox_repair": "sandbox_repair", "script": "script"})
    graph.add_edge("sandbox_repair", "sandbox")
    graph.add_edge("script", "visual_planning")
    graph.add_edge("visual_planning", "approval_gate")
    graph.add_conditional_edges(
        "web_capture",
        _route_web_capture,
        {"web_describe": "web_describe", "web_visual_planning": "web_visual_planning"},
    )
    graph.add_edge("web_describe", "web_script")
    graph.add_edge("web_script", "web_visual_planning")
    graph.add_edge("web_visual_planning", "approval_gate")
    return graph.compile(checkpointer=_CHECKPOINTER, name="Educational Video Pipeline")


def _get_graph() -> CompiledStateGraph:
    """Return the singleton compiled pipeline graph."""
    global _GRAPH  # noqa: PLW0603 - singleton avoids rebuilding nodes per job
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def invoke_generation_graph(job_id: str) -> None:
    """Run a video job graph until the human approval interrupt."""
    _job_or_raise(job_id)
    logger.info("pipeline_graph_generation_started", job_id=job_id)
    try:
        _get_graph().invoke({"job_id": job_id}, config=_graph_config(job_id))
    except GraphInterrupt:
        logger.info("pipeline_graph_generation_interrupted", job_id=job_id)


def resume_generation_graph(
    job_id: str,
    *,
    reviewer_edits: dict[str, Any] | None = None,
    approved: bool = True,
    rejection_reason: str | None = None,
) -> VideoJob | None:
    """Resume a paused video job graph with the reviewer decision."""
    decision: ReviewDecision = {
        "approved": approved,
        "artifacts": reviewer_edits or {},
        "reason": rejection_reason,
    }
    logger.info("pipeline_graph_resume_requested", job_id=job_id, approved=approved)
    _get_graph().invoke(Command(resume=decision), config=_graph_config(job_id))
    return video_store.get_job(job_id)
