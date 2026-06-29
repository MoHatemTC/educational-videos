"""Pipeline orchestrator.

Two job modes share one lifecycle (generate -> HITL approve -> render -> mp4):

* ``code_tutorial`` — research -> code -> self-healing sandbox -> script -> timeline,
  then render a code-typing video.
* ``web_explainer`` — navigate a URL -> screenshot -> Kimi-vision description ->
  script, then render a Ken-Burns video over the screenshots.

Both run as FastAPI background tasks. Every LLM/vision call is traced.
"""

import shutil
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.llm_client import LLMClient
from app.core.logging import logger
from app.core.prompt_chain import convert_script_to_timeline
from app.models.video_job import VideoJob
from app.services.pipeline.agents import generate_code, generate_script, research_topic
from app.services.pipeline.llm import PipelineLLM
from app.services.pipeline.render.ffmpeg_render import assemble_video
from app.services.pipeline.render.frames import render_frames
from app.services.pipeline.render.screenshot_video import render_screenshot_video
from app.services.pipeline.sandbox import self_heal_code
from app.services.pipeline.tts.audio import duration_seconds
from app.services.pipeline.tts.elevenlabs import synthesize
from app.services.pipeline.vision import capture_page, describe_screenshots, generate_web_script
from app.services.video_store import video_store

_RENDER_FPS = 10
_MAX_RENDER_SECONDS = 300.0


# ── Generation ───────────────────────────────────────────────────────────────
def run_generation(job_id: str) -> None:
    """Run the generation half of the pipeline up to the HITL approval gate."""
    logger.info("pipeline_generation_started", job_id=job_id)
    job = video_store.get_job(job_id)
    if job is None:
        logger.error("pipeline_generation_missing_job", job_id=job_id)
        return

    try:
        llm = PipelineLLM(job_id=job_id)
        if job.mode == "web_explainer":
            _generate_web_explainer(job_id, job, llm)
        else:
            _generate_code_tutorial(job_id, job, llm)
        logger.info("pipeline_generation_paused_for_review", job_id=job_id, mode=job.mode)
    except Exception as exc:  # noqa: BLE001 - background task must not crash silently
        logger.exception("pipeline_generation_failed", job_id=job_id, error=str(exc))
        video_store.update_job(job_id, status="error", current_step="error", error_message=str(exc))


def _generate_code_tutorial(job_id: str, job: VideoJob, llm: PipelineLLM) -> None:
    """Research -> code -> self-healing sandbox -> script -> timeline."""
    video_store.update_job(job_id, status="running", current_step="research")
    research_notes = research_topic(llm, job.topic, job.language)
    video_store.update_job(job_id, current_step="code", artifacts_merge={"research": research_notes})

    code = generate_code(llm, job.topic, research_notes)
    video_store.update_job(job_id, current_step="sandbox", artifacts_merge={"code": code})

    # Self-healing gate: downstream stages use the VALIDATED code.
    heal = self_heal_code(code, llm, job_id=job_id)
    code = heal.code
    video_store.update_job(
        job_id,
        current_step="script",
        artifacts_merge={
            "code": code,
            "code_validated": heal.validated,
            "code_output": heal.result.stdout[:2000] if heal.result else "",
            "sandbox_log": heal.log,
        },
    )

    script = generate_script(llm, job.topic, research_notes, code, job.language)
    video_store.update_job(job_id, current_step="timeline", artifacts_merge={"script": script})

    try:
        timeline = convert_script_to_timeline(script, LLMClient()).model_dump()
    except Exception as exc:  # noqa: BLE001
        logger.warning("timeline_generation_failed", job_id=job_id, error=str(exc))
        timeline = {"events": [], "error": str(exc)}

    video_store.update_job(
        job_id,
        status="awaiting_approval",
        current_step="awaiting_approval",
        awaiting_approval=True,
        review_status="pending",
        artifacts_merge={"timeline": timeline},
    )


def _generate_web_explainer(job_id: str, job: VideoJob, llm: PipelineLLM) -> None:
    """Navigate -> screenshot -> Kimi-vision description -> Egyptian-Arabic script."""
    video_store.update_job(job_id, status="running", current_step="research")
    shots_dir = Path(settings.VIDEO_DATA_DIR) / "screenshots" / job_id
    screenshots = capture_page(job.url, shots_dir)  # type: ignore[arg-type]
    description = describe_screenshots(screenshots, job.url or "", job_id=job_id)
    video_store.update_job(
        job_id,
        current_step="script",
        artifacts_merge={"screenshots": screenshots, "research": description, "code": None},
    )

    script = generate_web_script(llm, job.url or "", description, job.language)
    video_store.update_job(
        job_id,
        status="awaiting_approval",
        current_step="awaiting_approval",
        awaiting_approval=True,
        review_status="pending",
        artifacts_merge={"script": script, "timeline": {"events": []}},
    )


# ── Render ───────────────────────────────────────────────────────────────────
def run_render(job_id: str) -> None:
    """Render the approved job to a real MP4 (mode-specific)."""
    logger.info("pipeline_render_started", job_id=job_id)
    job = video_store.get_job(job_id)
    if job is None:
        logger.error("pipeline_render_missing_job", job_id=job_id)
        return

    if job.review_status != "approved":
        logger.error("pipeline_render_not_approved", job_id=job_id, review_status=job.review_status)
        video_store.update_job(job_id, status="error", current_step="error", error_message="render requires approval")
        return

    try:
        if job.mode == "web_explainer":
            _render_web_explainer(job_id, job)
        else:
            _render_code_tutorial(job_id, job)
        logger.info("pipeline_render_done", job_id=job_id, mode=job.mode)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline_render_failed", job_id=job_id, error=str(exc))
        video_store.update_job(job_id, status="error", current_step="error", error_message=str(exc))


def _synthesize_and_measure(job_id: str, script: str) -> tuple[str, float]:
    """Synthesize narration and return (audio_path, clamped_duration)."""
    video_store.update_job(job_id, status="rendering", current_step="tts", awaiting_approval=False)
    audio_path = synthesize(script)
    duration = max(3.0, min(duration_seconds(audio_path), _MAX_RENDER_SECONDS))
    video_store.update_job(
        job_id,
        current_step="render",
        artifacts_merge={"audio_path": str(audio_path), "audio_duration_s": round(duration, 2)},
    )
    return str(audio_path), duration


def _render_code_tutorial(job_id: str, job: VideoJob) -> None:
    """Synthesize narration, type out the code across frames, mux to MP4."""
    artifacts = job.artifacts or {}
    script = (artifacts.get("script") or job.topic).strip()
    code = (artifacts.get("code") or "# no code").strip()

    audio_path, duration = _synthesize_and_measure(job_id, script)
    frames_dir = Path(tempfile.gettempdir()) / "render" / job_id / "frames"
    try:
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        render_frames(code, job.topic, frames_dir, fps=_RENDER_FPS, duration_s=duration)
        video_path = Path(settings.VIDEO_OUTPUT_DIR) / job_id / "final.mp4"
        assemble_video(frames_dir, _RENDER_FPS, audio_path, video_path)
        video_store.update_job(
            job_id, status="done", current_step="done", artifacts_merge={"video_path": str(video_path)}
        )
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def _render_web_explainer(job_id: str, job: VideoJob) -> None:
    """Synthesize narration, Ken-Burns over the page screenshots, mux to MP4."""
    artifacts = job.artifacts or {}
    script = (artifacts.get("script") or job.topic).strip()
    screenshots = artifacts.get("screenshots") or []
    if not screenshots:
        raise RuntimeError("no screenshots available to render")

    audio_path, duration = _synthesize_and_measure(job_id, script)
    video_path = Path(settings.VIDEO_OUTPUT_DIR) / job_id / "final.mp4"
    render_screenshot_video(screenshots, audio_path, str(video_path), duration)
    video_store.update_job(job_id, status="done", current_step="done", artifacts_merge={"video_path": str(video_path)})
