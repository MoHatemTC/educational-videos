"""Pipeline orchestrator.

Two job modes share one lifecycle (generate -> HITL approve -> render -> mp4):

* ``code_tutorial`` — LangGraph nodes for research -> code -> sandbox/self-heal
  -> script -> visual planning, then a HITL graph interrupt before rendering.
* ``web_explainer`` — LangGraph nodes for navigation -> screenshot ->
  Kimi-vision description -> script -> visual planning, then the same HITL
  graph interrupt before rendering.

The generation half is a LangGraph ``StateGraph`` in
``app.services.pipeline.graph``. Rendering remains here because it runs only
after approval and reuses the existing TTS/frames/FFmpeg implementation.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.core.observability import langfuse_trace
from app.models.video_job import VideoJob
from app.services.pipeline.graph import invoke_generation_graph, resume_generation_graph
from app.services.pipeline.narration_guard import clean_narration_text
from app.services.pipeline.render.ffmpeg_render import assemble_video
from app.services.pipeline.render.frames import render_frames
from app.services.pipeline.render.screenshot_video import render_screenshot_video
from app.services.pipeline.tts.audio import duration_seconds
from app.services.pipeline.tts.elevenlabs import synthesize, voice_id_for_language
from app.services.video_store import video_store

_RENDER_FPS = 10
_MAX_RENDER_SECONDS = 300.0


# ── Generation ───────────────────────────────────────────────────────────────
def run_generation(job_id: str) -> None:
    """Run the LangGraph generation pipeline up to the HITL approval interrupt."""
    logger.info("pipeline_generation_started", job_id=job_id)
    job = video_store.get_job(job_id)
    if job is None:
        logger.error("pipeline_generation_missing_job", job_id=job_id)
        return

    with langfuse_trace(
        name="video.generation",
        as_type="agent",
        input_data={
            "job_id": job_id,
            "mode": job.mode,
            "topic": job.topic,
            "language": job.language,
            "url": job.url,
        },
        metadata={
            "job_id": job_id,
            "mode": job.mode,
            "environment": settings.ENVIRONMENT.value,
        },
        session_id=job_id,
        tags=["video-pipeline", "generation", "langgraph", job.mode, settings.ENVIRONMENT.value],
    ) as trace:
        try:
            invoke_generation_graph(job_id)
            if trace is not None:
                trace.update(output={"status": "awaiting_approval"})
            logger.info("pipeline_generation_paused_for_review", job_id=job_id, mode=job.mode)
        except Exception as exc:  # noqa: BLE001 - background task must not crash silently
            if trace is not None:
                trace.update(output={"status": "error", "error": str(exc)})
            logger.exception("pipeline_generation_failed", job_id=job_id, error=str(exc))
            video_store.update_job(job_id, status="error", current_step="error", error_message=str(exc))


def approve_generation(job_id: str, reviewer_edits: dict[str, Any] | None = None) -> VideoJob | None:
    """Resume the paused generation graph with an approval decision."""
    return resume_generation_graph(job_id, reviewer_edits=reviewer_edits, approved=True)


def reject_generation(job_id: str, reason: str | None = None) -> VideoJob | None:
    """Resume the paused generation graph with a rejection decision."""
    return resume_generation_graph(job_id, approved=False, rejection_reason=reason)


# ── Render ───────────────────────────────────────────────────────────────────
def run_render(job_id: str) -> None:
    """Render the approved job to a real MP4 (mode-specific)."""
    logger.info("pipeline_render_started", job_id=job_id)
    job = video_store.get_job(job_id)
    if job is None:
        logger.error("pipeline_render_missing_job", job_id=job_id)
        return

    with langfuse_trace(
        name="video.render",
        as_type="chain",
        input_data={"job_id": job_id, "mode": job.mode, "review_status": job.review_status},
        metadata={"job_id": job_id, "mode": job.mode, "environment": settings.ENVIRONMENT.value},
        session_id=job_id,
        tags=["video-pipeline", "render", job.mode, settings.ENVIRONMENT.value],
    ) as trace:
        if job.review_status != "approved":
            logger.error("pipeline_render_not_approved", job_id=job_id, review_status=job.review_status)
            video_store.update_job(
                job_id, status="error", current_step="error", error_message="render requires approval"
            )
            if trace is not None:
                trace.update(output={"status": "error", "error": "render requires approval"})
            return

        try:
            if job.mode == "web_explainer":
                _render_web_explainer(job_id, job)
            else:
                _render_code_tutorial(job_id, job)
            if trace is not None:
                trace.update(output={"status": "done"})
            logger.info("pipeline_render_done", job_id=job_id, mode=job.mode)
        except Exception as exc:  # noqa: BLE001
            if trace is not None:
                trace.update(output={"status": "error", "error": str(exc)})
            logger.exception("pipeline_render_failed", job_id=job_id, error=str(exc))
            video_store.update_job(job_id, status="error", current_step="error", error_message=str(exc))


def _synthesize_and_measure(job_id: str, script: str, language: str) -> tuple[str, float]:
    """Synthesize narration and return (audio_path, clamped_duration)."""
    video_store.update_job(job_id, status="rendering", current_step="tts", awaiting_approval=False)

    cleaned_script = clean_narration_text(script, language)
    if cleaned_script and cleaned_script != script:
        logger.warning("narration_sanitized_before_tts", job_id=job_id)
        script = cleaned_script
        video_store.update_job(
            job_id,
            artifacts_merge={"script": script, "script_sanitized_before_tts": True},
        )

    with langfuse_trace(
        name="video.tts",
        as_type="tool",
        input_data={"job_id": job_id, "characters": len(script)},
        metadata={
            "job_id": job_id,
            "provider": "elevenlabs",
            "language": language,
            "voice_id": voice_id_for_language(language),
            "environment": settings.ENVIRONMENT.value,
        },
        session_id=job_id,
        tags=["video-pipeline", "tts", settings.ENVIRONMENT.value],
    ) as trace:
        audio_path = synthesize(script, voice_id=voice_id_for_language(language))
        duration = max(3.0, min(duration_seconds(audio_path), _MAX_RENDER_SECONDS))
        if trace is not None:
            trace.update(output={"audio_path": str(audio_path), "duration_s": round(duration, 2)})
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

    audio_path, duration = _synthesize_and_measure(job_id, script, job.language)
    frames_dir = Path(tempfile.gettempdir()) / "render" / job_id / "frames"
    try:
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        timeline = artifacts.get("timeline") if not artifacts.get("timeline_error") else None
        render_frames(code, job.topic, frames_dir, fps=_RENDER_FPS, duration_s=duration, timeline=timeline)
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

    audio_path, duration = _synthesize_and_measure(job_id, script, job.language)
    video_path = Path(settings.VIDEO_OUTPUT_DIR) / job_id / "final.mp4"
    render_screenshot_video(screenshots, audio_path, str(video_path), duration)
    video_store.update_job(job_id, status="done", current_step="done", artifacts_merge={"video_path": str(video_path)})
