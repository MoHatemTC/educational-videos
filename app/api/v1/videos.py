"""Educational-video generation API.

Job lifecycle: ``POST /jobs`` persists a tracking row and enqueues generation
onto the configured video task queue. The client polls ``GET /jobs/{id}``; a
reviewer fetches ``/review``, then ``/approve`` (resumes the graph and enqueues
render) or ``/reject``; the finished MP4 is served from ``/result`` and
per-stage cost/latency from ``/traces``.

NOTE: slowapi's ``@limiter.limit`` is incompatible with FastAPI 0.121's router
internals (``_IncludedRouter`` has no ``.path``), so these routes are not
decorated with it. Re-introduce rate limiting via a compatible mechanism during
productization.
"""

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.logging import logger
from app.models.video_job import VideoJob
from app.schemas.video import (
    ApprovalRequest,
    JobCreateResponse,
    JobRequest,
    JobStatusResponse,
    RejectionRequest,
    ReviewArtifact,
    TraceRow,
)
from app.services.pipeline.orchestrator import approve_generation, reject_generation
from app.services.pipeline.task_queue import enqueue_generation, enqueue_render
from app.services.video_store import video_store

router = APIRouter()


def _job_or_404(job_id: str) -> VideoJob:
    """Fetch a job or raise 404."""
    job = video_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


def _status(job: VideoJob) -> JobStatusResponse:
    """Build the status response from a job row (artifacts minus the traces blob)."""
    return JobStatusResponse(
        job_id=job.id,
        topic=job.topic,
        language=job.language,
        mode=job.mode,
        url=job.url,
        status=job.status,
        current_step=job.current_step,
        awaiting_approval=job.awaiting_approval,
        review_status=job.review_status,
        error=job.error_message,
        artifacts={k: v for k, v in (job.artifacts or {}).items() if k != "traces"},
    )


@router.post("/jobs", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(body: JobRequest, background_tasks: BackgroundTasks) -> JobCreateResponse:
    """Create a generation job and enqueue it on the configured task queue."""
    job_id = str(uuid.uuid4())
    video_store.create_job(job_id, topic=body.topic, language=body.language, mode=body.mode, url=body.url)
    try:
        queued = enqueue_generation(job_id, background_tasks=background_tasks)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "video_job_enqueued",
        job_id=job_id,
        topic=body.topic,
        language=body.language,
        mode=body.mode,
        queue_backend=queued.backend,
        queue_task_id=queued.task_id,
    )
    return JobCreateResponse(job_id=job_id, status="pending")


@router.get("/jobs")
async def list_jobs(limit: int = 20) -> JSONResponse:
    """List the most recent jobs (newest first) for the dashboard."""
    jobs = video_store.list_jobs(limit=limit)
    rows = [
        {
            "job_id": j.id,
            "topic": j.topic,
            "language": j.language,
            "mode": j.mode,
            "url": j.url,
            "status": j.status,
            "current_step": j.current_step,
            "awaiting_approval": j.awaiting_approval,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]
    return JSONResponse({"jobs": rows})


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the current lifecycle state and artifact summary of a job."""
    return _status(_job_or_404(job_id))


_TERMINAL_STATES = {"awaiting_approval", "done", "error", "rejected", "render_pending"}


@router.get("/jobs/{job_id}/events")
async def stream_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of a job's live progress.

    Emits a ``data:`` event whenever the status/step changes and a final
    ``event: done`` when the job reaches a terminal state, so clients get live
    progress without polling. Falls back gracefully if the job disappears.
    """
    _job_or_404(job_id)

    async def event_generator():
        last_payload: str | None = None
        # Cap the stream (~10 min) so a stuck job can't hold the connection open.
        for _ in range(600):
            job = video_store.get_job(job_id)
            if job is None:
                yield 'event: done\ndata: {"status": "missing"}\n\n'
                return
            payload = json.dumps(
                {
                    "job_id": job.id,
                    "status": job.status,
                    "current_step": job.current_step,
                    "awaiting_approval": job.awaiting_approval,
                    "error": job.error_message,
                    "has": {
                        "research": bool((job.artifacts or {}).get("research")),
                        "code": bool((job.artifacts or {}).get("code")),
                        "script": bool((job.artifacts or {}).get("script")),
                        "timeline": bool((job.artifacts or {}).get("timeline")),
                    },
                }
            )
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if job.status in _TERMINAL_STATES:
                yield f"event: done\ndata: {payload}\n\n"
                return
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}/review", response_model=ReviewArtifact)
async def get_review(job_id: str) -> ReviewArtifact:
    """Return the editable artifacts (script/code/timeline) for human review."""
    job = _job_or_404(job_id)
    art = job.artifacts or {}
    return ReviewArtifact(
        job_id=job.id,
        topic=job.topic,
        language=job.language,
        mode=job.mode,
        url=job.url,
        status=job.status,
        script=art.get("script"),
        code=art.get("code"),
        timeline=art.get("timeline"),
        research=art.get("research"),
        screenshots=art.get("screenshots"),
    )


@router.post("/jobs/{job_id}/approve", response_model=JobStatusResponse)
async def approve_job(job_id: str, body: ApprovalRequest, background_tasks: BackgroundTasks) -> JobStatusResponse:
    """Approve a job, resume its paused graph checkpoint, and dispatch rendering."""
    job = _job_or_404(job_id)
    if not job.awaiting_approval:
        raise HTTPException(status_code=409, detail=f"job {job_id} is not awaiting approval (status={job.status})")

    merge: dict = {}
    if body.script is not None:
        merge["script"] = body.script
    if body.code is not None:
        merge["code"] = body.code

    updated = approve_generation(job_id, reviewer_edits=merge or None)
    try:
        queued = enqueue_render(job_id, background_tasks=background_tasks)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "video_job_approved",
        job_id=job_id,
        edited=bool(merge),
        queue_backend=queued.backend,
        queue_task_id=queued.task_id,
    )
    return _status(video_store.get_job(job_id) or updated or job)


@router.post("/jobs/{job_id}/reject", response_model=JobStatusResponse)
async def reject_job(job_id: str, body: RejectionRequest) -> JobStatusResponse:
    """Reject a job at the approval gate and resume the graph with that decision."""
    job = _job_or_404(job_id)
    updated = reject_generation(job_id, reason=body.reason)
    logger.info("video_job_rejected", job_id=job_id, reason=body.reason)
    return _status(updated or video_store.get_job(job_id) or job)


@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    """Stream the finished MP4 (or 409 if rendering is not complete)."""
    job = _job_or_404(job_id)
    video_path = (job.artifacts or {}).get("video_path")
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(status_code=409, detail=f"job {job_id} has no rendered video yet (status={job.status})")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@router.get("/jobs/{job_id}/traces")
async def get_traces(job_id: str) -> JSONResponse:
    """Return per-stage observability rows (tokens, est cost, latency)."""
    job = _job_or_404(job_id)
    raw_traces = (job.artifacts or {}).get("traces", [])
    rows = [TraceRow(**row).model_dump() for row in raw_traces]
    totals = {
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "est_cost_usd": round(sum(r["est_cost_usd"] for r in rows), 6),
        "latency_ms": sum(r["latency_ms"] for r in rows),
    }
    return JSONResponse({"job_id": job_id, "rows": rows, "totals": totals})
