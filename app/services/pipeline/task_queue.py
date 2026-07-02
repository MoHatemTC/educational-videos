"""Task queue integration for video generation and rendering.

The API process should only create or approve jobs and then enqueue work. In
MVP Docker, Celery workers consume the queued generation/render tasks from
Valkey/Redis. For local development, ``VIDEO_TASK_QUEUE_BACKEND=background_tasks``
keeps the old FastAPI ``BackgroundTasks`` behavior behind an explicit flag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from app.core.config import settings
from app.core.logging import logger
from app.services.pipeline.orchestrator import run_generation, run_render
from app.services.video_store import video_store

_GENERATE_TASK_NAME = "video.generate"
_RENDER_TASK_NAME = "video.render"
_SUPPORTED_BACKENDS = {"celery", "background_tasks", "inline"}


class BackgroundTaskSink(Protocol):
    """Subset of FastAPI ``BackgroundTasks`` used by the local fallback."""

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Schedule a callable after the response is sent."""


@dataclass(frozen=True)
class EnqueuedTask:
    """Metadata returned after a pipeline task has been queued."""

    job_id: str
    stage: str
    backend: str
    task_id: str | None = None


class _MissingCeleryApp:
    """Small import-time stand-in used when Celery is not installed locally."""

    def task(self, *_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that leaves task functions unchanged."""

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _decorator

    def send_task(self, *_args: Any, **_kwargs: Any) -> Any:
        """Raise a clear error when Celery mode is selected without Celery."""
        raise RuntimeError("Celery is not installed; use uv sync or set VIDEO_TASK_QUEUE_BACKEND=background_tasks")


def _create_celery_app() -> Any:
    """Create the Celery application, or a local stub if Celery is unavailable."""
    try:
        celery_module = cast(Any, import_module("celery"))
    except ModuleNotFoundError:
        return _MissingCeleryApp()

    celery_cls = celery_module.Celery
    app = celery_cls(
        "educational_video_pipeline",
        broker=settings.VIDEO_TASK_QUEUE_BROKER_URL,
        backend=settings.VIDEO_TASK_QUEUE_RESULT_BACKEND,
    )
    app.conf.update(
        broker_connection_retry_on_startup=True,
        broker_transport_options={"visibility_timeout": settings.VIDEO_TASK_QUEUE_VISIBILITY_TIMEOUT},
        result_backend_transport_options={"visibility_timeout": settings.VIDEO_TASK_QUEUE_VISIBILITY_TIMEOUT},
        result_expires=86_400,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = _create_celery_app()


def _queue_backend() -> str:
    """Return the configured queue backend, normalized and validated."""
    backend = str(settings.VIDEO_TASK_QUEUE_BACKEND).lower().strip()
    if backend not in _SUPPORTED_BACKENDS:
        raise RuntimeError(
            f"unsupported VIDEO_TASK_QUEUE_BACKEND={backend!r}; expected one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    return backend


def _queue_artifact(stage: str, backend: str, task_id: str | None) -> dict[str, Any]:
    """Build a serializable queue metadata artifact."""
    return {"queue": {"stage": stage, "backend": backend, "task_id": task_id, "status": "queued"}}


def _worker_artifact(stage: str, status: str, request: Any | None = None, error: str | None = None) -> dict[str, Any]:
    """Build a serializable worker metadata artifact."""
    task_id = str(getattr(request, "id", "") or "") or None
    retries = int(getattr(request, "retries", 0) or 0)
    worker: dict[str, Any] = {
        "stage": stage,
        "backend": "celery",
        "task_id": task_id,
        "attempt": retries + 1,
        "status": status,
    }
    if error:
        worker["error"] = error
    return {"worker": worker}


def _update_queued(job_id: str, stage: str, backend: str, task_id: str | None) -> None:
    """Persist that a job has been accepted by a queue backend."""
    if stage == "generation":
        video_store.update_job(
            job_id,
            current_step="generation_queued",
            artifacts_merge=_queue_artifact(stage, backend, task_id),
        )
    else:
        video_store.update_job(
            job_id,
            status="render_pending",
            current_step="render_queued",
            awaiting_approval=False,
            artifacts_merge=_queue_artifact(stage, backend, task_id),
        )


def _update_queue_error(job_id: str, stage: str, exc: Exception) -> None:
    """Persist a queue submission failure so the UI can show it."""
    video_store.update_job(
        job_id,
        status="error",
        current_step="queue_error",
        error_message=f"failed to enqueue {stage}: {exc}",
        artifacts_merge={
            "queue": {"stage": stage, "backend": "celery", "status": "enqueue_failed", "error": str(exc)}
        },
    )


def _enqueue_celery(job_id: str, stage: str, task_name: str) -> EnqueuedTask:
    """Enqueue a Celery task and persist queue metadata."""
    try:
        async_result = celery_app.send_task(task_name, args=[job_id])
    except Exception as exc:  # noqa: BLE001 - queue failures must be visible on the job
        _update_queue_error(job_id, stage, exc)
        logger.exception("video_task_enqueue_failed", job_id=job_id, stage=stage, error=str(exc))
        raise RuntimeError(f"failed to enqueue {stage} task") from exc

    task_id = str(getattr(async_result, "id", "") or "") or None
    _update_queued(job_id, stage, "celery", task_id)
    logger.info("video_task_enqueued", job_id=job_id, stage=stage, backend="celery", task_id=task_id)
    return EnqueuedTask(job_id=job_id, stage=stage, backend="celery", task_id=task_id)


def _enqueue_fallback(
    job_id: str,
    stage: str,
    func: Callable[[str], None],
    background_tasks: BackgroundTaskSink | None,
) -> EnqueuedTask:
    """Use FastAPI BackgroundTasks only when explicitly configured for local dev."""
    if background_tasks is None:
        raise RuntimeError("BackgroundTasks fallback requires a BackgroundTasks object")
    _update_queued(job_id, stage, "background_tasks", None)
    background_tasks.add_task(func, job_id)
    logger.info("video_task_enqueued", job_id=job_id, stage=stage, backend="background_tasks")
    return EnqueuedTask(job_id=job_id, stage=stage, backend="background_tasks")


def _run_inline(job_id: str, stage: str, func: Callable[[str], None]) -> EnqueuedTask:
    """Run a task synchronously; intended only for narrow test/debug usage."""
    _update_queued(job_id, stage, "inline", None)
    func(job_id)
    return EnqueuedTask(job_id=job_id, stage=stage, backend="inline")


def enqueue_generation(job_id: str, background_tasks: BackgroundTaskSink | None = None) -> EnqueuedTask:
    """Queue video generation and return immediately with task metadata."""
    backend = _queue_backend()
    if backend == "celery":
        return _enqueue_celery(job_id, "generation", _GENERATE_TASK_NAME)
    if backend == "inline":
        return _run_inline(job_id, "generation", run_generation)
    return _enqueue_fallback(job_id, "generation", run_generation, background_tasks)


def enqueue_render(job_id: str, background_tasks: BackgroundTaskSink | None = None) -> EnqueuedTask:
    """Queue approved video rendering and return immediately with task metadata."""
    backend = _queue_backend()
    if backend == "celery":
        return _enqueue_celery(job_id, "render", _RENDER_TASK_NAME)
    if backend == "inline":
        return _run_inline(job_id, "render", run_render)
    return _enqueue_fallback(job_id, "render", run_render, background_tasks)


def _execute_worker_task(job_id: str, stage: str, func: Callable[[str], None], request: Any | None = None) -> None:
    """Run a worker task with persisted started/failed/finished visibility."""
    video_store.update_job(job_id, artifacts_merge=_worker_artifact(stage, "started", request))
    try:
        func(job_id)
    except Exception as exc:  # noqa: BLE001 - Celery should retry unexpected worker crashes
        video_store.update_job(
            job_id,
            status="error",
            current_step="worker_error",
            error_message=f"{stage} worker failed: {exc}",
            artifacts_merge=_worker_artifact(stage, "failed", request, error=str(exc)),
        )
        logger.exception("video_worker_task_failed", job_id=job_id, stage=stage, error=str(exc))
        raise
    video_store.update_job(job_id, artifacts_merge=_worker_artifact(stage, "finished", request))


@celery_app.task(
    name=_GENERATE_TASK_NAME,
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": settings.VIDEO_TASK_QUEUE_MAX_RETRIES},
)
def run_generation_task(self: Any, job_id: str) -> None:
    """Celery task entrypoint for graph-based video generation."""
    _execute_worker_task(job_id, "generation", run_generation, request=cast(Any, self).request)


@celery_app.task(
    name=_RENDER_TASK_NAME,
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": settings.VIDEO_TASK_QUEUE_MAX_RETRIES},
)
def run_render_task(self: Any, job_id: str) -> None:
    """Celery task entrypoint for approved video rendering."""
    _execute_worker_task(job_id, "render", run_render, request=cast(Any, self).request)
