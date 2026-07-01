"""Tests for decoupled video task queue dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.services.pipeline import task_queue


class _FakeStore:
    """Capture video_store updates made by queue helpers."""

    def __init__(self) -> None:
        """Initialize an empty update log."""
        self.updates: list[dict[str, Any]] = []

    def update_job(self, job_id: str, *, artifacts_merge: dict[str, Any] | None = None, **fields: Any) -> None:
        """Record a job update call."""
        self.updates.append({"job_id": job_id, "artifacts_merge": artifacts_merge, **fields})


class _FakeAsyncResult:
    """Minimal stand-in for Celery AsyncResult."""

    id = "celery-task-123"


class _FakeCeleryApp:
    """Capture Celery send_task calls."""

    def __init__(self) -> None:
        """Initialize an empty task call log."""
        self.calls: list[tuple[str, list[str]]] = []

    def send_task(self, name: str, args: list[str]) -> _FakeAsyncResult:
        """Record a Celery enqueue request."""
        self.calls.append((name, args))
        return _FakeAsyncResult()


class _FakeBackgroundTasks:
    """Capture FastAPI BackgroundTasks fallback scheduling."""

    def __init__(self) -> None:
        """Initialize an empty task list."""
        self.tasks: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []

    def add_task(self, func: Callable[..., Any], *args: Any, **_kwargs: Any) -> None:
        """Record a fallback task."""
        self.tasks.append((func, args))


def test_enqueue_generation_uses_celery_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Celery mode should submit generation to the broker, not FastAPI background tasks."""
    fake_store = _FakeStore()
    fake_celery = _FakeCeleryApp()
    monkeypatch.setattr(task_queue.settings, "VIDEO_TASK_QUEUE_BACKEND", "celery")
    monkeypatch.setattr(task_queue, "video_store", fake_store)
    monkeypatch.setattr(task_queue, "celery_app", fake_celery)

    result = task_queue.enqueue_generation("job-1")

    assert result.backend == "celery"
    assert result.task_id == "celery-task-123"
    assert fake_celery.calls == [("video.generate", ["job-1"])]
    assert fake_store.updates[-1]["current_step"] == "generation_queued"
    assert fake_store.updates[-1]["artifacts_merge"]["queue"] == {
        "stage": "generation",
        "backend": "celery",
        "task_id": "celery-task-123",
        "status": "queued",
    }


def test_enqueue_render_uses_background_tasks_only_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old in-process scheduling path should be gated behind the fallback flag."""
    fake_store = _FakeStore()
    fake_background = _FakeBackgroundTasks()

    def _fake_render(job_id: str) -> None:
        """Tiny render stand-in."""
        assert job_id == "job-2"

    monkeypatch.setattr(task_queue.settings, "VIDEO_TASK_QUEUE_BACKEND", "background_tasks")
    monkeypatch.setattr(task_queue, "video_store", fake_store)
    monkeypatch.setattr(task_queue, "run_render", _fake_render)

    result = task_queue.enqueue_render("job-2", background_tasks=fake_background)

    assert result.backend == "background_tasks"
    assert fake_background.tasks == [(_fake_render, ("job-2",))]
    assert fake_store.updates[-1]["status"] == "render_pending"
    assert fake_store.updates[-1]["current_step"] == "render_queued"
    assert fake_store.updates[-1]["artifacts_merge"]["queue"]["backend"] == "background_tasks"


def test_worker_task_records_failure_for_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected worker failures should be visible in video_store for polling clients."""
    fake_store = _FakeStore()
    monkeypatch.setattr(task_queue, "video_store", fake_store)

    class _Request:
        """Minimal Celery request stand-in."""

        id = "worker-task-1"
        retries = 1

    def _boom(_job_id: str) -> None:
        """Fail like a crashing worker task."""
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        task_queue._execute_worker_task("job-3", "generation", _boom, request=_Request())

    assert fake_store.updates[0]["artifacts_merge"]["worker"]["status"] == "started"
    assert fake_store.updates[-1]["status"] == "error"
    assert fake_store.updates[-1]["current_step"] == "worker_error"
    assert fake_store.updates[-1]["artifacts_merge"]["worker"] == {
        "stage": "generation",
        "backend": "celery",
        "task_id": "worker-task-1",
        "attempt": 2,
        "status": "failed",
        "error": "boom",
    }
