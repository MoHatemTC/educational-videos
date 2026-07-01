"""Webhook notifications for video job terminal states."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.services.video_store import video_store

_TERMINAL_STATUSES = {"done", "error", "rejected"}


def _payload(job_id: str, event: str) -> dict[str, Any] | None:
    """Build a webhook payload for a job, or ``None`` if the job vanished."""
    job = video_store.get_job(job_id)
    if job is None:
        return None

    payload: dict[str, Any] = {
        "event": event,
        "job_id": job.id,
        "status": job.status,
        "current_step": job.current_step,
        "review_status": job.review_status,
        "mode": job.mode,
        "topic": job.topic,
        "language": job.language,
        "url": job.url,
        "error": job.error_message,
    }
    if settings.VIDEO_WEBHOOK_INCLUDE_ARTIFACTS:
        payload["artifacts"] = {k: v for k, v in (job.artifacts or {}).items() if k != "traces"}
    return payload


def notify_job_status(job_id: str, event: str | None = None) -> None:
    """POST a job status webhook when configured.

    The notifier is best-effort and never raises into generation/render paths.
    """
    webhook_url = settings.VIDEO_WEBHOOK_URL.strip()
    if not webhook_url:
        return

    payload = _payload(job_id, event or "video.job.updated")
    if payload is None:
        return
    if payload["status"] not in _TERMINAL_STATUSES:
        return

    try:
        response = httpx.post(webhook_url, json=payload, timeout=settings.VIDEO_WEBHOOK_TIMEOUT_SECONDS)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - notification failure must not fail the job
        logger.warning("video_webhook_notification_failed", job_id=job_id, error=str(exc))
        return

    logger.info("video_webhook_notification_sent", job_id=job_id, webhook_event=payload["event"])
