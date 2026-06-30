"""HTTP client for the educational-video backend.

Thin wrapper over the FastAPI endpoints. Holds no pipeline logic — the Streamlit
app talks to the backend exclusively through these functions.
"""

import json
import os
from collections.abc import Iterator

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
_TIMEOUT = httpx.Timeout(30.0)


def _url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def create_job(topic: str, language: str, mode: str = "code_tutorial", url: str | None = None) -> dict:
    """POST /videos/jobs — start a generation job (code tutorial or web explainer)."""
    body: dict = {"topic": topic, "language": language, "mode": mode}
    if url:
        body["url"] = url
    resp = httpx.post(_url("/videos/jobs"), json=body, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def list_jobs(limit: int = 20) -> list[dict]:
    """GET /videos/jobs — recent jobs."""
    resp = httpx.get(_url("/videos/jobs"), params={"limit": limit}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("jobs", [])


def get_status(job_id: str) -> dict:
    """GET /videos/jobs/{id}."""
    resp = httpx.get(_url(f"/videos/jobs/{job_id}"), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_review(job_id: str) -> dict:
    """GET /videos/jobs/{id}/review."""
    resp = httpx.get(_url(f"/videos/jobs/{job_id}/review"), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def approve(job_id: str, script: str | None = None, code: str | None = None) -> dict:
    """POST /videos/jobs/{id}/approve (optionally with edits)."""
    body: dict = {}
    if script is not None:
        body["script"] = script
    if code is not None:
        body["code"] = code
    resp = httpx.post(_url(f"/videos/jobs/{job_id}/approve"), json=body, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def reject(job_id: str, reason: str | None = None) -> dict:
    """POST /videos/jobs/{id}/reject."""
    resp = httpx.post(_url(f"/videos/jobs/{job_id}/reject"), json={"reason": reason}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_traces(job_id: str) -> dict:
    """GET /videos/jobs/{id}/traces."""
    resp = httpx.get(_url(f"/videos/jobs/{job_id}/traces"), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def result_url(job_id: str) -> str:
    """Direct URL of the finished MP4 (for download links / st.video)."""
    return _url(f"/videos/jobs/{job_id}/result")


def get_result_bytes(job_id: str) -> bytes | None:
    """Fetch the finished MP4 bytes, or None if not ready (409)."""
    resp = httpx.get(_url(f"/videos/jobs/{job_id}/result"), timeout=httpx.Timeout(120.0))
    if resp.status_code == 409:
        return None
    resp.raise_for_status()
    return resp.content


def stream_events(job_id: str) -> Iterator[dict]:
    """Consume the SSE progress stream, yielding each event payload as a dict.

    Blocks until the job reaches a terminal state (or the stream ends). Used by
    the Generate page to render live progress.
    """
    with httpx.stream("GET", _url(f"/videos/jobs/{job_id}/events"), timeout=httpx.Timeout(None)) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line.startswith("data:"):
                raw = line[len("data:") :].strip()
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue
