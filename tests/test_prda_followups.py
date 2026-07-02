"""Tests for Issue #26 PRD-alignment follow-ups."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.core import rate_limit
from app.schemas.video import JobRequest
from app.services.pipeline import notifications
from app.services.pipeline.graph import _route_mode, _route_web_capture
from app.services.pipeline.sandbox import runner
from app.services.pipeline.tts.elevenlabs import _provider_voice_settings


def test_job_request_accepts_raw_script_without_topic() -> None:
    """A finished narration script can be submitted instead of a topic."""
    body = JobRequest(raw_script="This is a finished tutorial narration that is already approved.", language="en")

    assert body.resolved_topic().startswith("This is a finished tutorial")
    assert body.resolved_raw_script() == "This is a finished tutorial narration that is already approved."


def test_route_mode_skips_research_for_raw_script_jobs() -> None:
    """Raw-script code jobs should go directly to visual planning."""
    assert _route_mode({"mode": "code_tutorial", "raw_script": "Already written narration."}) == "visual_planning"


def test_route_web_capture_skips_description_when_raw_script_is_supplied() -> None:
    """Raw-script web jobs still capture screenshots but skip description/script nodes."""
    assert _route_web_capture({"raw_script": "Already written narration."}) == "web_visual_planning"
    assert _route_web_capture({}) == "web_describe"


@pytest.mark.asyncio
async def test_video_rate_limit_blocks_after_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compatible video rate limiter should block requests over the videos limit."""
    rate_limit.reset_rate_limit_state()
    monkeypatch.setattr(rate_limit.settings, "RATE_LIMIT_ENDPOINTS", {"videos": ["2 per minute"]})
    request: Any = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))

    await rate_limit.enforce_video_rate_limit(request)
    await rate_limit.enforce_video_rate_limit(request)
    with pytest.raises(Exception) as exc_info:
        await rate_limit.enforce_video_rate_limit(request)

    assert exc_info.value.status_code == 429


def test_tts_provider_settings_do_not_send_provider_neutral_emotion() -> None:
    """Emotion is accepted as metadata but not sent as an unsupported ElevenLabs setting."""
    provider_settings = _provider_voice_settings(
        {"stability": 0.4, "similarity_boost": 0.8, "style": 0.3, "emotion": "excited"}
    )

    assert provider_settings == {"stability": 0.4, "similarity_boost": 0.8, "style": 0.3}


def test_docker_sandbox_disables_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker sandbox mode should execute generated code with networking disabled."""
    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], **_kwargs: Any) -> SimpleNamespace:
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runner.settings, "SANDBOX_BACKEND", "docker")
    monkeypatch.setattr(runner.settings, "SANDBOX_DOCKER_IMAGE", "python:3.13-slim")
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(runner.subprocess, "run", _fake_run)

    result = runner.run_code("print('ok')")

    assert result.ok is True
    assert "--network" in captured["cmd"]
    assert "none" in captured["cmd"]
    assert "--read-only" in captured["cmd"]


def test_webhook_notification_posts_terminal_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured webhooks should receive completion/failure payloads."""
    posted: dict[str, Any] = {}
    job = SimpleNamespace(
        id="job-1",
        status="done",
        current_step="done",
        review_status="approved",
        mode="code_tutorial",
        topic="Demo",
        language="en",
        url=None,
        error_message=None,
        artifacts={},
    )

    class _Response:
        """Tiny httpx response stand-in."""

        def raise_for_status(self) -> None:
            """Pretend the webhook accepted the payload."""

    monkeypatch.setattr(notifications.settings, "VIDEO_WEBHOOK_URL", "https://n8n.example/webhook/video")
    monkeypatch.setattr(notifications.settings, "VIDEO_WEBHOOK_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(notifications.settings, "VIDEO_WEBHOOK_INCLUDE_ARTIFACTS", False)
    monkeypatch.setattr(notifications.video_store, "get_job", lambda _job_id: job)

    def _fake_post(url: str, json: dict[str, Any], timeout: float) -> _Response:
        posted.update({"url": url, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(notifications.httpx, "post", _fake_post)

    notifications.notify_job_status("job-1", event="video.job.completed")

    assert posted["url"] == "https://n8n.example/webhook/video"
    assert posted["json"]["event"] == "video.job.completed"
    assert posted["json"]["status"] == "done"
