"""Tests for the LangGraph video-pipeline routing helpers."""

from app.services.pipeline.graph import _route_mode, _route_sandbox


def test_route_mode_sends_web_jobs_to_web_branch() -> None:
    """Web explainer jobs should start at the web capture node."""
    assert _route_mode({"mode": "web_explainer"}) == "web_capture"


def test_route_mode_defaults_to_code_tutorial_branch() -> None:
    """Non-web jobs should start at the research node."""
    assert _route_mode({"mode": "code_tutorial"}) == "research"


def test_route_sandbox_repairs_failed_code_before_retry_limit() -> None:
    """A failed sandbox attempt should route to repair while attempts remain."""
    assert _route_sandbox({"code_validated": False, "sandbox_attempt": 0}) == "sandbox_repair"


def test_route_sandbox_continues_after_success() -> None:
    """Validated code should continue to script generation."""
    assert _route_sandbox({"code_validated": True, "sandbox_attempt": 0}) == "script"


def test_route_sandbox_continues_after_retry_limit() -> None:
    """Exhausted repair attempts should continue with the final code artifact."""
    assert _route_sandbox({"code_validated": False, "sandbox_attempt": 3}) == "script"
