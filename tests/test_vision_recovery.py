"""Tests for the canonical vision recovery manager and agent loop."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from PIL import Image, ImageDraw

from app.services.pipeline.vision.agent import VisionComputerUseAgent
from app.services.pipeline.vision.recovery import (
    AnthropicVisionRecoveryClient,
    InterruptionType,
    RecoveryAction,
    RecoveryConfig,
    RecoveryDecision,
    RecoveryEvent,
    RecoveryManager,
    RecoveryOutcome,
    RecoveryTarget,
    VisionRecoveryPlan,
)
from app.core.config import settings as recovery_settings


REQUIRED_LOG_KEYS = {
    "timestamp",
    "url",
    "step_id",
    "interruption_class",
    "action_taken",
    "retry_index",
    "outcome",
    "confidence",
    "reason",
    "screenshot_hash",
    "details",
}


def make_blank_page() -> Image.Image:
    """Create a simple stable application page."""
    image = Image.new("RGB", (1024, 768), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 56, 960, 112), fill=(244, 246, 248), outline=(215, 220, 225))
    draw.rectangle((90, 150, 420, 185), fill=(235, 239, 242))
    draw.rectangle((90, 215, 920, 640), fill=(250, 250, 250), outline=(228, 228, 228))
    return image


def make_interrupted_page() -> Image.Image:
    """Create a visually divergent page for diff tests."""
    image = make_blank_page()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 500, 1024, 768), fill=(20, 30, 45))
    return image


def image_to_bytes(image: Image.Image) -> bytes:
    """Serialize an image to PNG bytes."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeVisionRecoveryClient:
    """Fake vision client that returns queued plans and records calls."""

    def __init__(self, plans: list[VisionRecoveryPlan]) -> None:
        """Initialize with deterministic plans."""
        self.plans = list(plans)
        self.calls: list[tuple[bytes, Mapping[str, Any] | None]] = []

    def analyze_interruption(
        self,
        screenshot: bytes,
        context: Mapping[str, Any] | None = None,
    ) -> VisionRecoveryPlan:
        """Return the next queued plan or a no-interruption plan."""
        self.calls.append((screenshot, context))
        if self.plans:
            return self.plans.pop(0)
        return none_plan()


class FakeComputerBackend:
    """Fake backend recording computer-use style actions."""

    def __init__(
        self,
        screenshots: list[Image.Image] | None = None,
        order: list[str] | None = None,
        url: str = "https://example.com/page",
    ) -> None:
        """Initialize with optional queued screenshots and ordering log."""
        self.actions: list[tuple] = []
        self.screenshots = screenshots or []
        self.order = order
        self.url = url

    def screenshot(self) -> Image.Image:
        """Return the next queued screenshot."""
        self.actions.append(("screenshot",))
        if self.order is not None:
            self.order.append("screenshot")
        if self.screenshots:
            return self.screenshots.pop(0)
        return make_blank_page()

    def key(self, value: str) -> None:
        """Record a key press."""
        self.actions.append(("key", value))

    def click(self, x: int, y: int) -> None:
        """Record a mouse click."""
        self.actions.append(("click", x, y))

    def wait(self, ms: int) -> None:
        """Record a wait."""
        self.actions.append(("wait", ms))

    def reload(self) -> None:
        """Record a page reload."""
        self.actions.append(("reload",))

    def back(self) -> None:
        """Record browser back."""
        self.actions.append(("back",))

    def scroll(self, dx: int, dy: int) -> None:
        """Record a scroll."""
        self.actions.append(("scroll", dx, dy))


class FakeAnthropicClient:
    """Fake Anthropic client returning a fixed text response."""

    def __init__(self, text: str) -> None:
        """Initialize with response text."""
        self.messages = SimpleNamespace(create=self.create)
        self.text = text

    def create(self, **_kwargs: Any) -> Any:
        """Return a minimal Anthropic-like response object."""
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def none_plan() -> VisionRecoveryPlan:
    """Return a no-interruption vision plan."""
    return VisionRecoveryPlan(
        interruption_type=InterruptionType.NONE,
        confidence=1.0,
        explanation="No interruption.",
        target=None,
        actions=[],
        blocked=False,
    )


def plan(
    interruption_type: InterruptionType,
    *,
    target: RecoveryTarget | None = None,
    actions: list[RecoveryAction] | None = None,
    blocked: bool = False,
) -> VisionRecoveryPlan:
    """Build a deterministic fake vision plan."""
    return VisionRecoveryPlan(
        interruption_type=interruption_type,
        confidence=0.92,
        explanation=f"Detected {interruption_type.value}.",
        target=target,
        actions=actions or [],
        blocked=blocked,
    )


def manager_with_fake_client(
    tmp_path: Path,
    plans: list[VisionRecoveryPlan],
    config: RecoveryConfig | None = None,
) -> tuple[RecoveryManager, FakeVisionRecoveryClient]:
    """Create a recovery manager with temp logging and fake vision."""
    fake = FakeVisionRecoveryClient(plans)
    recovery_config = config or RecoveryConfig(log_path=tmp_path / "events.jsonl")
    recovery_config.log_path = tmp_path / "events.jsonl"
    return RecoveryManager(config=recovery_config, vision_client=fake), fake


def recovery_config(tmp_path: Path, **overrides: Any) -> RecoveryConfig:
    """Create a recovery config with generous defaults for loop tests."""
    values = {
        "log_path": tmp_path / "events.jsonl",
        "max_total_recovery_attempts": 10,
        "max_attempts_per_step": 10,
        "max_attempts_per_interruption_type": 10,
    }
    values.update(overrides)
    return RecoveryConfig(**values)


def escape_action() -> RecoveryAction:
    """Return a valid Escape key recovery action."""
    return RecoveryAction("key", {"value": "Escape"})


def run_recovery(
    manager: RecoveryManager,
    *,
    url: str = "https://example.com/page",
    backend: FakeComputerBackend | None = None,
) -> RecoveryDecision:
    """Run a standard recovery attempt against an interrupted page."""
    return manager.recover_and_retry(
        step_id="step",
        original_step=None,
        expected_screenshot=make_blank_page(),
        current_screenshot=make_interrupted_page(),
        backend=backend or FakeComputerBackend(screenshots=[make_blank_page()]),
        context={"url": url},
    )


def test_recovery_manager_calls_vision_client(tmp_path: Path) -> None:
    """Classification comes from the vision client."""
    manager, fake = manager_with_fake_client(tmp_path, [none_plan()])
    decision = run_recovery(manager, url="https://example.com/unit")
    assert decision.outcome == RecoveryOutcome.NO_INTERRUPTION
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0][0], bytes)
    assert fake.calls[0][1] == {"url": "https://example.com/unit"}


def test_cookie_banner_uses_vision_target_coordinates(tmp_path: Path) -> None:
    """Cookie recovery clicks the coordinate returned by the vision plan."""
    target = RecoveryTarget("Accept all", 123, 456, "Dismisses banner.")
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [plan(InterruptionType.COOKIE_BANNER, target=target), none_plan()],
    )
    backend = FakeComputerBackend(screenshots=[make_blank_page()])
    decision = run_recovery(manager, backend=backend)
    assert decision.outcome == RecoveryOutcome.RECOVERED
    assert ("click", 123, 456) in backend.actions
    assert ("click", 880, 675) not in backend.actions


def test_popup_uses_vision_target_coordinates(tmp_path: Path) -> None:
    """Popup recovery follows explicit coordinates from the vision plan."""
    actions = [
        RecoveryAction("click", {"x": 321, "y": 222}),
        RecoveryAction("wait", {"ms": 700}),
    ]
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [plan(InterruptionType.POPUP_MODAL, actions=actions), none_plan()],
    )
    backend = FakeComputerBackend(screenshots=[make_blank_page()])
    decision = run_recovery(manager, backend=backend)
    assert decision.outcome == RecoveryOutcome.RECOVERED
    assert ("click", 321, 222) in backend.actions


def test_login_wall_blocks_from_vision_plan(tmp_path: Path) -> None:
    """Login walls block when the vision plan marks them blocked."""
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [plan(InterruptionType.LOGIN_WALL, blocked=True)],
    )
    backend = FakeComputerBackend()
    decision = run_recovery(manager, backend=backend)
    assert decision.outcome == RecoveryOutcome.BLOCKED
    assert backend.actions == []


def test_captcha_blocks_from_vision_plan(tmp_path: Path) -> None:
    """Captchas block without any solving or bypass actions."""
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [plan(InterruptionType.CAPTCHA, blocked=True)],
    )
    backend = FakeComputerBackend()
    decision = run_recovery(manager, backend=backend)
    assert decision.outcome == RecoveryOutcome.BLOCKED
    assert decision.actions == []
    assert backend.actions == []


def test_recovery_executes_planned_actions(tmp_path: Path) -> None:
    """The manager executes the action sequence returned by the vision model."""
    actions = [
        RecoveryAction("reload"),
        RecoveryAction("wait", {"ms": 700}),
        RecoveryAction("back"),
    ]
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [plan(InterruptionType.NAVIGATION_ERROR, actions=actions), none_plan()],
    )
    backend = FakeComputerBackend(screenshots=[make_blank_page()])
    decision = run_recovery(manager, backend=backend)
    assert decision.outcome == RecoveryOutcome.RECOVERED
    assert backend.actions[:3] == [("reload",), ("wait", 700), ("back",)]


def test_reobserve_happens_before_retry(tmp_path: Path) -> None:
    """The original step is retried only after the post-action screenshot."""
    order: list[str] = []

    def original_step() -> None:
        order.append("original")

    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(
                InterruptionType.COOKIE_BANNER,
                actions=[RecoveryAction("key", {"value": "Escape"})],
            ),
            none_plan(),
        ],
    )
    backend = FakeComputerBackend(screenshots=[make_blank_page()], order=order)
    decision = manager.recover_and_retry(
        step_id="ordering",
        original_step=original_step,
        expected_screenshot=make_blank_page(),
        current_screenshot=make_interrupted_page(),
        backend=backend,
        context={"url": "https://example.com/order"},
    )
    assert decision.outcome == RecoveryOutcome.RECOVERED
    assert order == ["screenshot", "original"]


def test_original_step_retried_after_successful_reobserve(tmp_path: Path) -> None:
    """A confirmed recovery retries the original step once."""
    calls = {"count": 0}

    def original_step() -> None:
        calls["count"] += 1

    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(
                InterruptionType.POPUP_MODAL,
                actions=[RecoveryAction("key", {"value": "Escape"})],
            ),
            none_plan(),
        ],
    )
    decision = manager.recover_and_retry(
        step_id="retry",
        original_step=original_step,
        expected_screenshot=make_blank_page(),
        current_screenshot=make_interrupted_page(),
        backend=FakeComputerBackend(screenshots=[make_blank_page()]),
        context={"url": "https://example.com/retry"},
    )
    assert decision.should_retry_original_step is True
    assert calls["count"] == 1


def test_loop_guard_stops_after_limit(tmp_path: Path) -> None:
    """Basic retry counters still return retry exhausted."""
    config = recovery_config(tmp_path, max_total_recovery_attempts=1)
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(
                InterruptionType.COOKIE_BANNER,
                actions=[RecoveryAction("key", {"value": "Escape"})],
            ),
            none_plan(),
            plan(
                InterruptionType.COOKIE_BANNER,
                actions=[RecoveryAction("key", {"value": "Escape"})],
            ),
        ],
        config=config,
    )
    first = run_recovery(manager)
    second_backend = FakeComputerBackend(screenshots=[make_blank_page()])
    second = run_recovery(manager, backend=second_backend)
    assert first.outcome == RecoveryOutcome.RECOVERED
    assert second.outcome == RecoveryOutcome.RETRY_EXHAUSTED
    assert second_backend.actions == []


def test_same_class_same_url_streak_exhausts(tmp_path: Path) -> None:
    """Repeated same class on the same URL exhausts the streak guard."""
    config = recovery_config(tmp_path, max_consecutive_same_class_url=2)
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
            none_plan(),
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
            none_plan(),
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
        ],
        config=config,
    )
    assert run_recovery(manager).outcome == RecoveryOutcome.RECOVERED
    assert run_recovery(manager).outcome == RecoveryOutcome.RECOVERED
    assert run_recovery(manager).outcome == RecoveryOutcome.RETRY_EXHAUSTED


def test_same_class_different_url_resets_streak(tmp_path: Path) -> None:
    """Changing URLs resets the same-class streak."""
    config = recovery_config(tmp_path, max_consecutive_same_class_url=1)
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
            none_plan(),
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
            none_plan(),
        ],
        config=config,
    )
    assert run_recovery(manager, url="https://example.com/a").outcome == RecoveryOutcome.RECOVERED
    assert run_recovery(manager, url="https://example.com/b").outcome == RecoveryOutcome.RECOVERED


def test_different_class_same_url_resets_streak(tmp_path: Path) -> None:
    """Changing interruption class resets the same-URL streak."""
    config = recovery_config(tmp_path, max_consecutive_same_class_url=1)
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
            none_plan(),
            plan(InterruptionType.POPUP_MODAL, actions=[escape_action()]),
            none_plan(),
        ],
        config=config,
    )
    assert run_recovery(manager).outcome == RecoveryOutcome.RECOVERED
    assert run_recovery(manager).outcome == RecoveryOutcome.RECOVERED


def test_none_classification_resets_streak(tmp_path: Path) -> None:
    """A NONE classification resets the same-class same-URL streak."""
    config = recovery_config(tmp_path, max_consecutive_same_class_url=1)
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
            none_plan(),
            none_plan(),
            plan(InterruptionType.COOKIE_BANNER, actions=[escape_action()]),
            none_plan(),
        ],
        config=config,
    )
    assert run_recovery(manager).outcome == RecoveryOutcome.RECOVERED
    assert run_recovery(manager).outcome == RecoveryOutcome.NO_INTERRUPTION
    assert run_recovery(manager).outcome == RecoveryOutcome.RECOVERED


def test_jsonl_logging_required_schema(tmp_path: Path) -> None:
    """Recovery logs include the required JSONL schema keys."""
    manager, _fake = manager_with_fake_client(
        tmp_path,
        [
            plan(
                InterruptionType.COOKIE_BANNER,
                actions=[RecoveryAction("key", {"value": "Escape"})],
            ),
            none_plan(),
        ],
    )
    run_recovery(manager, url="https://example.com/schema")
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert set(event) == REQUIRED_LOG_KEYS
    assert event["url"] == "https://example.com/schema"
    assert event["interruption_class"] == "COOKIE_BANNER"
    assert event["action_taken"] == ["key"]
    assert event["retry_index"] == 1
    assert event["outcome"] == "recovered"
    assert event["confidence"] == 0.92
    assert "diff_score" in event["details"]


def test_manual_jsonl_logging_still_writes_valid_json(tmp_path: Path) -> None:
    """Direct log_event calls still append valid JSONL."""
    log_path = tmp_path / "manual.jsonl"
    manager = RecoveryManager(config=RecoveryConfig(log_path=log_path))
    manager.log_event(
        RecoveryEvent(
            timestamp="2026-06-22T12:00:00+03:00",
            url="unknown",
            step_id="unit",
            interruption_class="NONE",
            action_taken=[],
            retry_index=0,
            outcome="none",
            confidence=None,
            reason="No interruption.",
            screenshot_hash=None,
            details={"ok": True},
        ),
    )
    event = json.loads(log_path.read_text(encoding="utf-8"))
    assert set(event) == REQUIRED_LOG_KEYS
    assert event["details"] == {"ok": True}


def test_config_loading_uses_app_settings(tmp_path: Path, monkeypatch: Any) -> None:
    """Recovery config is now env-driven through app/core/config.py settings."""
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_MODE", "vision_model")
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_MAX_TOTAL_ATTEMPTS", 4)
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_MAX_SAME_CLASS_URL", 2)
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_PROVIDER", "anthropic")
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_MODEL", "claude-opus-4-8")
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_REQUIRE_JSON_PLAN", True)
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_MIN_CONFIDENCE", 0.55)
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_INCLUDE_SCREENSHOT_HASH", False)

    config = RecoveryConfig.from_settings()

    assert config.mode == "vision_model"
    assert config.max_total_recovery_attempts == 4
    assert config.max_consecutive_same_class_url == 2
    assert config.vision_provider == "anthropic"
    assert config.vision_model == "claude-opus-4-8"
    assert config.require_json_plan is True
    assert config.min_confidence == 0.55
    assert config.include_screenshot_hash is False


def test_screenshot_diff_still_detects_divergence(tmp_path: Path) -> None:
    """Screenshot diff remains available for divergence confirmation."""
    manager = RecoveryManager(config=RecoveryConfig(log_path=tmp_path / "events.jsonl"))
    assert manager.screenshot_diff(make_blank_page(), make_interrupted_page()) >= 0.18


def test_no_divergence_returns_no_interruption(tmp_path: Path) -> None:
    """No divergence short-circuits before calling the vision client."""
    manager, fake = manager_with_fake_client(
        tmp_path,
        [plan(InterruptionType.COOKIE_BANNER)],
    )
    decision = manager.recover_and_retry(
        step_id="same",
        original_step=None,
        expected_screenshot=make_blank_page(),
        current_screenshot=make_blank_page(),
        backend=FakeComputerBackend(),
    )
    assert decision.outcome == RecoveryOutcome.NO_INTERRUPTION
    assert fake.calls == []


@pytest.mark.asyncio
async def test_agent_loop_calls_recovery_hook_after_observe_action(tmp_path: Path) -> None:
    """The real agent loop calls the recovery hook after observe/action."""
    del tmp_path
    order: list[str] = []

    class FakeRecoveryManager:
        """Small recovery manager test double."""

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def post_observe_recovery_hook(self, **kwargs: Any) -> RecoveryDecision:
            order.append("hook")
            self.calls.append(kwargs)
            return RecoveryDecision(
                RecoveryOutcome.NO_INTERRUPTION,
                InterruptionType.NONE,
                [],
                False,
                "No interruption.",
            )

    def task_step() -> None:
        order.append("task")

    backend = FakeComputerBackend(
        screenshots=[make_blank_page(), make_interrupted_page()],
        order=order,
        url="https://example.com/agent",
    )
    recovery_manager = FakeRecoveryManager()
    agent = VisionComputerUseAgent(backend, recovery_manager)  # type: ignore[arg-type]
    decision = await agent.run_step("agent_step", task_step, make_blank_page())
    assert decision.outcome == RecoveryOutcome.NO_INTERRUPTION
    assert order == ["screenshot", "task", "screenshot", "hook"]
    assert recovery_manager.calls[0]["context"] == {"url": "https://example.com/agent"}
    assert recovery_manager.calls[0]["step_id"] == "agent_step"


def test_anthropic_client_uses_settings_model(monkeypatch: Any) -> None:
    """Anthropic client reads its model from app settings when no model is passed."""
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_MODEL", "claude-opus-4-8")
    client = AnthropicVisionRecoveryClient(client=FakeAnthropicClient('{"interruption_type":"NONE"}'))
    assert client.model == "claude-opus-4-8"


def test_anthropic_client_requires_configured_model(monkeypatch: Any) -> None:
    """Anthropic client fails clearly if no settings model is available."""
    monkeypatch.setattr(recovery_settings, "VISION_RECOVERY_MODEL", "")
    monkeypatch.setattr(recovery_settings, "DEFAULT_LLM_MODEL", "")
    try:
        AnthropicVisionRecoveryClient(client=FakeAnthropicClient('{"interruption_type":"NONE"}'))
    except RuntimeError as exc:
        assert "VISION_RECOVERY_MODEL or DEFAULT_LLM_MODEL" in str(exc)
    else:  # pragma: no cover - this branch should never run.
        raise AssertionError("Expected missing model RuntimeError.")


def test_unexpected_model_interruption_class_falls_back_to_none() -> None:
    """Unexpected model classes warn and safely fall back to NONE."""
    client = AnthropicVisionRecoveryClient(
        client=FakeAnthropicClient('{"interruption_type":"ALIEN","confidence":0.9}'),
        model="claude-opus-4-8",
    )
    plan_result = client.analyze_interruption(image_to_bytes(make_blank_page()))
    assert plan_result.interruption_type == InterruptionType.NONE
    assert plan_result.confidence == 0.0


def test_malformed_model_json_falls_back_to_none() -> None:
    """Malformed model JSON warns and safely falls back to NONE."""
    client = AnthropicVisionRecoveryClient(
        client=FakeAnthropicClient("not json"),
        model="claude-opus-4-8",
    )
    plan_result = client.analyze_interruption(image_to_bytes(make_blank_page()))
    assert plan_result.interruption_type == InterruptionType.NONE
    assert plan_result.explanation == "Unexpected model output; falling back to NONE."


def test_invalid_action_from_model_is_ignored_or_safely_falls_back() -> None:
    """Unsupported model actions are ignored while valid actions remain."""
    client = AnthropicVisionRecoveryClient(
        client=FakeAnthropicClient(
            json.dumps(
                {
                    "interruption_type": "COOKIE_BANNER",
                    "confidence": 0.9,
                    "explanation": "Banner.",
                    "target": None,
                    "actions": [
                        {"name": "dom_click", "args": {"selector": "#accept"}},
                        {"name": "click", "args": {"x": 12, "y": 34}},
                    ],
                },
            ),
        ),
        model="claude-opus-4-8",
    )
    plan_result = client.analyze_interruption(image_to_bytes(make_blank_page()))
    assert plan_result.interruption_type == InterruptionType.COOKIE_BANNER
    assert plan_result.actions == [RecoveryAction("click", {"x": 12, "y": 34})]
