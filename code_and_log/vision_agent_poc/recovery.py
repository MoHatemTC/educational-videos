"""Vision-model-driven UI recovery for computer-use agents."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from PIL import Image, ImageChops, ImageStat


class InterruptionType(str, Enum):
    """Known visual interruption classes."""

    NONE = "NONE"
    COOKIE_BANNER = "COOKIE_BANNER"
    POPUP_MODAL = "POPUP_MODAL"
    LOGIN_WALL = "LOGIN_WALL"
    CAPTCHA = "CAPTCHA"
    LAYOUT_SHIFT = "LAYOUT_SHIFT"
    NAVIGATION_ERROR = "NAVIGATION_ERROR"
    UNKNOWN = "UNKNOWN"


class RecoveryOutcome(str, Enum):
    """Possible recovery outcomes."""

    NO_INTERRUPTION = "none"
    RECOVERED = "recovered"
    BLOCKED = "blocked"
    RETRY_EXHAUSTED = "retry_exhausted"
    FAILED = "failed"


@dataclass(frozen=True)
class RecoveryAction:
    """A backend-agnostic computer-use recovery action."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryTarget:
    """A vision-identified target for a recovery action."""

    label: str
    x: int
    y: int
    reason: str


@dataclass(frozen=True)
class VisionRecoveryPlan:
    """A structured interruption analysis returned by a vision model."""

    interruption_type: InterruptionType
    confidence: float
    explanation: str
    target: RecoveryTarget | None
    actions: list[RecoveryAction]
    blocked: bool = False


class VisionRecoveryClient(Protocol):
    """Protocol for vision-model recovery planning clients."""

    def analyze_interruption(
        self,
        screenshot: bytes,
        context: Mapping[str, Any] | None = None,
    ) -> VisionRecoveryPlan:
        """Analyze a screenshot and return a safe recovery plan."""
        raise NotImplementedError


@dataclass
class RecoveryConfig:
    """Configuration values for visual recovery."""

    enabled: bool = True
    mode: str = "vision_model"
    max_total_recovery_attempts: int = 8
    max_attempts_per_step: int = 3
    max_attempts_per_interruption_type: int = 2
    max_consecutive_same_class_url: int = 2
    screenshot_diff_threshold: float = 0.18
    stable_screen_threshold: float = 0.03
    recovery_wait_ms: int = 700
    blocked_types: tuple[InterruptionType, ...] = (
        InterruptionType.CAPTCHA,
        InterruptionType.LOGIN_WALL,
    )
    log_path: Path = Path("logs/recovery_events.jsonl")
    include_screenshot_hash: bool = True
    vision_provider: str = "anthropic"
    vision_model: str = field(default_factory=lambda: _model_from_environment(required=False))
    require_json_plan: bool = True
    min_confidence: float = 0.55

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecoveryConfig:
        """Load recovery configuration from a YAML file."""
        return load_recovery_config(path)


@dataclass(frozen=True)
class RecoveryEvent:
    """A structured recovery event suitable for JSONL logging."""

    timestamp: str
    url: str
    step_id: str
    interruption_class: str
    action_taken: list[str]
    retry_index: int
    outcome: str
    confidence: float | None
    reason: str
    screenshot_hash: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryDecision:
    """Decision returned after a recovery attempt."""

    outcome: RecoveryOutcome
    interruption_type: InterruptionType
    actions: list[RecoveryAction]
    should_retry_original_step: bool
    reason: str


class AnthropicVisionRecoveryClient:
    """Anthropic Claude implementation of the vision recovery client."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str | None = None,
        recovery_wait_ms: int = 700,
        min_confidence: float = 0.55,
    ) -> None:
        """Initialize with an Anthropic client or lazily create one."""
        self.model = model or _model_from_environment(required=True)
        self.recovery_wait_ms = recovery_wait_ms
        self.min_confidence = min_confidence
        if client is not None:
            self.client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            msg = "The anthropic package is required to use AnthropicVisionRecoveryClient."
            raise RuntimeError(msg) from exc
        self.client = anthropic.Anthropic()

    def analyze_interruption(
        self,
        screenshot: bytes,
        context: Mapping[str, Any] | None = None,
    ) -> VisionRecoveryPlan:
        """Ask Claude to classify the interruption and plan safe recovery."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt(context)},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(screenshot).decode("ascii"),
                            },
                        },
                    ],
                },
            ],
        )
        return _safe_plan_from_response(response)

    def _prompt(self, context: Mapping[str, Any] | None) -> str:
        """Build the strict JSON instruction prompt for Claude."""
        context_json = json.dumps(dict(context or {}), default=str, sort_keys=True)
        return (
            "You are controlling a browser using vision only. "
            "Classify the interruption type visible in the screenshot. "
            "Do not use DOM assumptions, selectors, XPath, JavaScript, or "
            "element handles. Do not solve captchas. For captchas and login "
            "walls, return blocked=true and no recovery actions. If there is "
            "a dismiss, accept, close, or continue-without-signing-in button, "
            "return its approximate screen coordinates. Return only valid JSON "
            'with this exact shape: {"interruption_type":"COOKIE_BANNER",'
            '"confidence":0.92,"explanation":"...","blocked":false,'
            '"target":{"label":"Accept all","x":842,"y":704,'
            '"reason":"Button dismisses the banner."},"actions":['
            '{"name":"click","args":{"x":842,"y":704}},'
            f'{{"name":"wait","args":{{"ms":{self.recovery_wait_ms}}}}}'
            "]}. Allowed interruption_type values are NONE, COOKIE_BANNER, "
            "POPUP_MODAL, LOGIN_WALL, CAPTCHA, LAYOUT_SHIFT, NAVIGATION_ERROR, "
            "and UNKNOWN. Allowed action names are click, key, wait, reload, "
            f"back, and scroll. Context: {context_json}"
        )


class RecoveryManager:
    """Detect, classify, and recover from visual UI interruptions."""

    def __init__(
        self,
        config: RecoveryConfig | None = None,
        vision_client: VisionRecoveryClient | None = None,
    ) -> None:
        """Initialize with optional config and injected vision client."""
        self.config = config or RecoveryConfig()
        self.vision_client = vision_client
        self.total_attempts = 0
        self.attempts_by_step: dict[str, int] = {}
        self.attempts_by_type: dict[InterruptionType, int] = {}
        self._last_streak_key: tuple[str, InterruptionType] | None = None
        self._same_class_url_streak = 0

    def screenshot_diff(
        self,
        previous: bytes | Image.Image,
        current: bytes | Image.Image,
    ) -> float:
        """Return a normalized visual difference score between two screenshots."""
        previous_image = _to_rgb_image(previous)
        current_image = _to_rgb_image(current)
        if previous_image.size != current_image.size:
            current_image = current_image.resize(previous_image.size)
        diff = ImageChops.difference(previous_image, current_image)
        channels = ImageStat.Stat(diff).mean
        score = sum(channels) / (255.0 * len(channels))
        return max(0.0, min(1.0, score))

    def detect_divergence(
        self,
        expected: bytes | Image.Image | None,
        current: bytes | Image.Image,
    ) -> bool:
        """Return whether current screenshot diverges from the expected screenshot."""
        if not self.config.enabled or expected is None:
            return False
        return self.screenshot_diff(expected, current) >= self.config.screenshot_diff_threshold

    def classify_interruption(
        self,
        screenshot: bytes | Image.Image,
        context: Mapping[str, Any] | None = None,
    ) -> InterruptionType:
        """Classify a screenshot by asking the configured vision client."""
        return self.analyze_interruption(screenshot, context).interruption_type

    def analyze_interruption(
        self,
        screenshot: bytes | Image.Image,
        context: Mapping[str, Any] | None = None,
    ) -> VisionRecoveryPlan:
        """Return a vision recovery plan for a screenshot."""
        return self._vision_client().analyze_interruption(
            _to_png_bytes(screenshot),
            context,
        )

    def choose_strategy(
        self,
        interruption_type: InterruptionType,
        image_size: tuple[int, int] | None = None,
    ) -> list[RecoveryAction]:
        """Return only non-coordinate fallback actions for compatible callers."""
        del image_size
        wait = self.config.recovery_wait_ms
        if interruption_type in {InterruptionType.CAPTCHA, InterruptionType.LOGIN_WALL}:
            return []
        if interruption_type == InterruptionType.UNKNOWN:
            return [
                RecoveryAction("key", {"value": "Escape"}),
                RecoveryAction("wait", {"ms": wait}),
            ]
        if interruption_type == InterruptionType.LAYOUT_SHIFT:
            return [
                RecoveryAction("wait", {"ms": wait}),
                RecoveryAction("scroll", {"dx": 0, "dy": 180}),
                RecoveryAction("wait", {"ms": wait}),
            ]
        if interruption_type == InterruptionType.NAVIGATION_ERROR:
            return [
                RecoveryAction("reload"),
                RecoveryAction("wait", {"ms": wait}),
                RecoveryAction("back"),
                RecoveryAction("wait", {"ms": wait}),
            ]
        return []

    def execute_recovery(
        self,
        actions: Sequence[RecoveryAction],
        backend: Any,
    ) -> dict[str, list[str]]:
        """Execute planned recovery actions through an optional-method backend."""
        executed: list[str] = []
        skipped: list[str] = []
        for action in actions:
            if action.name not in _ALLOWED_ACTIONS:
                skipped.append(action.name)
                continue
            method = getattr(backend, action.name, None)
            if method is None:
                skipped.append(action.name)
                continue
            method(**action.args)
            executed.append(action.name)
        if actions and not executed:
            msg = "No recovery actions could be executed by the backend."
            raise RuntimeError(msg)
        return {"executed": executed, "skipped": skipped}

    def guard_allows_retry(
        self,
        step_id: str,
        interruption_type: InterruptionType,
    ) -> bool:
        """Return whether recovery may proceed under retry limits."""
        if self.total_attempts >= self.config.max_total_recovery_attempts:
            return False
        if self.attempts_by_step.get(step_id, 0) >= self.config.max_attempts_per_step:
            return False
        if self.attempts_by_type.get(interruption_type, 0) >= self.config.max_attempts_per_interruption_type:
            return False
        return True

    def log_event(self, event: RecoveryEvent) -> None:
        """Append a recovery event as one valid JSONL line."""
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def post_observe_recovery_hook(
        self,
        *,
        step_id: str,
        original_step: Callable[[], Any] | None,
        expected_screenshot: bytes | Image.Image | None,
        observed_screenshot: bytes | Image.Image,
        backend: Any,
        context: Mapping[str, Any] | None = None,
    ) -> RecoveryDecision:
        """Analyze, recover, re-observe, confirm, and retry after observation."""
        observed_image = _to_rgb_image(observed_screenshot)
        diff_score = None if expected_screenshot is None else self.screenshot_diff(expected_screenshot, observed_image)
        url = self._resolve_url(context, backend)
        if not self.config.enabled:
            return RecoveryDecision(
                RecoveryOutcome.NO_INTERRUPTION,
                InterruptionType.NONE,
                [],
                False,
                "Recovery is disabled.",
            )
        # Screenshot diff is only a divergence gate. Classification and target
        # selection remain the vision client's job.
        if expected_screenshot is not None and not self.detect_divergence(
            expected_screenshot,
            observed_image,
        ):
            self._reset_streak()
            return self._no_interruption(
                step_id,
                url,
                observed_image,
                diff_score,
                "No meaningful divergence detected.",
            )

        failure_type = InterruptionType.UNKNOWN
        failure_actions: list[RecoveryAction] = []
        failure_attempt = self.attempts_by_step.get(step_id, 0)
        try:
            plan = self.analyze_interruption(observed_image, context)
            interruption_type = plan.interruption_type
            failure_type = interruption_type
            if interruption_type == InterruptionType.NONE:
                self._reset_streak()
                return self._no_interruption(
                    step_id,
                    url,
                    observed_image,
                    diff_score,
                    "Vision model returned NONE.",
                )
            if not self._streak_allows_retry(url, interruption_type):
                reason = "Same interruption class repeated on the same URL."
                self._log(
                    url=url,
                    step_id=step_id,
                    interruption_type=interruption_type,
                    actions=[],
                    retry_index=self.attempts_by_step.get(step_id, 0),
                    outcome=RecoveryOutcome.RETRY_EXHAUSTED,
                    confidence=plan.confidence,
                    reason=reason,
                    screenshot=observed_image,
                    details={
                        "diff_score": diff_score,
                        "streak": self._same_class_url_streak,
                    },
                )
                return RecoveryDecision(
                    RecoveryOutcome.RETRY_EXHAUSTED,
                    interruption_type,
                    [],
                    False,
                    reason,
                )
            if not self.guard_allows_retry(step_id, interruption_type):
                reason = "Recovery retry guard blocked the attempt."
                self._log(
                    url=url,
                    step_id=step_id,
                    interruption_type=interruption_type,
                    actions=[],
                    retry_index=self.attempts_by_step.get(step_id, 0),
                    outcome=RecoveryOutcome.RETRY_EXHAUSTED,
                    confidence=plan.confidence,
                    reason=reason,
                    screenshot=observed_image,
                    details={"diff_score": diff_score},
                )
                return RecoveryDecision(
                    RecoveryOutcome.RETRY_EXHAUSTED,
                    interruption_type,
                    [],
                    False,
                    reason,
                )

            attempt_number = self._record_attempt(step_id, interruption_type)
            actions = self._actions_from_plan(plan)
            failure_attempt = attempt_number
            failure_actions = actions
            if plan.blocked or interruption_type in self.config.blocked_types:
                # Login walls require user credentials or account choices; the
                # agent must escalate instead of guessing or bypassing access.
                # Captchas are explicitly blocked: detecting them is allowed,
                # but solving, evading, or automating them is not.
                reason = "Blocked interruption; no bypass attempted."
                self._log(
                    url=url,
                    step_id=step_id,
                    interruption_type=interruption_type,
                    actions=[],
                    retry_index=attempt_number,
                    outcome=RecoveryOutcome.BLOCKED,
                    confidence=plan.confidence,
                    reason=reason,
                    screenshot=observed_image,
                    details={
                        "diff_score": diff_score,
                        "explanation": plan.explanation,
                    },
                )
                return RecoveryDecision(
                    RecoveryOutcome.BLOCKED,
                    interruption_type,
                    [],
                    False,
                    reason,
                )

            execution = self.execute_recovery(actions, backend)
            after_screenshot = self._observe_after_recovery(backend)
            after_plan = self.analyze_interruption(after_screenshot, context) if after_screenshot is not None else None
            # Re-observe before retrying so the original action is repeated
            # only after the interruption is gone or the screen changed.
            details = self._recovery_details(
                expected_screenshot,
                observed_image,
                after_screenshot,
                after_plan,
            )
            details.update(
                {
                    "diff_score": diff_score,
                    "confidence": plan.confidence,
                    "explanation": plan.explanation,
                    "target": asdict(plan.target) if plan.target is not None else None,
                    "executed_actions": execution["executed"],
                    "skipped_actions": execution["skipped"],
                },
            )
            if not self._recovery_confirmed(details):
                reason = "Recovery could not be confirmed after re-observe."
                self._log(
                    url=url,
                    step_id=step_id,
                    interruption_type=interruption_type,
                    actions=actions,
                    retry_index=attempt_number,
                    outcome=RecoveryOutcome.FAILED,
                    confidence=plan.confidence,
                    reason=reason,
                    screenshot=after_screenshot or observed_image,
                    details=details,
                )
                return RecoveryDecision(
                    RecoveryOutcome.FAILED,
                    interruption_type,
                    actions,
                    False,
                    reason,
                )

            if original_step is not None:
                original_step()
            reason = "Recovery confirmed after re-observe."
            self._log(
                url=url,
                step_id=step_id,
                interruption_type=interruption_type,
                actions=actions,
                retry_index=attempt_number,
                outcome=RecoveryOutcome.RECOVERED,
                confidence=plan.confidence,
                reason=reason,
                screenshot=after_screenshot or observed_image,
                details=details,
            )
            return RecoveryDecision(
                RecoveryOutcome.RECOVERED,
                interruption_type,
                actions,
                original_step is not None,
                reason,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"Recovery failed: {exc}"
            self._log(
                url=url,
                step_id=step_id,
                interruption_type=failure_type,
                actions=failure_actions,
                retry_index=failure_attempt,
                outcome=RecoveryOutcome.FAILED,
                confidence=None,
                reason=reason,
                screenshot=observed_image,
                details={
                    "diff_score": diff_score,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            return RecoveryDecision(
                RecoveryOutcome.FAILED,
                failure_type,
                failure_actions,
                False,
                reason,
            )

    def recover_and_retry(
        self,
        *,
        step_id: str,
        original_step: Callable[[], Any] | None,
        expected_screenshot: bytes | Image.Image | None,
        current_screenshot: bytes | Image.Image,
        backend: Any,
        context: Mapping[str, Any] | None = None,
    ) -> RecoveryDecision:
        """Backward-compatible wrapper around the post-observe hook."""
        return self.post_observe_recovery_hook(
            step_id=step_id,
            original_step=original_step,
            expected_screenshot=expected_screenshot,
            observed_screenshot=current_screenshot,
            backend=backend,
            context=context,
        )

    def _vision_client(self) -> VisionRecoveryClient:
        """Return the injected vision client or create the configured default."""
        if self.vision_client is None:
            self.vision_client = AnthropicVisionRecoveryClient(
                model=self.config.vision_model,
                recovery_wait_ms=self.config.recovery_wait_ms,
                min_confidence=self.config.min_confidence,
            )
        return self.vision_client

    def _actions_from_plan(self, plan: VisionRecoveryPlan) -> list[RecoveryAction]:
        """Prefer vision-planned actions and create minimal safe fallbacks."""
        if plan.interruption_type in self.config.blocked_types or plan.blocked:
            return []
        if plan.interruption_type == InterruptionType.UNKNOWN:
            return self.choose_strategy(InterruptionType.UNKNOWN)
        if plan.actions:
            return list(plan.actions)
        if plan.target is not None:
            return [
                RecoveryAction("click", {"x": plan.target.x, "y": plan.target.y}),
                RecoveryAction("wait", {"ms": self.config.recovery_wait_ms}),
            ]
        return self.choose_strategy(plan.interruption_type)

    def _record_attempt(
        self,
        step_id: str,
        interruption_type: InterruptionType,
    ) -> int:
        """Increment retry counters and return the step attempt number."""
        self.total_attempts += 1
        self.attempts_by_step[step_id] = self.attempts_by_step.get(step_id, 0) + 1
        self.attempts_by_type[interruption_type] = self.attempts_by_type.get(interruption_type, 0) + 1
        return self.attempts_by_step[step_id]

    def _observe_after_recovery(self, backend: Any) -> Image.Image | None:
        """Take a screenshot after recovery if the backend supports it."""
        screenshot_method = getattr(backend, "screenshot", None)
        if screenshot_method is None:
            return None
        return _to_rgb_image(screenshot_method())

    def _recovery_details(
        self,
        expected_screenshot: bytes | Image.Image | None,
        before: Image.Image,
        after: Image.Image | None,
        after_plan: VisionRecoveryPlan | None,
    ) -> dict[str, Any]:
        """Build details describing whether recovery was confirmed."""
        if after is None:
            return {
                "reobserved": False,
                "vision_after_type": None,
                "vision_after_none": False,
            }
        before_after_diff = self.screenshot_diff(before, after)
        details: dict[str, Any] = {
            "reobserved": True,
            "before_after_diff": before_after_diff,
            "screen_changed_meaningfully": (before_after_diff >= self.config.stable_screen_threshold),
            "vision_after_type": (after_plan.interruption_type.value if after_plan is not None else None),
            "vision_after_none": (after_plan is not None and after_plan.interruption_type == InterruptionType.NONE),
        }
        if expected_screenshot is not None:
            before_expected = self.screenshot_diff(expected_screenshot, before)
            after_expected = self.screenshot_diff(expected_screenshot, after)
            details["before_expected_diff"] = before_expected
            details["after_expected_diff"] = after_expected
            details["closer_to_expected"] = after_expected < before_expected
        return details

    def _recovery_confirmed(self, details: Mapping[str, Any]) -> bool:
        """Return whether re-observation confirmed recovery."""
        return bool(
            details.get("vision_after_none")
            or details.get("screen_changed_meaningfully")
            or details.get("closer_to_expected")
        )

    def _streak_allows_retry(
        self,
        url: str,
        interruption_type: InterruptionType,
    ) -> bool:
        """Track consecutive same-class same-URL failures."""
        key = (url, interruption_type)
        # Same-class same-URL streaks catch loops that normal per-step counters
        # can miss when the agent keeps revisiting the same blocked surface.
        if self._last_streak_key == key:
            self._same_class_url_streak += 1
        else:
            self._last_streak_key = key
            self._same_class_url_streak = 1
        return self._same_class_url_streak <= self.config.max_consecutive_same_class_url

    def _reset_streak(self) -> None:
        """Reset the same-class same-URL streak after stable or NONE screens."""
        self._last_streak_key = None
        self._same_class_url_streak = 0

    def _resolve_url(
        self,
        context: Mapping[str, Any] | None,
        backend: Any,
    ) -> str:
        """Resolve the current URL from context or optional backend accessors."""
        if context is not None and context.get("url"):
            return str(context["url"])
        for name in ("current_url", "url"):
            value = getattr(backend, name, None)
            try:
                if callable(value):
                    value = value()
                if value:
                    return str(value)
            except Exception:  # noqa: BLE001
                continue
        page = getattr(backend, "page", None)
        value = getattr(page, "url", None)
        return str(value) if value else "unknown"

    def _no_interruption(
        self,
        step_id: str,
        url: str,
        screenshot: Image.Image,
        diff_score: float | None,
        reason: str,
    ) -> RecoveryDecision:
        """Log and return a no-interruption decision."""
        self._log(
            url=url,
            step_id=step_id,
            interruption_type=InterruptionType.NONE,
            actions=[],
            retry_index=self.attempts_by_step.get(step_id, 0),
            outcome=RecoveryOutcome.NO_INTERRUPTION,
            confidence=None,
            reason=reason,
            screenshot=screenshot,
            details={"diff_score": diff_score},
        )
        return RecoveryDecision(
            RecoveryOutcome.NO_INTERRUPTION,
            InterruptionType.NONE,
            [],
            False,
            reason,
        )

    def _log(
        self,
        *,
        url: str,
        step_id: str,
        interruption_type: InterruptionType,
        actions: Sequence[RecoveryAction],
        retry_index: int,
        outcome: RecoveryOutcome,
        confidence: float | None,
        reason: str,
        screenshot: bytes | Image.Image | None,
        details: dict[str, Any],
    ) -> None:
        """Create and write a recovery event."""
        event = RecoveryEvent(
            timestamp=_timestamp(),
            url=url,
            step_id=step_id,
            interruption_class=interruption_type.value,
            action_taken=[action.name for action in actions],
            retry_index=retry_index,
            outcome=outcome.value,
            confidence=confidence,
            reason=reason,
            screenshot_hash=(
                _screenshot_hash(screenshot)
                if screenshot is not None and self.config.include_screenshot_hash
                else None
            ),
            details=details,
        )
        self.log_event(event)


def _model_from_environment(*, required: bool) -> str:
    """Return the configured Anthropic model from the environment.

    The recovery layer deliberately does not own a hardcoded source-code
    model ID. In the integrated app this should come from DEFAULT_LLM_MODEL;
    ANTHROPIC_MODEL is accepted for standalone code_and_log runs.
    """
    model = os.getenv("ANTHROPIC_MODEL") or os.getenv("DEFAULT_LLM_MODEL") or ""
    if required and not model:
        msg = (
            "Set ANTHROPIC_MODEL or DEFAULT_LLM_MODEL to a valid Anthropic "
            "model ID before creating AnthropicVisionRecoveryClient."
        )
        raise RuntimeError(msg)
    return model


def _resolve_env_template(value: Any) -> Any:
    """Resolve simple ${ENV:-fallback} config values without adding dependencies."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not (stripped.startswith("${") and stripped.endswith("}")):
        return value
    expression = stripped[2:-1]
    name, separator, fallback = expression.partition(":-")
    env_value = os.getenv(name.strip())
    if env_value:
        return env_value
    if separator:
        return fallback
    return ""


def load_recovery_config(path: str | Path) -> RecoveryConfig:
    """Load recovery config from an agent YAML file."""
    try:
        import yaml
    except ImportError:
        raw = _load_simple_yaml(path)
    else:
        with Path(path).open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}

    recovery = raw.get("recovery", raw)
    logging = raw.get("logging", {})
    vision_model = raw.get("vision_model", {})
    values: dict[str, Any] = {}
    for field_name in (
        "enabled",
        "mode",
        "max_total_recovery_attempts",
        "max_attempts_per_step",
        "max_attempts_per_interruption_type",
        "max_consecutive_same_class_url",
        "screenshot_diff_threshold",
        "stable_screen_threshold",
        "recovery_wait_ms",
    ):
        if field_name in recovery:
            values[field_name] = recovery[field_name]

    blocked_types = recovery.get("blocked_types")
    if blocked_types is not None:
        values["blocked_types"] = tuple(_interruption_type_from_string(item) for item in blocked_types)

    log_path = logging.get("path", recovery.get("log_path"))
    if log_path is not None:
        values["log_path"] = Path(log_path)

    include_hash = logging.get("include_screenshot_hash")
    include_hash = recovery.get("include_screenshot_hash", include_hash)
    if include_hash is not None:
        values["include_screenshot_hash"] = bool(include_hash)

    if "provider" in vision_model:
        values["vision_provider"] = vision_model["provider"]
    if "model" in vision_model:
        values["vision_model"] = str(_resolve_env_template(vision_model["model"]))
    if "require_json_plan" in vision_model:
        values["require_json_plan"] = bool(vision_model["require_json_plan"])
    if "min_confidence" in vision_model:
        values["min_confidence"] = float(vision_model["min_confidence"])

    return RecoveryConfig(**values)


def _json_from_model_response(response: Any) -> Mapping[str, Any]:
    """Extract strict JSON from an Anthropic response object."""
    texts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text is None and isinstance(block, Mapping):
            text = block.get("text")
        if text:
            texts.append(str(text))
    raw_text = "".join(texts).strip()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        msg = "Anthropic vision recovery response was not valid JSON."
        raise ValueError(msg) from exc
    if not isinstance(data, Mapping):
        msg = "Anthropic vision recovery JSON must be an object."
        raise ValueError(msg)
    return data


def _safe_plan_from_response(response: Any) -> VisionRecoveryPlan:
    """Parse a model response, warning and falling back to NONE on bad output."""
    try:
        return _plan_from_mapping(_json_from_model_response(response))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Invalid vision recovery model output: %s", exc)
        return _fallback_none_plan()


def _plan_from_mapping(data: Mapping[str, Any]) -> VisionRecoveryPlan:
    """Validate and convert JSON-like data into a recovery plan."""
    raw_type = data.get("interruption_type")
    if raw_type is None:
        logging.warning("Vision recovery model output missing interruption_type.")
        return _fallback_none_plan()
    try:
        interruption_type = _interruption_type_from_string(raw_type)
    except ValueError:
        logging.warning("Unexpected interruption class from model: %s", raw_type)
        return _fallback_none_plan()
    target = _target_from_mapping(data.get("target"))
    actions = [
        action for action in (_action_from_mapping(item) for item in data.get("actions", [])) if action is not None
    ]
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        logging.warning("Invalid confidence from model: %s", data.get("confidence"))
        confidence = 0.0
    return VisionRecoveryPlan(
        interruption_type=interruption_type,
        confidence=confidence,
        explanation=str(data.get("explanation", "")),
        target=target,
        actions=actions,
        blocked=bool(data.get("blocked", False)),
    )


def _target_from_mapping(value: Any) -> RecoveryTarget | None:
    """Convert a JSON target object into a recovery target."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        logging.warning("Ignoring invalid recovery target: %s", value)
        return None
    try:
        return RecoveryTarget(
            label=str(value.get("label", "")),
            x=int(value["x"]),
            y=int(value["y"]),
            reason=str(value.get("reason", "")),
        )
    except (KeyError, TypeError, ValueError):
        logging.warning("Ignoring recovery target with invalid coordinates: %s", value)
        return None


def _action_from_mapping(value: Any) -> RecoveryAction | None:
    """Convert a JSON action object into a recovery action."""
    if not isinstance(value, Mapping):
        logging.warning("Ignoring invalid recovery action: %s", value)
        return None
    name = str(value.get("name", ""))
    if name not in _ALLOWED_ACTIONS:
        logging.warning("Ignoring unsupported recovery action: %s", name)
        return None
    args = value.get("args", {})
    if not isinstance(args, Mapping):
        logging.warning("Ignoring recovery action with invalid args: %s", value)
        return None
    return RecoveryAction(name=name, args=dict(args))


def _fallback_none_plan() -> VisionRecoveryPlan:
    """Return the required safe fallback plan for unexpected model output."""
    return VisionRecoveryPlan(
        interruption_type=InterruptionType.NONE,
        confidence=0.0,
        explanation="Unexpected model output; falling back to NONE.",
        target=None,
        actions=[],
        blocked=False,
    )


def _to_rgb_image(value: bytes | Image.Image) -> Image.Image:
    """Convert supported screenshot input to an RGB Pillow image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.open(BytesIO(value)).convert("RGB")


def _to_png_bytes(value: bytes | Image.Image) -> bytes:
    """Normalize supported screenshot input to PNG bytes."""
    image = _to_rgb_image(value)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _screenshot_hash(value: bytes | Image.Image) -> str:
    """Return a sha256 hash of normalized PNG screenshot bytes."""
    return hashlib.sha256(_to_png_bytes(value)).hexdigest()


def _timestamp() -> str:
    """Return the current local time as an ISO timestamp with timezone."""
    return datetime.now().astimezone().isoformat()


def _interruption_type_from_string(value: Any) -> InterruptionType:
    """Convert config or model strings into interruption enum values."""
    if isinstance(value, InterruptionType):
        return value
    normalized = str(value).strip().upper()
    try:
        return InterruptionType[normalized]
    except KeyError as exc:
        msg = f"Unknown interruption type: {value}"
        raise ValueError(msg) from exc


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Parse the limited YAML subset used by agent_config.yaml when PyYAML is absent."""
    root: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    current_list_key: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            key = line[:-1].strip()
            root[key] = {}
            section = root[key]
            current_list_key = None
            continue
        if section is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_list_key is not None:
            section[current_list_key].append(_parse_simple_yaml_scalar(stripped[2:].strip()))
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            section[key] = []
            current_list_key = key
        else:
            section[key] = _parse_simple_yaml_scalar(value)
            current_list_key = None
    return root


def _parse_simple_yaml_scalar(value: str) -> Any:
    """Parse a basic YAML scalar."""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


_ALLOWED_ACTIONS = frozenset({"click", "key", "wait", "reload", "back", "scroll"})
