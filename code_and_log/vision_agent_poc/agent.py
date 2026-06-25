"""Minimal vision-only computer-use agent loop with recovery integration."""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image

from vision_agent_poc.recovery import RecoveryDecision, RecoveryManager


class VisionComputerUseAgent:
    """Minimal vision-only computer-use agent loop with recovery integration."""

    def __init__(self, backend: Any, recovery_manager: RecoveryManager) -> None:
        """Initialize the agent with a screenshot/action backend."""
        self.backend = backend
        self.recovery_manager = recovery_manager

    def run_step(
        self,
        step_id: str,
        task_step: Callable[[], Any],
        expected_screenshot: bytes | Image.Image | None = None,
    ) -> RecoveryDecision:
        """Run Observe -> Reason/Plan/Act -> Recovery Hook for one step."""
        self.backend.screenshot()

        # Reason/Plan/Act is represented by the caller-provided computer-use
        # action. The recovery layer observes only screenshots after it runs.
        task_step()

        after_action = self.backend.screenshot()
        return self.recovery_manager.post_observe_recovery_hook(
            step_id=step_id,
            original_step=task_step,
            expected_screenshot=expected_screenshot,
            observed_screenshot=after_action,
            backend=self.backend,
            context={"url": self.current_url()},
        )

    def current_url(self) -> str:
        """Return the current URL using optional backend accessors only."""
        for name in ("current_url", "url"):
            value = getattr(self.backend, name, None)
            try:
                if callable(value):
                    value = value()
                if value:
                    return str(value)
            except Exception:  # noqa: BLE001
                continue
        page = getattr(self.backend, "page", None)
        value = getattr(page, "url", None)
        return str(value) if value else "unknown"
