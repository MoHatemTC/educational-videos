"""Canonical browser vision-agent wrapper with recovery integration.

The repository now treats ``app.services.pipeline.vision`` as the canonical
vision-agent package. This wrapper is async-friendly at the step boundary while
keeping ``RecoveryManager`` sync: recovery decisions, retry guards, screenshot
hashing, and JSONL event writes are deterministic local work and are normally
run inside the decoupled worker process that owns the browser session.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from PIL import Image

from app.services.pipeline.vision.recovery import RecoveryDecision, RecoveryManager


class VisionBackend(Protocol):
    """Minimal backend contract for DOM-independent vision automation."""

    def screenshot(self) -> Image.Image | bytes | Awaitable[Image.Image | bytes]:
        """Capture the current browser screenshot."""
        raise NotImplementedError


class VisionComputerUseAgent:
    """Observe/action/recovery loop for a coordinate-based vision backend."""

    def __init__(self, backend: VisionBackend, recovery_manager: RecoveryManager) -> None:
        """Initialize the agent with a screenshot/action backend."""
        self.backend = backend
        self.recovery_manager = recovery_manager

    async def run_step(
        self,
        step_id: str,
        task_step: Callable[[], Any],
        expected_screenshot: bytes | Image.Image | None = None,
    ) -> RecoveryDecision:
        """Run Observe -> Act -> Recovery Hook for one async-compatible step."""
        await self._screenshot()
        await _maybe_await(task_step())

        after_action = await self._screenshot()
        return self.recovery_manager.post_observe_recovery_hook(
            step_id=step_id,
            original_step=task_step,
            expected_screenshot=expected_screenshot,
            observed_screenshot=after_action,
            backend=self.backend,
            context={"url": await self.current_url()},
        )

    async def current_url(self) -> str:
        """Return the current URL using optional backend accessors only."""
        for name in ("current_url", "url"):
            value = getattr(self.backend, name, None)
            try:
                if callable(value):
                    value = value()
                value = await _maybe_await(value)
                if value:
                    return str(value)
            except Exception:  # noqa: BLE001 - best-effort optional backend metadata
                continue
        page = getattr(self.backend, "page", None)
        value = getattr(page, "url", None)
        value = await _maybe_await(value)
        return str(value) if value else "unknown"

    async def _screenshot(self) -> bytes | Image.Image:
        """Capture a screenshot from a sync or async backend."""
        return await _maybe_await(self.backend.screenshot())


async def _maybe_await(value: Any) -> Any:
    """Await values from async backends while accepting sync test doubles."""
    if inspect.isawaitable(value):
        return await value
    return value
