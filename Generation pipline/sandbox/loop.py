"""sandbox/loop.py — Bounded self-healing execution loop.

Runs code → if it fails, asks an LLM to fix it → retries → repeat up to
config.max_correction_attempts times.  Returns a :class:`HealingResult`
with the final code and execution trace.
"""

from __future__ import annotations

import json
import logging
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
)

import anthropic

from .config import SandboxConfig
from .parser import ExecutionResult
from .prompt_builder import build_correction_prompt
from .runner import SandboxRunner

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────


class HealingResult:
    """Full audit trail of the self-healing process."""

    def __init__(
        self,
        *,
        final_code: str,
        final_result: ExecutionResult,
        attempts: int,
        history: List[Tuple[str, ExecutionResult]],
        healed: bool,
    ) -> None:
        """Initialize the pipeline with the provided configuration."""
        self.final_code = final_code
        self.final_result = final_result
        self.attempts = attempts
        self.history = history  # [(code, result), ...]
        self.healed = healed  # True if code ran successfully

    def __repr__(self) -> str:
        """Initialize the pipeline with the provided configuration."""
        status = "✓ healed" if self.healed else "✗ exhausted"
        return f"<HealingResult {status} after {self.attempts} attempt(s)>"


class SelfHealingLoop:
    """Orchestrates: sandbox run → parse → LLM correction → sandbox run → ….

    Uses the Anthropic SDK for correction calls.  The client reads the
    ``ANTHROPIC_API_KEY`` environment variable automatically.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        """Initialize the pipeline with the provided configuration."""
        self.config = config or SandboxConfig()
        self.runner = SandboxRunner(self.config)
        self._client: Optional[anthropic.Anthropic] = None
        self._ensure_log_dir()

    # ──────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────

    def run(self, code: str) -> HealingResult:
        """Execute *code*, self-correcting up to max_correction_attempts times."""
        history: List[Tuple[str, ExecutionResult]] = []
        current_code = code
        final_result: Optional[ExecutionResult] = None

        for attempt in range(1, self.config.max_correction_attempts + 2):
            logger.info("Sandbox attempt %d", attempt)
            result = self.runner.run(current_code)
            final_result = result
            history.append((current_code, result))
            self._log_attempt(attempt, current_code, result)

            if result.success:
                logger.info("Code executed successfully on attempt %d", attempt)
                return HealingResult(
                    final_code=current_code,
                    final_result=result,
                    attempts=attempt,
                    history=history,
                    healed=True,
                )

            if attempt > self.config.max_correction_attempts:
                logger.warning("Exhausted %d correction attempts", self.config.max_correction_attempts)
                break

            logger.info(
                "Attempt %d failed (%s). Requesting LLM correction…",
                attempt,
                result.errors[0].error_type if result.errors else "unknown",
            )
            corrected = self._request_correction(result, attempt, history[:-1])
            if corrected is None or corrected.strip() == current_code.strip():
                logger.warning("LLM returned identical code; aborting loop.")
                break
            current_code = corrected

        return HealingResult(
            final_code=current_code,
            final_result=final_result,  # type: ignore[arg-type]
            attempts=len(history),
            history=history,
            healed=False,
        )

    async def run_async(self, code: str) -> HealingResult:
        """Async façade — runs the sync loop in a thread pool."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, code)

    # ──────────────────────────────────────────────────────────────────
    # LLM correction
    # ──────────────────────────────────────────────────────────────────

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _request_correction(
        self,
        result: ExecutionResult,
        attempt: int,
        history: List[Tuple[str, ExecutionResult]],
    ) -> Optional[str]:
        """Call the Anthropic API to get a corrected version of the code."""
        system_prompt, user_prompt = build_correction_prompt(result=result, attempt=attempt, history=history)
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=4096,
                temperature=self.config.correction_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            corrected = response.content[0].text.strip()
            # Strip accidental markdown fences
            if corrected.startswith("```"):
                lines = corrected.splitlines()
                # Remove first and last fence lines
                inner = []
                in_fence = False
                for line in lines:
                    if line.startswith("```") and not in_fence:
                        in_fence = True
                        continue
                    if line.startswith("```") and in_fence:
                        break
                    if in_fence:
                        inner.append(line)
                corrected = "\n".join(inner)
            return corrected
        except Exception as exc:
            logger.error("LLM correction call failed: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────

    def _ensure_log_dir(self) -> None:
        Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)

    def _log_attempt(self, attempt: int, code: str, result: ExecutionResult) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt,
            "success": result.success,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
            "errors": [
                {
                    "type": e.error_type,
                    "message": e.error_message,
                    "line": e.first_failing_line,
                }
                for e in result.errors
            ],
            "stdout_preview": result.stdout[:200],
            "code_length": len(code),
        }
        try:
            with open(self.config.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Could not write execution log: %s", exc)
