"""sandbox/runner.py — Execute untrusted Python code inside a sandbox.

Priority:
  1. Docker (if SDK available and config.use_docker is True)
  2. subprocess + resource limits (stdlib fallback)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

from .config import SandboxConfig
from .parser import (
    ExecutionResult,
    parse_execution_output,
)

logger = logging.getLogger(__name__)


class SandboxRunner:
    """Runs a Python code string in an isolated environment and returns an.

    :class:`ExecutionResult` with stdout, stderr, exit code, and timing.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        """Initialize the pipeline with the provided configuration."""
        self.config = config or SandboxConfig()
        self._docker_available = self._check_docker()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def run_async(self, code: str) -> ExecutionResult:
        """Async wrapper — runs the sync executor in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, code)

    def run(self, code: str) -> ExecutionResult:
        """Synchronously execute *code* and return a structured result."""
        if self.config.use_docker and self._docker_available:
            return self._run_docker(code)
        return self._run_subprocess(code)

    # ──────────────────────────────────────────────────────────────────
    # Docker path
    # ──────────────────────────────────────────────────────────────────

    def _check_docker(self) -> bool:
        try:
            import docker  # type: ignore

            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def _run_docker(self, code: str) -> ExecutionResult:
        import docker  # type: ignore

        client = docker.from_env()

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            t0 = time.perf_counter()
            result = client.containers.run(
                image=self.config.docker_image,
                command=["python", "/code.py"],
                volumes={tmp_path: {"bind": "/code.py", "mode": "ro"}},
                mem_limit=self.config.docker_mem_limit,
                network_disabled=self.config.docker_network_disabled,
                remove=True,
                stdout=True,
                stderr=True,
                timeout=self.config.timeout_seconds,
            )
            elapsed = time.perf_counter() - t0
            stdout = result.decode("utf-8", errors="replace")
            stderr = ""
            exit_code = 0
        except Exception as exc:  # ContainerError / APIError / timeout
            elapsed = time.perf_counter() - t0
            stdout = ""
            stderr = str(exc)
            exit_code = 1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        (stdout + stderr)[: self.config.max_output_bytes]
        return parse_execution_output(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=elapsed,
            code=code,
        )

    # ──────────────────────────────────────────────────────────────────
    # Subprocess + resource-limit fallback
    # ──────────────────────────────────────────────────────────────────

    def _run_subprocess(self, code: str) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        preexec = self._build_preexec()

        try:
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                preexec_fn=preexec,
                env=self._safe_env(),
            )
            elapsed = time.perf_counter() - t0
            stdout = proc.stdout[: self.config.max_output_bytes]
            stderr = proc.stderr[: self.config.max_output_bytes]
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            elapsed = float(self.config.timeout_seconds)
            stdout = ""
            stderr = f"TimeoutExpired: execution exceeded {self.config.timeout_seconds}s"
            exit_code = 124
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            stdout = ""
            stderr = str(exc)
            exit_code = 1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return parse_execution_output(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=elapsed,
            code=code,
        )

    @staticmethod
    def _safe_env() -> dict[str, str]:
        """Build a minimal environment for the sandboxed subprocess.

        Only benign, non-secret variables are forwarded. The full host
        environment (which may hold ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, etc.)
        must never reach model-generated code.
        """
        allowed = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "TZ")
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        return env

    @staticmethod
    def _build_preexec():
        """Return a preexec_fn that applies soft resource limits on POSIX.

        On non-POSIX systems (Windows) this returns None gracefully.
        """
        if sys.platform == "win32":
            return None

        try:
            import resource  # POSIX only

            def _limit():
                # Max CPU seconds
                resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
                # Max file size: 10 MB
                resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
                # Max open files: 64
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

            return _limit
        except ImportError:
            return None
