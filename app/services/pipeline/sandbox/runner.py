"""Resource-limited subprocess runner for untrusted generated code.

Runs a Python snippet in a child process with CPU-time and address-space
(memory) rlimits and a wall-clock timeout. Captures stdout/stderr/exit code into
a typed ``ExecutionResult``.

LIMITATION (MVP): outbound network is NOT hard-blocked here — a subprocess can
still open sockets. The code we run is stdlib-only Kimi output, and CPU/memory/
wall-clock are enforced. A fully isolated sandbox (Docker/nsjail network ns) is a
productization item.
"""

import importlib
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast


@dataclass
class ExecutionResult:
    """Outcome of running a code snippet."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    runtime_s: float

    @property
    def ok(self) -> bool:
        """True when the snippet ran to completion with exit code 0."""
        return self.exit_code == 0 and not self.timed_out


def _to_text(value: str | bytes | None) -> str:
    """Normalize subprocess timeout output to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _limits(cpu_seconds: int, memory_mb: int) -> Callable[[], None] | None:
    """Return a preexec_fn that applies CPU + address-space rlimits in the child."""
    if sys.platform == "win32":
        return None

    resource_module = cast(Any, importlib.import_module("resource"))

    def _apply() -> None:
        resource_module.setrlimit(resource_module.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        mem_bytes = memory_mb * 1024 * 1024
        resource_module.setrlimit(resource_module.RLIMIT_AS, (mem_bytes, mem_bytes))

    return _apply


def run_code(
    code: str,
    *,
    timeout_s: int = 15,
    cpu_seconds: int = 10,
    memory_mb: int = 256,
) -> ExecutionResult:
    """Execute ``code`` in a limited subprocess and capture the result.

    Args:
        code: Python source to run.
        timeout_s: Wall-clock timeout; the process is killed past it.
        cpu_seconds: RLIMIT_CPU for the child.
        memory_mb: RLIMIT_AS (address space) for the child, in MiB.

    Returns:
        An ``ExecutionResult`` (never raises for normal failures/timeouts).
    """
    with tempfile.TemporaryDirectory(prefix="sandbox_") as work_dir:
        script_path = os.path.join(work_dir, "snippet.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(code)

        # Minimal env: keep PATH so the interpreter resolves, drop everything else
        # (no API keys, no proxy config leaks into executed code).
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_limits(cpu_seconds, memory_mb),
                env=env,
                cwd=work_dir,
            )
            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
                runtime_s=round(time.monotonic() - started, 3),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _to_text(exc.stdout)
            stderr = _to_text(exc.stderr)
            return ExecutionResult(
                exit_code=-1,
                stdout=stdout,
                stderr=f"{stderr}\n[killed: exceeded {timeout_s}s wall-clock timeout]",
                timed_out=True,
                runtime_s=round(time.monotonic() - started, 3),
            )
