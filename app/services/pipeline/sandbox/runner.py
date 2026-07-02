"""Resource-limited runner for untrusted generated code.

Default local mode runs a Python snippet in a subprocess with CPU-time and
address-space rlimits where the platform supports them. For hardened deployments,
``SANDBOX_BACKEND=docker`` runs the snippet in a throwaway container with
``--network none``, memory/PID limits, and a read-only filesystem.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from app.core.config import settings


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


def _minimal_env() -> dict[str, str]:
    """Return an environment without user secrets or proxy/API keys."""
    return {"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def _run_subprocess(
    script_path: str,
    work_dir: str,
    *,
    timeout_s: int,
    cpu_seconds: int,
    memory_mb: int,
) -> ExecutionResult:
    """Execute a snippet with the local Python interpreter."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_limits(cpu_seconds, memory_mb),
            env=_minimal_env(),
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


def _run_docker(script_path: str, work_dir: str, *, timeout_s: int, memory_mb: int) -> ExecutionResult:
    """Execute a snippet inside Docker with networking disabled."""
    docker = shutil.which("docker")
    if docker is None:
        return ExecutionResult(
            exit_code=127,
            stdout="",
            stderr="docker sandbox requested but docker executable was not found",
            timed_out=False,
            runtime_s=0.0,
        )

    mount_dir = Path(work_dir).resolve()
    cmd = [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        f"{memory_mb}m",
        "--cpus",
        "1",
        "--pids-limit",
        "64",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=32m",
        "-v",
        f"{mount_dir.as_posix()}:/sandbox:ro",
        "-w",
        "/sandbox",
        settings.SANDBOX_DOCKER_IMAGE,
        "python",
        Path(script_path).name,
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=_minimal_env())
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
            stderr=f"{stderr}\n[docker sandbox killed: exceeded {timeout_s}s wall-clock timeout]",
            timed_out=True,
            runtime_s=round(time.monotonic() - started, 3),
        )


def run_code(
    code: str,
    *,
    timeout_s: int = 15,
    cpu_seconds: int = 10,
    memory_mb: int = 256,
) -> ExecutionResult:
    """Execute ``code`` in the configured sandbox and capture the result.

    Args:
        code: Python source to run.
        timeout_s: Wall-clock timeout; the process/container is killed past it.
        cpu_seconds: RLIMIT_CPU for subprocess mode on non-Windows platforms.
        memory_mb: Memory limit in MiB.

    Returns:
        An ``ExecutionResult`` (never raises for normal failures/timeouts).
    """
    with tempfile.TemporaryDirectory(prefix="sandbox_") as work_dir:
        script_path = os.path.join(work_dir, "snippet.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(code)

        if settings.SANDBOX_BACKEND == "docker":
            return _run_docker(script_path, work_dir, timeout_s=timeout_s, memory_mb=memory_mb)
        return _run_subprocess(
            script_path,
            work_dir,
            timeout_s=timeout_s,
            cpu_seconds=cpu_seconds,
            memory_mb=memory_mb,
        )
