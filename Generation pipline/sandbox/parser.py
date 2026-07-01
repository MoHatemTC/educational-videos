"""sandbox/parser.py — Parse raw subprocess output into typed execution records.

Extracts traceback frames, error types, and line numbers so the
self-correction prompt builder has precise, structured failure data.
"""

from __future__ import annotations

import re
from datetime import (
    datetime,
    timezone,
)
from typing import (
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────


class TracebackFrame(BaseModel):
    """One frame extracted from a Python traceback."""

    filename: str
    lineno: int
    function: str
    source_line: Optional[str] = None


class ErrorRecord(BaseModel):
    """Structured representation of a single Python error."""

    error_type: str = Field(description="Exception class name, e.g. 'NameError'.")
    error_message: str = Field(description="The exception message string.")
    traceback_frames: List[TracebackFrame] = Field(default_factory=list)
    raw_traceback: str = Field(default="", description="Full raw traceback text.")
    first_failing_line: Optional[int] = Field(default=None, description="Line number of the deepest frame.")


class ExecutionResult(BaseModel):
    """Complete outcome of one sandbox run."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0
    errors: List[ErrorRecord] = Field(default_factory=list)
    code_snapshot: str = Field(default="", description="The code that was executed.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches the header of each traceback frame:
#   File "foo.py", line 42, in my_func
_FRAME_RE = re.compile(r'File "(?P<filename>[^"]+)", line (?P<lineno>\d+), in (?P<function>\S+)')

# Matches the final error line: ExceptionType: message
_ERROR_LINE_RE = re.compile(
    r"^(?P<etype>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*Error[^:]*|[A-Za-z_][A-Za-z0-9_]*(?:Exception|Warning))[:\s](?P<msg>.*)$"
)

# SyntaxError has a different format
_SYNTAX_ERROR_RE = re.compile(r"SyntaxError: (?P<msg>.+)")

_TIMEOUT_SENTINEL = "TimeoutExpired"


def _split_tracebacks(stderr: str) -> List[str]:
    """Split stderr into individual traceback blocks.

    Each block starts with 'Traceback (most recent call last):'
    """
    blocks: List[str] = []
    current: List[str] = []
    in_tb = False

    for line in stderr.splitlines():
        if line.startswith("Traceback (most recent call last):"):
            if current:
                blocks.append("\n".join(current))
            current = [line]
            in_tb = True
        elif in_tb:
            current.append(line)
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current))

    return blocks if blocks else [stderr]


def _parse_single_traceback(tb_text: str) -> Optional[ErrorRecord]:
    """Parse one traceback string into an :class:`ErrorRecord`."""
    frames: List[TracebackFrame] = []
    lines = tb_text.strip().splitlines()

    # Collect frames
    i = 0
    while i < len(lines):
        m = _FRAME_RE.search(lines[i])
        if m:
            source = lines[i + 1].strip() if i + 1 < len(lines) else None
            frames.append(
                TracebackFrame(
                    filename=m.group("filename"),
                    lineno=int(m.group("lineno")),
                    function=m.group("function"),
                    source_line=source,
                )
            )
            i += 2
            continue
        i += 1

    # Find the error type/message — last non-empty line
    error_type = "UnknownError"
    error_msg = tb_text.strip().splitlines()[-1] if tb_text.strip() else ""

    for line in reversed(lines):
        line = line.strip()
        em = _ERROR_LINE_RE.match(line)
        if em:
            error_type = em.group("etype")
            error_msg = em.group("msg").strip()
            break
        if line.startswith("SyntaxError:"):
            error_type = "SyntaxError"
            error_msg = line[len("SyntaxError:") :].strip()
            break

    first_failing_line = frames[-1].lineno if frames else None

    return ErrorRecord(
        error_type=error_type,
        error_message=error_msg,
        traceback_frames=frames,
        raw_traceback=tb_text,
        first_failing_line=first_failing_line,
    )


def _handle_timeout(stderr: str) -> ErrorRecord:
    return ErrorRecord(
        error_type="TimeoutExpired",
        error_message=stderr,
        traceback_frames=[],
        raw_traceback=stderr,
        first_failing_line=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def parse_execution_output(
    *,
    stdout: str,
    stderr: str,
    exit_code: int,
    duration_seconds: float,
    code: str,
) -> ExecutionResult:
    """Combine raw execution outputs into a typed :class:`ExecutionResult`.

    Parses every traceback block found in *stderr* into :class:`ErrorRecord`
    instances so the correction loop has precise, structured failure data.
    """
    errors: List[ErrorRecord] = []

    if exit_code != 0:
        if _TIMEOUT_SENTINEL in stderr:
            errors.append(_handle_timeout(stderr))
        else:
            blocks = _split_tracebacks(stderr)
            for block in blocks:
                if "Traceback" in block or "Error" in block or "Exception" in block:
                    record = _parse_single_traceback(block)
                    if record:
                        errors.append(record)

        # If we have stderr but couldn't parse a structured error, add a raw one
        if not errors and stderr.strip():
            errors.append(
                ErrorRecord(
                    error_type="RuntimeError",
                    error_message=stderr.strip()[:500],
                    raw_traceback=stderr,
                )
            )

    return ExecutionResult(
        success=exit_code == 0,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        errors=errors,
        code_snapshot=code,
    )
