"""Parse a Python traceback from stderr into structured fields.

Handles runtime tracebacks (``Traceback (most recent call last): ...``) and the
distinct ``SyntaxError`` shape (no Traceback header, caret line). Only the parsed
fields are fed back into the repair prompt — never raw stdout — to limit the
prompt-injection surface.
"""

import re

_FILE_LINE_RE = re.compile(r'File "([^"]+)", line (\d+)')
_EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt)):?(?:\s+(?P<msg>.*))?$")


def parse_traceback(stderr: str) -> dict | None:
    """Extract structured error info from stderr, or None if there is no error.

    Returns:
        ``{exception_type, exception_message, file, line, innermost_frame}`` or
        ``None`` when stderr contains no recognizable error.
    """
    if not stderr or not stderr.strip():
        return None

    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    # Exception type + message: the last line matching "SomeError: message".
    exc_type, exc_message = "Error", lines[-1].strip()
    for line in reversed(lines):
        match = _EXC_RE.match(line.strip())
        if match:
            exc_type = match.group("type")
            exc_message = (match.group("msg") or "").strip()
            break

    # File + line: last "File "...", line N" reference (innermost frame).
    file_ref, line_no = None, None
    for line in reversed(lines):
        fmatch = _FILE_LINE_RE.search(line)
        if fmatch:
            file_ref = fmatch.group(1)
            line_no = int(fmatch.group(2))
            break

    # Innermost code frame: the source line shown just after the File reference,
    # or the caret-annotated line for SyntaxError.
    innermost = None
    for idx, line in enumerate(lines):
        if _FILE_LINE_RE.search(line) and idx + 1 < len(lines):
            candidate = lines[idx + 1].strip()
            if not candidate.startswith("File "):
                innermost = candidate

    return {
        "exception_type": exc_type,
        "exception_message": exc_message,
        "file": file_ref,
        "line": line_no,
        "innermost_frame": innermost,
    }
