"""Real audio-duration measurement (never estimated from text length)."""

import json
import subprocess
from pathlib import Path

from app.core.logging import logger


def duration_seconds(audio_path: str | Path) -> float:
    """Return the real playback duration of an audio file in seconds.

    Uses ffprobe (always present in the image). Falls back to mutagen if ffprobe
    is unavailable. Raises on total failure so the render does not proceed with a
    bogus duration.
    """
    audio_path = str(audio_path)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception as exc:  # noqa: BLE001 - fall back to mutagen
        logger.warning("ffprobe_duration_failed", error=str(exc))

    from mutagen import File as MutagenFile  # local import: optional fallback path

    audio = MutagenFile(audio_path)
    if audio is None or not getattr(audio, "info", None):
        raise RuntimeError(f"could not measure audio duration for {audio_path}")
    return float(audio.info.length)
