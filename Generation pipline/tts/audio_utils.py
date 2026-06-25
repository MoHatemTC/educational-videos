"""tts/audio_utils.py — Measure real audio clip duration using multiple backends.

Priority:
  1. ffprobe  (most accurate, handles any container)
  2. mutagen  (fast, pure-Python, wide format support)
  3. pydub    (fallback decoder)
  4. Raw MP3 frame counting (last resort, no external deps)

All backends return duration in seconds as a float.
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def get_audio_duration(path: Path | str) -> float:
    """Return the real playback duration of an audio file in seconds.

    Tries multiple backends in order of accuracy and raises RuntimeError
    if none succeed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    for backend in (_duration_ffprobe, _duration_mutagen, _duration_pydub, _duration_mp3_frames):
        try:
            result = backend(path)
            if result is not None and result > 0:
                logger.debug("Audio duration via %s: %.3fs", backend.__name__, result)
                return round(result, 4)
        except Exception as exc:
            logger.debug("%s failed: %s", backend.__name__, exc)

    raise RuntimeError(f"Could not determine audio duration for: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Backend implementations
# ─────────────────────────────────────────────────────────────────────────────


def _duration_ffprobe(path: Path) -> Optional[float]:
    """Use ffprobe JSON output — most accurate."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        return None
    data = json.loads(proc.stdout)
    for stream in data.get("streams", []):
        dur = stream.get("duration")
        if dur:
            return float(dur)
    # fallback: format-level duration
    fmt = data.get("format", {})
    dur = fmt.get("duration")
    return float(dur) if dur else None


def _duration_mutagen(path: Path) -> Optional[float]:
    """Use mutagen — pure Python, wide format support."""
    from mutagen import File as MutaFile  # type: ignore

    audio = MutaFile(str(path))
    if audio is None:
        return None
    info = getattr(audio, "info", None)
    if info is None:
        return None
    return float(info.length)


def _duration_pydub(path: Path) -> Optional[float]:
    """Use pydub — handles MP3/WAV/OGG."""
    from pydub import AudioSegment  # type: ignore

    seg = AudioSegment.from_file(str(path))
    return len(seg) / 1000.0


def _duration_mp3_frames(path: Path) -> Optional[float]:
    """Estimate MP3 duration by scanning frame headers.

    Works without any external tools but may be approximate for VBR.
    """
    data = path.read_bytes()

    # Skip ID3v2 tag
    offset = 0
    if data[:3] == b"ID3":
        # ID3v2 size is stored as 4 synchsafe bytes at offset 6
        sz = (data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14 | (data[8] & 0x7F) << 7 | (data[9] & 0x7F)
        offset = 10 + sz

    total_duration = 0.0
    frames_found = 0
    i = offset

    while i < len(data) - 4:
        # Sync word: 0xFFE0 or better 0xFFFA/0xFFFB
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            header = struct.unpack(">I", data[i : i + 4])[0]
            version = (header >> 19) & 0x03
            layer = (header >> 17) & 0x03
            bitrate_idx = (header >> 12) & 0x0F
            samplerate_idx = (header >> 10) & 0x03
            padding = (header >> 9) & 0x01

            # Only handle MPEG-1 Layer-3 for simplicity
            if version == 3 and layer == 1:
                bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
                samplerates = [44100, 48000, 32000, 0]
                br = bitrates[bitrate_idx] * 1000  # bits per second
                sr = samplerates[samplerate_idx]
                if br > 0 and sr > 0:
                    frame_size = (144 * br // sr) + padding
                    frame_duration = 1152 / sr  # samples per frame / sample rate
                    total_duration += frame_duration
                    frames_found += 1
                    i += max(frame_size, 1)
                    continue
        i += 1

    return total_duration if frames_found > 10 else None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers for timeline use
# ─────────────────────────────────────────────────────────────────────────────


def compute_stretch_factor(actual_duration: float, target_duration: float) -> float:
    """Return the multiplicative factor to stretch actual → target.

    Clamps to [0.5, 2.0] to avoid unrealistic stretch values.
    """
    if target_duration <= 0 or actual_duration <= 0:
        return 1.0
    factor = target_duration / actual_duration
    return max(0.5, min(2.0, factor))


def adjust_timestamps(timestamps: list[float], stretch_factor: float, offset: float = 0.0) -> list[float]:
    """Scale a list of timestamps by *stretch_factor* and shift by *offset* seconds."""
    return [round(t * stretch_factor + offset, 4) for t in timestamps]
