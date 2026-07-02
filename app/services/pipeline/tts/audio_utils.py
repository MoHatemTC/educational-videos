"""app/services/pipeline/tts/audio_utils.py — Timeline stretch/adjust helpers.

Only the two math utilities that timeline_sync.py needs live here.
Audio duration measurement lives in app/services/pipeline/tts/audio.py
(main's existing module — do not duplicate it).

#17 note: _duration_mp3_frames is also ported for tests that need it,
but production code uses audio.duration_seconds (ffprobe/mutagen backed).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import (
    List,
    Optional,
)

import structlog

logger = structlog.get_logger(__name__)

# ── Stretch / adjust ──────────────────────────────────────────────────────────

_MIN_STRETCH = 0.5
_MAX_STRETCH = 2.0


def compute_stretch_factor(
    actual_duration: float,
    target_duration: float,
) -> float:
    """Return the ratio to scale event timestamps so they fit target_duration.

    Clamped to [0.5, 2.0] to avoid extreme distortion. Returns 1.0 when
    either value is zero to avoid division errors.

    Args:
        actual_duration: Original total event span in seconds.
        target_duration: Measured audio duration to stretch into.

    Returns:
        Clamped stretch factor (float).
    """
    if actual_duration <= 0 or target_duration <= 0:
        return 1.0
    factor = target_duration / actual_duration
    return max(_MIN_STRETCH, min(_MAX_STRETCH, factor))


def adjust_timestamps(
    timestamps: List[float],
    stretch_factor: float,
    offset: float,
) -> List[float]:
    """Apply stretch_factor and additive offset to a list of timestamps.

    Args:
        timestamps: Original timestamps in seconds.
        stretch_factor: Multiplicative scale applied before the offset.
        offset: Additive shift in seconds (e.g. segment start position).

    Returns:
        New list of adjusted timestamps.
    """
    return [round(t * stretch_factor + offset, 4) for t in timestamps]


# ── MP3 frame duration (test utility) ────────────────────────────────────────


def _duration_mp3_frames(path: Path) -> Optional[float]:
    """Estimate duration by counting MPEG-1 Layer-3 frames.

    This is a lightweight fallback used only in tests that write minimal
    silent MP3 fixtures. Production code uses audio.duration_seconds
    (ffprobe/mutagen backed) which is more accurate.

    Returns None if the file contains no recognisable MPEG frames.
    """
    MPEG1_L3_BITRATES = {
        1: 32,
        2: 40,
        3: 48,
        4: 56,
        5: 64,
        6: 80,
        7: 96,
        8: 112,
        9: 128,
        10: 160,
        11: 192,
        12: 224,
        13: 256,
        14: 320,
    }
    SAMPLES_PER_FRAME = 1152
    SAMPLE_RATE = 44100

    data = path.read_bytes()
    frame_count = 0
    i = 0
    while i < len(data) - 4:
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            header = struct.unpack(">I", data[i : i + 4])[0]
            bitrate_idx = (header >> 12) & 0xF
            bitrate = MPEG1_L3_BITRATES.get(bitrate_idx)
            if bitrate:
                frame_size = 144 * bitrate * 1000 // SAMPLE_RATE + ((header >> 9) & 1)
                frame_count += 1
                i += frame_size
                continue
        i += 1

    if frame_count == 0:
        return None
    return (frame_count * SAMPLES_PER_FRAME) / SAMPLE_RATE
