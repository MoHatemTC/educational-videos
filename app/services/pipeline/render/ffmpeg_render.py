"""Assemble PNG frames + narration audio into a final MP4 via FFmpeg."""

import subprocess
from pathlib import Path
from typing import Any, cast

from app.core.logging import logger

_FFMPEG_TIMEOUT_S = 600


def _ffmpeg_executable() -> str:
    """Return a usable FFmpeg executable path."""
    try:
        module = cast(Any, __import__("imageio_ffmpeg", fromlist=["get_ffmpeg_exe"]))
        return str(module.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        logger.warning("bundled_ffmpeg_unavailable", error=str(exc))
        return "ffmpeg"


def assemble_video(frames_dir: str | Path, fps: int, audio_path: str | Path, output_path: str | Path) -> Path:
    """Mux ``frame_%05d.png`` frames with narration into an H.264/AAC MP4.

    Args:
        frames_dir: Directory containing ``frame_00001.png`` ... frames.
        fps: Frame rate the frames were rendered at.
        audio_path: Narration audio file.
        output_path: Destination ``.mp4`` path.

    Returns:
        The output path.

    Raises:
        RuntimeError: If FFmpeg exits non-zero.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _ffmpeg_executable(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        f"{frames_dir}/frame_%05d.png",
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        logger.error("ffmpeg_assemble_timeout", timeout_s=_FFMPEG_TIMEOUT_S)
        raise RuntimeError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s") from exc
    if result.returncode != 0:
        logger.error("ffmpeg_assemble_failed", stderr=result.stderr[-1200:])
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-400:]}")

    logger.info("video_assembled", output=str(output_path), size_bytes=output_path.stat().st_size)
    return output_path
