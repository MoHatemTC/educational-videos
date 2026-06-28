"""Assemble PNG frames + narration audio into a final MP4 via FFmpeg."""

import subprocess
from pathlib import Path

from app.core.logging import logger


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
        "ffmpeg",
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg_assemble_failed", stderr=result.stderr[-1200:])
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-400:]}")

    logger.info("video_assembled", output=str(output_path), size_bytes=output_path.stat().st_size)
    return output_path
