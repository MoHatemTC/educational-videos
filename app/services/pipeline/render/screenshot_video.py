"""Render web-page screenshots as a Ken-Burns (vertical scroll) video.

Each screenshot is scaled to 1280 wide and slowly panned top->bottom over its
slice of the narration, so a tall full-page capture reads like scrolling the
page. Multiple screenshots are concatenated. Narration audio drives the length.
"""

import subprocess
from pathlib import Path

from app.core.logging import logger
from app.services.pipeline.tts.audio import duration_seconds

WIDTH, HEIGHT, FPS = 1280, 720, 10
_BG = "0x1e1e2e"
_FFMPEG_TIMEOUT_S = 600


def _split_durations(total_s: float, n: int) -> list[float]:
    """Split a total duration into n roughly-equal slices (>= 1s each)."""
    each = max(1.0, total_s / n)
    return [each] * n


def render_screenshot_video(
    screenshots: list[str],
    narration_audio: str,
    out_path: str,
    duration_s: float = 0.0,
) -> str:
    """Render a vertical-scroll video over screenshots, muxed with narration.

    Args:
        screenshots: Screenshot paths, shown in order.
        narration_audio: Narration audio (the real length is authoritative).
        out_path: Destination ``.mp4`` path.
        duration_s: Optional hint; clamped to the real audio length.

    Returns:
        The output path.

    Raises:
        ValueError: If no screenshots are given.
        RuntimeError: If FFmpeg exits non-zero.
    """
    if not screenshots:
        raise ValueError("render_screenshot_video requires at least one screenshot")

    audio_len = duration_seconds(narration_audio)
    total_s = audio_len if duration_s <= 0 else min(duration_s, audio_len)
    total_s = max(total_s, 3.0)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    per_clip = _split_durations(total_s, len(screenshots))
    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    for idx, (shot, dur) in enumerate(zip(screenshots, per_clip, strict=True)):
        inputs += ["-loop", "1", "-t", f"{dur:.3f}", "-i", str(shot)]
        # scale to width, pad to >=720 tall, then crop a 720-tall window panning
        # from the top to the bottom over the clip (static for short pages).
        chains.append(
            f"[{idx}:v]scale={WIDTH}:-2,"
            rf"pad={WIDTH}:'max(ih\,{HEIGHT})':0:0:color={_BG},"
            rf"crop={WIDTH}:{HEIGHT}:0:'min((ih-{HEIGHT})*t/{dur:.3f}\,ih-{HEIGHT})',"
            f"fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v{idx}]"
        )
        labels.append(f"[v{idx}]")

    audio_idx = len(screenshots)
    filter_complex = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(screenshots)}:v=1:a=0[v]"

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-i",
        str(narration_audio),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        f"{audio_idx}:a",
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
        str(out),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        logger.error("screenshot_video_timeout", timeout_s=_FFMPEG_TIMEOUT_S)
        raise RuntimeError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s") from exc
    if result.returncode != 0:
        logger.error("screenshot_video_failed", stderr=result.stderr[-1500:])
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-400:]}")

    logger.info("screenshot_video_rendered", output=str(out), size_bytes=out.stat().st_size)
    return str(out)
