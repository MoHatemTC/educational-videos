"""Render code-typing animation frames with Pillow.

Produces ``frame_NNNNN.png`` files: a titled, dark-themed editor that types the
code character-by-character over the first ``type_fraction`` of the clip, then
holds the full code. Frame count is derived from the narration duration so the
animation length matches the audio.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.core.logging import logger

WIDTH, HEIGHT = 1280, 720
BG = (30, 30, 46)
TITLE_BG = (49, 50, 68)
FG = (205, 214, 244)
ACCENT = (137, 180, 250)
DIM = (108, 112, 134)

_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to Pillow's default."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def render_frames(
    code: str,
    title: str,
    frames_dir: str | Path,
    fps: int = 10,
    duration_s: float = 10.0,
    type_fraction: float = 0.85,
) -> int:
    """Render the typing animation to ``frames_dir`` and return the frame count.

    Args:
        code: Source code to type.
        title: Title shown in the header bar.
        frames_dir: Output directory for PNG frames.
        fps: Frames per second.
        duration_s: Total animation duration (matches narration length).
        type_fraction: Fraction of the clip spent typing before holding.

    Returns:
        Number of frames written.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    code = code.rstrip() or "# (no code)"
    lines = code.split("\n")
    total_chars = len(code)
    total_frames = max(1, int(round(duration_s * fps)))
    type_frames = max(1, int(total_frames * type_fraction))

    code_font = _font(_MONO, 26)
    title_font = _font(_BOLD, 34)
    ln_font = _font(_MONO, 20)
    line_height = 36
    x_code = 90
    y_top = 110

    for i in range(total_frames):
        reveal = total_chars if i >= type_frames else int(total_chars * (i / type_frames))
        shown_lines = code[:reveal].split("\n")

        img = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, WIDTH, 70], fill=TITLE_BG)
        draw.text((30, 16), title[:60], font=title_font, fill=ACCENT)

        y = y_top
        for idx, _ in enumerate(lines):
            draw.text((30, y), str(idx + 1).rjust(2), font=ln_font, fill=DIM)
            if idx < len(shown_lines):
                text = shown_lines[idx]
                draw.text((x_code, y), text, font=code_font, fill=FG)
                is_cursor_line = idx == len(shown_lines) - 1 and reveal < total_chars
                if is_cursor_line:
                    w = draw.textlength(text, font=code_font)
                    draw.rectangle([x_code + w + 2, y + 2, x_code + w + 13, y + 30], fill=ACCENT)
            y += line_height
            if y > HEIGHT - 40:
                break

        img.save(frames_dir / f"frame_{i + 1:05d}.png")

    logger.info("frames_rendered", count=total_frames, fps=fps, duration_s=round(duration_s, 2))
    return total_frames
