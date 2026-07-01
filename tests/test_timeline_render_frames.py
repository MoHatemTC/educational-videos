"""Tests for timeline-driven frame rendering."""

from pathlib import Path

from PIL import Image

from app.services.pipeline.render.frames import BG, HIGHLIGHT_BG, render_frames


def _pixel(path: Path, xy: tuple[int, int]) -> tuple[int, int, int]:
    """Return an RGB pixel from a rendered frame."""
    with Image.open(path) as image:
        return image.convert("RGB").getpixel(xy)


def test_timeline_highlight_renders_only_in_expected_frame_window(tmp_path: Path) -> None:
    """Highlight events should affect only frames inside their reconciled window."""
    code = "first_line\nsecond_line\nthird_line"
    timeline = {
        "events": [
            {"event_type": "type", "start_ms": 0, "end_ms": 5000, "code": code},
            {"event_type": "highlight", "start_ms": 2000, "end_ms": 3000, "start_line": 2, "end_line": 2},
        ]
    }

    frame_count = render_frames(code, "Timeline test", tmp_path, fps=2, duration_s=5.0, timeline=timeline)

    assert frame_count == 10
    # Frame numbers are one-based on disk. At 2 FPS, frame 4 is t=1.5s,
    # frame 5 is t=2.0s, frame 6 is t=2.5s, and frame 7 is t=3.0s.
    sample = (82, 151)
    assert _pixel(tmp_path / "frame_00004.png", sample) == BG
    assert _pixel(tmp_path / "frame_00005.png", sample) == HIGHLIGHT_BG
    assert _pixel(tmp_path / "frame_00006.png", sample) == HIGHLIGHT_BG
    assert _pixel(tmp_path / "frame_00007.png", sample) == BG


def test_invalid_timeline_falls_back_to_duration_based_render(tmp_path: Path) -> None:
    """Invalid timelines should not block rendering older duration-based videos."""
    code = "print('fallback')"
    timeline = {"events": []}

    frame_count = render_frames(code, "Fallback test", tmp_path, fps=2, duration_s=2.0, timeline=timeline)

    assert frame_count == 4
    assert (tmp_path / "frame_00001.png").exists()
    assert (tmp_path / "frame_00004.png").exists()
