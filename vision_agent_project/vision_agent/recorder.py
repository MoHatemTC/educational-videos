"""vision_agent.recorder.

Output-persistence layer for the vision agent.

Responsibilities
----------------
  * Save per-step PNG screenshots to ``screenshots/``
  * Compile the collected frames into an animated GIF at ``run.gif``
  * Optionally save the final text result to a Markdown file

The Recorder is intentionally decoupled from the browser and VLM – it only
deals with bytes and paths.

Usage
-----
    recorder = Recorder(output_dir=Path("output"))
    recorder.save_screenshot(step=1, png_bytes=raw_bytes)
    ...
    recorder.save_gif(fps=1)
    recorder.save_report(result_text)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    List,
    Optional,
)

logger = logging.getLogger(__name__)


class Recorder:
    """Collects PNG frames and converts them to an animated GIF.

    Parameters
    ----------
    output_dir:
        Root output directory.  Sub-directories ``screenshots/`` and ``video/``
        are created automatically.
    gif_path:
        Destination for the animated GIF.  Defaults to ``output_dir/run.gif``.
    """

    def __init__(
        self,
        output_dir: Path = Path("output"),
        gif_path: Optional[Path] = None,
    ) -> None:
        """Initialize the browser controller."""
        self.output_dir = Path(output_dir)
        self.screenshots_dir = self.output_dir / "screenshots"
        self.video_dir = self.output_dir / "video"
        self.gif_path = gif_path or (self.output_dir / "run.gif")

        # In-memory ordered list of (step, path) tuples
        self._frames: List[Path] = []

        # Create directories up-front
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Screenshot persistence
    # ------------------------------------------------------------------ #

    def save_screenshot(self, step: int, png_bytes: bytes) -> Path:
        """Write a PNG frame to disk and register it for GIF compilation.

        Parameters
        ----------
        step:
            1-based step index used to name the file (``step_01.png``).
        png_bytes:
            Raw PNG bytes from the browser.

        Returns:
        -------
        Path
            Absolute path to the saved file.
        """
        filename = self.screenshots_dir / f"step_{step:03d}.png"
        filename.write_bytes(png_bytes)
        self._frames.append(filename)
        logger.debug("Screenshot saved → %s", filename)
        return filename

    # ------------------------------------------------------------------ #
    # GIF compilation
    # ------------------------------------------------------------------ #

    def save_gif(self, fps: float = 1.0, max_size: tuple[int, int] = (1280, 900)) -> Optional[Path]:
        """Compile collected screenshots into an animated GIF using Pillow.

        The frames are resized (aspect-ratio-preserving) to ``max_size`` to
        keep the output file manageable.

        Parameters
        ----------
        fps:
            Frames per second.  Each frame's display duration = 1000/fps ms.
        max_size:
            Maximum (width, height) for each frame.

        Returns:
        -------
        Path | None
            Path to the created GIF, or None if fewer than 2 frames are available.
        """
        if len(self._frames) < 1:
            logger.warning("No frames recorded – skipping GIF generation.")
            return None

        try:
            from PIL import Image  # type: ignore
        except ImportError:
            logger.warning("Pillow not installed – skipping GIF generation. Install with: pip install Pillow")
            return None

        duration_ms = int(1000 / fps)
        pil_frames: list = []

        for path in self._frames:
            try:
                img = Image.open(path).convert("RGBA")
                img.thumbnail(max_size, Image.LANCZOS)
                # GIF requires palette mode
                pil_frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=256))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping frame %s: %s", path, exc)

        if not pil_frames:
            logger.warning("All frames failed to load – skipping GIF.")
            return None

        self.gif_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            self.gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )
        logger.info("GIF saved → %s  (%d frames @ %.1f fps)", self.gif_path, len(pil_frames), fps)
        return self.gif_path

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #

    def save_report(self, result: str, filename: str = "report.md") -> Path:
        """Write the agent's final result text to a Markdown file."""
        path = self.output_dir / filename
        path.write_text(result, encoding="utf-8")
        logger.info("Report saved → %s", path)
        return path

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @property
    def frame_count(self) -> int:
        """Return the number of recorded frames."""
        return len(self._frames)

    def clear_frames(self) -> None:
        """Reset the in-memory frame list (does not delete files on disk)."""
        self._frames.clear()
