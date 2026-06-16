#!/usr/bin/env python3
"""run.py — CLI entry point for the Vision-Based UI Agent (Gemini backend).

Usage
-----
    python run.py [TASK]
    TASK_PROMPT="List top HN headlines" python run.py
    DEFAULT_MODEL=gemini-2.5-pro python run.py "Search Wikipedia for Playwright"

Environment variables
---------------------
  GOOGLE_API_KEY   – Required. Google AI Studio / Gemini API key.
  DEFAULT_MODEL    – Gemini model string (default: gemini-2.5-flash).
  TASK_PROMPT      – Task when no CLI argument is given.
  MAX_STEPS        – Max Observe→Act iterations (default: 30).
  HEADLESS         – true/false (default: false, Xvfb handles display in Docker).
  RECORD_VIDEO     – true/false (default: true).
  OUTPUT_DIR       – Root output directory (default: output).
  WINDOW_SIZE      – e.g. 1280x900 (default: 1280x900).
  CHROME_PATH      – Explicit path to Chromium executable.
  GIF_FPS          – Frames per second for run.gif (default: 1).
  LOG_LEVEL        – DEBUG / INFO / WARNING (default: INFO).
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _setup_logging(level_str: str = "INFO") -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "playwright", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _bool_env(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _int_env(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float_env(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


DEFAULT_TASK = (
    "Go to https://en.wikipedia.org, search for 'Computer vision', open the "
    "article, and report its first paragraph as Markdown."
)


def get_task() -> str:
    """Return the task from command-line arguments."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return os.getenv("TASK_PROMPT", DEFAULT_TASK).strip()


def get_window_size() -> tuple[int, int]:
    """Return browser viewport size from WINDOW_SIZE."""
    raw = os.getenv("WINDOW_SIZE", "1280x900").lower()
    if "x" in raw:
        parts = raw.split("x", 1)
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 1280, 900


async def main() -> int:
    """Run the vision agent application."""
    _setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    log = logging.getLogger("run")

    from vision_agent.agent import VisionAgent

    # Validate required key early for a clear error message
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        log.error("GOOGLE_API_KEY is not set. Copy .env.example → .env and add your key.")
        return 1

    task = get_task()
    width, height = get_window_size()
    output_dir = Path(os.getenv("OUTPUT_DIR", "output"))

    log.info("═" * 60)
    log.info("Vision Agent — Gemini backend")
    log.info("═" * 60)
    log.info("Task        : %s", task)
    log.info("Model       : %s", os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"))
    log.info("Output dir  : %s", output_dir.resolve())
    log.info("Max steps   : %d", _int_env("MAX_STEPS", 30))
    log.info("Headless    : %s", _bool_env("HEADLESS", False))
    log.info("Record video: %s", _bool_env("RECORD_VIDEO", True))
    log.info("Window size : %dx%d", width, height)
    log.info("═" * 60)

    try:
        agent = VisionAgent(
            task=task,
            google_api_key=api_key,
            model=os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"),
            max_steps=_int_env("MAX_STEPS", 30),
            output_dir=output_dir,
            headless=_bool_env("HEADLESS", False),
            chrome_path=os.getenv("CHROME_PATH") or None,
            window_width=width,
            window_height=height,
            record_video=_bool_env("RECORD_VIDEO", True),
            gif_fps=_float_env("GIF_FPS", 1.0),
        )
        result = await agent.run()

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("Agent failed: %s", exc)
        return 1

    log.info("═" * 60)
    log.info("RESULT:")
    log.info("%s", result)
    log.info("═" * 60)
    log.info("Screenshots : %s/screenshots/", output_dir)
    log.info("GIF         : %s/run.gif", output_dir)
    log.info("Report      : %s/report.md", output_dir)
    log.info("Video       : %s/video/", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
