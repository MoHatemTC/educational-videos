r"""vision_agent.agent.

Core Observe → Reason → Plan → Act loop.

Architecture
------------

    ┌────────────┐  screenshot  ┌──────────────┐  base64+prompt  ┌──────────────────┐
    │  Browser   │ ──────────▶  │   VisionAgent │ ──────────────▶ │  Google Gemini   │
    │Controller  │              │   (this file) │ ◀────────────── │  API             │
    └────────────┘  execute()   └──────────────┘  action JSON     └──────────────────┘

Loop stages
-----------
  1. **Observe**  – capture a full-page screenshot via BrowserController
  2. **Reason**   – POST the encoded frame + goal + history to Gemini Vision
  3. **Plan**     – parse the VLM's JSON action payload via Action.from_vlm_text
  4. **Act**      – execute the action (or break if ActionType.DONE)

VLM system prompt
-----------------
The model is instructed to respond with ONLY a JSON block:

    {"action": "click",    "x": 412, "y": 308}
    {"action": "type",     "text": "Computer vision\\n"}
    {"action": "scroll",   "x": 760, "y": 400, "delta_y": 300}
    {"action": "navigate", "url": "https://..."}
    {"action": "wait",     "seconds": 2}
    {"action": "done",     "result": "First paragraph text ..."}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from vision_agent.actions import (
    Action,
    ActionType,
)
from vision_agent.browser import BrowserController
from vision_agent.recorder import Recorder

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini REST endpoint — model and key are substituted at call time
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

# ------------------------------------------------------------------ #
# VLM system prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are a vision-based browser automation agent.
You receive a screenshot of the current browser state and a goal to achieve.
Your job is to decide the single next action to take.

ALWAYS respond with ONLY a valid JSON object (no markdown, no explanation):

For clicking:    {"action": "click",    "x": <int>, "y": <int>}
For typing:      {"action": "type",     "text": "<string>"}
For scrolling:   {"action": "scroll",   "x": <int>, "y": <int>, "delta_y": <int>}
For navigating:  {"action": "navigate", "url": "<url>"}
For waiting:     {"action": "wait",     "seconds": <float>}
For finishing:   {"action": "done",     "result": "<final answer text>"}

Rules:
- Coordinates are pixel positions on the screenshot (x=column, y=row).
- Use "done" only when the goal is fully accomplished.
- "result" in the done action must contain the requested information.
- Never include any text outside the JSON object.
- If you are unsure, scroll down to reveal more content.
"""


class VisionAgent:
    """Vision-based UI agent that drives a browser without CSS selectors or XPath.

    Uses the Google Gemini Vision API directly.

    Parameters
    ----------
    task:
        Natural-language goal for the agent.
    google_api_key:
        Google AI Studio / Gemini API key.
    model:
        Gemini model name (e.g. ``gemini-2.5-flash``, ``gemini-2.5-pro``).
    max_steps:
        Maximum Observe-Act iterations before the agent gives up.
    output_dir:
        Root directory for screenshots, GIF, and report.
    headless:
        Run Chromium headless.
    chrome_path:
        Optional path to a Chromium / Chrome executable.
    window_width / window_height:
        Browser viewport dimensions.
    record_video:
        Enable Playwright video recording.
    gif_fps:
        Frames-per-second for the output GIF.
    """

    def __init__(
        self,
        task: str,
        *,
        google_api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_steps: int = 30,
        output_dir: Path = Path("output"),
        headless: bool = False,
        chrome_path: Optional[str] = None,
        window_width: int = 1280,
        window_height: int = 900,
        record_video: bool = True,
        gif_fps: float = 1.0,
    ) -> None:
        """Initialize the browser controller."""
        self.task = task

        # Gemini config — env vars are the fallback
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY must be set (either as a constructor argument or environment variable).")

        self.max_steps = max_steps
        self.output_dir = Path(output_dir)
        self.gif_fps = gif_fps

        # Sub-components
        self.browser = BrowserController(
            headless=headless,
            chrome_path=chrome_path,
            window_width=window_width,
            window_height=window_height,
            video_dir=(self.output_dir / "video") if record_video else None,
        )
        self.recorder = Recorder(
            output_dir=self.output_dir,
            gif_path=self.output_dir / "run.gif",
        )

        # Accumulated text history shown to the VLM on each turn
        self._history: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self) -> str:
        """Open the browser and run the Observe → Reason → Plan → Act loop.

        Returns:
        -------
        str
            The agent's final result string (from a ``done`` action) or a
            fallback message if max_steps was reached.
        """
        result = "(Agent did not produce a result within the step limit.)"

        async with self.browser:
            logger.info("Agent starting. Task: %s", self.task)

            for step in range(1, self.max_steps + 1):
                logger.info("━━━ Step %d / %d ━━━", step, self.max_steps)

                # ── 1. OBSERVE ──────────────────────────────────────────
                png_bytes = await self._observe()
                self.recorder.save_screenshot(step=step, png_bytes=png_bytes)

                # ── 2. REASON ───────────────────────────────────────────
                vlm_text = await self._reason(png_bytes)

                # ── 3. PLAN ─────────────────────────────────────────────
                try:
                    action = self._plan(vlm_text)
                except ValueError as exc:
                    logger.error("Plan stage failed: %s", exc)
                    action = Action(action_type=ActionType.SCROLL, delta_y=400)

                logger.info("Planned action: %r", action)

                # ── 4. ACT ──────────────────────────────────────────────
                if action.is_terminal():
                    result = action.result or result
                    logger.info("Agent completed task. Result: %s", result[:200])
                    break

                await self._act(action, step)

            else:
                logger.warning("Max steps (%d) reached without completion.", self.max_steps)

        # ── OUTPUT PERSISTENCE ──────────────────────────────────────────
        logger.info("Saving outputs to %s ...", self.output_dir)
        self.recorder.save_report(result)
        gif = self.recorder.save_gif(fps=self.gif_fps)
        if gif:
            logger.info("GIF → %s", gif)

        return result

    # ------------------------------------------------------------------ #
    # Stage implementations
    # ------------------------------------------------------------------ #

    async def _observe(self) -> bytes:
        """Stage 1 – Capture a full-page screenshot."""
        start = time.perf_counter()
        png_bytes = await self.browser.screenshot_bytes()
        elapsed = time.perf_counter() - start
        url = await self.browser.current_url()
        logger.debug("Observe: %d bytes, url=%s  (%.2fs)", len(png_bytes), url, elapsed)
        return png_bytes

    async def _reason(self, png_bytes: bytes) -> str:
        """Stage 2 – Send the screenshot + task to the Gemini Vision API.

        Returns the raw text response from the model.
        """
        b64 = base64.b64encode(png_bytes).decode("ascii")

        history_summary = f"Steps completed so far: {len(self._history)}\n" + (
            "\n".join(f"  - {h}" for h in self._history[-5:]) if self._history else "  (none)"
        )

        user_text = (
            f"Goal: {self.task}\n\n"
            f"{history_summary}\n\n"
            "The current browser screenshot is attached. "
            "What is the single best next action?"
        )

        # Gemini multimodal request body
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 512,
            },
        }

        start = time.perf_counter()
        response_text = await self._call_gemini(payload)
        elapsed = time.perf_counter() - start

        logger.debug("Reason: Gemini responded in %.2fs  →  %s", elapsed, response_text[:120])

        # Keep a short text summary of what was decided (no images in history to save tokens)
        self._history.append(response_text[:200])

        return response_text

    def _plan(self, vlm_text: str) -> Action:
        """Stage 3 – Parse the VLM text into a typed Action."""
        return Action.from_vlm_text(vlm_text)

    async def _act(self, action: Action, step: int) -> None:
        """Stage 4 – Execute the action through BrowserController."""
        start = time.perf_counter()
        await self.browser.execute(action)
        elapsed = time.perf_counter() - start
        logger.debug("Act: %r executed in %.2fs", action, elapsed)

    # ------------------------------------------------------------------ #
    # Gemini HTTP call
    # ------------------------------------------------------------------ #

    async def _call_gemini(self, payload: dict) -> str:
        """POST to the Gemini generateContent REST endpoint.

        Uses httpx.AsyncClient so the entire agent stays fully async.
        Raises RuntimeError on non-200 responses.
        """
        url = GEMINI_API_URL.format(model=self.model, api_key=self.google_api_key)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            raise RuntimeError(f"Gemini API request failed: HTTP {response.status_code}\n{response.text[:500]}")

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {json.dumps(data)[:400]}") from exc
