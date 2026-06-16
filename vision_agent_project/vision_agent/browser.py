"""
vision_agent.browser
====================
Low-level Playwright / browser-use CDP wrapper.

Responsibilities
----------------
  * Launch / close a Chromium browser session (headed or headless)
  * Capture full-page screenshots as raw PNG bytes or base64 strings
  * Execute typed `Action` objects (click, type, scroll, navigate, wait)
  * Optionally record a video of the entire session

The class intentionally has *no* knowledge of VLM calls or the agent loop –
it is a pure browser I/O layer.

Usage
-----
    async with BrowserController() as bc:
        await bc.navigate("https://example.com")
        screenshot_b64 = await bc.screenshot_base64()
        action = Action(action_type=ActionType.CLICK, x=200, y=100)
        await bc.execute(action)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Optional

from vision_agent.actions import Action, ActionType

logger = logging.getLogger(__name__)


class BrowserController:
    """
    Async context-manager that owns a single Playwright Page.

    Parameters
    ----------
    headless:
        Run Chromium headless (no visible window).  Set False when an Xvfb
        virtual display is available (e.g. inside Docker).
    chrome_path:
        Explicit path to the Chromium / Chrome executable.  Leave None to let
        browser-use / Playwright resolve it automatically.
    window_width / window_height:
        Viewport size in pixels.
    video_dir:
        If set, Playwright will record a .webm video to this directory.
    slow_mo_ms:
        Milliseconds of delay between Playwright actions (useful for debugging).
    """

    def __init__(
        self,
        *,
        headless: bool = False,
        chrome_path: Optional[str] = None,
        window_width: int = 1280,
        window_height: int = 900,
        video_dir: Optional[Path] = None,
        slow_mo_ms: int = 0,
    ) -> None:
        self.headless = headless
        self.chrome_path = chrome_path
        self.window_width = window_width
        self.window_height = window_height
        self.video_dir = video_dir
        self.slow_mo_ms = slow_mo_ms

        # Populated during __aenter__
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    # ------------------------------------------------------------------ #
    # Context-manager lifecycle
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "BrowserController":
        await self._launch()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def _launch(self) -> None:
        """Start Playwright and open the first page."""
        # Import here so the rest of the module can be imported without
        # Playwright installed (e.g. during unit-test collection).
        from playwright.async_api import async_playwright  # type: ignore

        logger.info(
            "Launching Chromium (headless=%s, %dx%d)",
            self.headless,
            self.window_width,
            self.window_height,
        )

        self._playwright = await async_playwright().start()

        launch_opts: dict = {
            "headless": self.headless,
            "slow_mo": self.slow_mo_ms,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                f"--window-size={self.window_width},{self.window_height}",
            ],
        }
        if self.chrome_path:
            launch_opts["executable_path"] = self.chrome_path

        self._browser = await self._playwright.chromium.launch(**launch_opts)

        # Context holds viewport + optional video recorder
        context_opts: dict = {
            "viewport": {"width": self.window_width, "height": self.window_height},
        }
        if self.video_dir:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            context_opts["record_video_dir"] = str(self.video_dir)
            context_opts["record_video_size"] = {
                "width": self.window_width,
                "height": self.window_height,
            }
            logger.info("Video recording → %s", self.video_dir)

        self._context = await self._browser.new_context(**context_opts)
        self._page = await self._context.new_page()
        logger.info("Browser ready.")

    async def close(self) -> None:
        """Gracefully stop recording (if active) and close the browser."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during browser close: %s", exc)
        finally:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
        logger.info("Browser closed.")

    # ------------------------------------------------------------------ #
    # Screenshot (Observe)
    # ------------------------------------------------------------------ #

    async def screenshot_bytes(self) -> bytes:
        """Return a full-page PNG screenshot as raw bytes."""
        self._require_page()
        data = await self._page.screenshot(full_page=True, type="png")
        logger.debug("Screenshot captured (%d bytes)", len(data))
        return data

    async def screenshot_base64(self) -> str:
        """Return a full-page PNG screenshot as a base64-encoded string (no data-URI prefix)."""
        raw = await self.screenshot_bytes()
        return base64.b64encode(raw).decode("ascii")

    async def current_url(self) -> str:
        """Return the current page URL."""
        self._require_page()
        return self._page.url

    # ------------------------------------------------------------------ #
    # Action execution (Act)
    # ------------------------------------------------------------------ #

    async def execute(self, action: Action) -> None:
        """
        Dispatch a typed Action to the appropriate Playwright call.

        Raises
        ------
        ValueError
            For ActionType.DONE – the caller (agent loop) should handle that.
        RuntimeError
            If the page is not open.
        """
        self._require_page()
        atype = action.action_type

        logger.debug("Executing %r", action)

        if atype == ActionType.CLICK:
            await self._click(action.x, action.y)

        elif atype == ActionType.TYPE:
            await self._type(action.text or "")

        elif atype == ActionType.SCROLL:
            await self._scroll(action.x or 0, action.y or 0, action.delta_y)

        elif atype == ActionType.NAVIGATE:
            await self._navigate(action.url or "about:blank")

        elif atype == ActionType.WAIT:
            await asyncio.sleep(action.seconds)
            logger.debug("Waited %.1fs", action.seconds)

        elif atype == ActionType.DONE:
            raise ValueError(
                "ActionType.DONE must be handled by the agent loop, not executed."
            )

        else:
            raise ValueError(f"Unrecognised ActionType: {atype!r}")

    # ------------------------------------------------------------------ #
    # Private action implementations
    # ------------------------------------------------------------------ #

    async def _click(self, x: Optional[int], y: Optional[int]) -> None:
        if x is None or y is None:
            raise ValueError("click action requires x and y coordinates.")
        await self._page.mouse.click(x, y)
        logger.debug("Clicked (%d, %d)", x, y)
        # Brief pause so the page can react before the next screenshot
        await asyncio.sleep(0.4)

    async def _type(self, text: str) -> None:
        await self._page.keyboard.type(text, delay=30)
        logger.debug("Typed %d chars", len(text))

    async def _scroll(self, x: int, y: int, delta_y: int) -> None:
        await self._page.mouse.wheel(delta_x=0, delta_y=delta_y)
        logger.debug("Scrolled at (%d,%d) by %d", x, y, delta_y)
        await asyncio.sleep(0.3)

    async def navigate(self, url: str) -> None:
        """Public shortcut for navigating – used by run.py on startup."""
        await self._navigate(url)

    async def _navigate(self, url: str) -> None:
        await self._page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        logger.info("Navigated → %s", url)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _require_page(self) -> None:
        if self._page is None:
            raise RuntimeError(
                "BrowserController is not open. "
                "Use it as an async context manager: `async with BrowserController() as bc:`"
            )
