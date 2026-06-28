"""Headless-Chromium page capture via Playwright.

Navigates to a URL and saves a full-page screenshot (plus optional viewport
shots for very tall pages). The orchestrator runs as a sync FastAPI background
task off the event loop, so ``capture_page`` wraps the async Playwright API with
``asyncio.run``.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

from app.core.logging import logger

_LAUNCH_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]


async def navigate_and_capture(
    url: str,
    out_dir: Path,
    viewport_shots: int = 0,
    width: int = 1280,
    height: int = 900,
) -> list[Path]:
    """Navigate to ``url`` and capture a full-page screenshot (+ optional shots).

    Args:
        url: Page to open.
        out_dir: Directory to write PNG screenshots into.
        viewport_shots: Extra viewport screenshots (scroll down) for tall pages.
        width: Viewport width.
        height: Viewport height.

    Returns:
        Saved screenshot paths (full page first).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=_LAUNCH_ARGS)
        page = await browser.new_page(viewport={"width": width, "height": height})
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:  # noqa: BLE001 - best-effort settle, don't fail capture
                pass
            await asyncio.sleep(0.5)

            full = out_dir / "full_page.png"
            await page.screenshot(path=str(full), full_page=True, type="png")
            paths.append(full)

            for i in range(viewport_shots):
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
                await asyncio.sleep(0.4)
                shot = out_dir / f"viewport_{i + 1:02d}.png"
                await page.screenshot(path=str(shot), full_page=False, type="png")
                paths.append(shot)
        finally:
            await browser.close()

    logger.info("page_captured", url=url, screenshots=len(paths), out_dir=str(out_dir))
    return paths


def capture_page(url: str, out_dir: str | Path, viewport_shots: int = 0) -> list[str]:
    """Sync wrapper over :func:`navigate_and_capture` for the sync pipeline."""
    paths = asyncio.run(navigate_and_capture(url, Path(out_dir), viewport_shots=viewport_shots))
    return [str(p) for p in paths]
