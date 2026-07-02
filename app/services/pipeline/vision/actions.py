"""Coordinate-based browser action loop for web explainer demos.

The loop deliberately acts through viewport coordinates instead of DOM selectors,
matching the PRD's computer-use style automation requirement. A caller can pass a
VLM/browser-agent action plan consisting of click, scroll, type, and wait steps;
the function captures a screenshot after navigation and after each action.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

from app.core.logging import logger
from app.services.pipeline.vision.browser import _LAUNCH_ARGS, _ensure_public_http_url


async def _apply_action(page: Page, action: dict[str, Any]) -> None:
    """Apply one coordinate action to the Playwright page."""
    kind = str(action.get("action") or "").lower()
    if kind == "click":
        await page.mouse.click(int(action.get("x", 0)), int(action.get("y", 0)))
    elif kind == "scroll":
        await page.mouse.wheel(0, int(action.get("delta_y", action.get("y", 600))))
    elif kind == "type":
        if action.get("x") is not None and action.get("y") is not None:
            await page.mouse.click(int(action["x"]), int(action["y"]))
        await page.keyboard.type(str(action.get("text") or ""))
    elif kind == "wait":
        await page.wait_for_timeout(int(action.get("wait_ms", 500)))
    else:
        raise ValueError(f"unsupported vision action: {kind!r}")


async def navigate_act_and_capture(
    url: str,
    out_dir: Path,
    actions: list[dict[str, Any]],
    width: int = 1280,
    height: int = 900,
) -> list[Path]:
    """Navigate, perform coordinate actions, and capture screenshots.

    Args:
        url: Public http(s) URL to drive.
        out_dir: Screenshot output directory.
        actions: Coordinate action dictionaries.
        width: Browser viewport width.
        height: Browser viewport height.

    Returns:
        Saved screenshot paths, starting with the initial page state.
    """
    _ensure_public_http_url(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots: list[Path] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=_LAUNCH_ARGS)
        page = await browser.new_page(viewport={"width": width, "height": height})
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:  # noqa: BLE001 - best-effort settle only
                pass

            initial = out_dir / "action_00.png"
            await page.screenshot(path=str(initial), full_page=False, type="png")
            screenshots.append(initial)

            for index, action in enumerate(actions, start=1):
                await _apply_action(page, action)
                await page.wait_for_timeout(int(action.get("wait_after_ms", 350)))
                shot = out_dir / f"action_{index:02d}.png"
                await page.screenshot(path=str(shot), full_page=False, type="png")
                screenshots.append(shot)
        finally:
            await browser.close()

    logger.info("coordinate_action_capture_done", url=url, actions=len(actions), screenshots=len(screenshots))
    return screenshots


def capture_page_with_actions(url: str, out_dir: str | Path, actions: list[dict[str, Any]]) -> list[str]:
    """Sync wrapper for coordinate-action web capture."""
    paths = asyncio.run(navigate_act_and_capture(url, Path(out_dir), actions))
    return [str(path) for path in paths]
