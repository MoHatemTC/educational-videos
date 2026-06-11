"""Vision-based UI agent PoC (browser-use): screenshot -> VLM -> click/type/scroll.

Give the agent a goal in plain English. It runs the Observe-Reason-Plan-Act loop
(screenshot -> model -> action) over a real Chromium until the goal is met or
MAX_STEPS is reached. No CSS selectors, no XPath. Routed through a LiteLLM proxy.
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from browser_use import Agent, BrowserSession
from browser_use.llm import ChatOpenAI

load_dotenv()

USE_VISION = os.getenv("USE_VISION", "true").lower() in ("1", "true", "yes")
HEADLESS = os.getenv("HEADLESS", "false").lower() in ("1", "true", "yes")
RECORD_VIDEO = os.getenv("RECORD_VIDEO", "true").lower() in ("1", "true", "yes")
CHROME_PATH = os.getenv("CHROME_PATH")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))

DEFAULT_TASK = (
    "Go to https://en.wikipedia.org, search for 'Computer vision', open the "
    "article, and report its first paragraph as Markdown."
)


def get_task() -> str:
    """Task from CLI args, else TASK_PROMPT env var, else the built-in default."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return os.getenv("TASK_PROMPT", DEFAULT_TASK).strip()


def build_llm() -> ChatOpenAI:
    """Chat model pointed at the LiteLLM proxy (OpenAI-compatible)."""
    base_url = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY")
    model = os.getenv("DEFAULT_MODEL")
    if not base_url or not api_key or not model:
        sys.exit("Set LITELLM_BASE_URL, LITELLM_API_KEY, and DEFAULT_MODEL in your .env.")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


async def main():
    """Run the vision agent proof-of-concept."""
    if not USE_VISION:
        print("[info] USE_VISION=false — using DOM/accessibility mode.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = OUTPUT_DIR / "run.gif"

    session_kwargs = {"headless": HEADLESS}
    if CHROME_PATH:
        session_kwargs["executable_path"] = CHROME_PATH
        session_kwargs["chromium_sandbox"] = False  # required for root in Docker

    video_dir = OUTPUT_DIR / "video"
    if RECORD_VIDEO:
        video_dir.mkdir(parents=True, exist_ok=True)
        session_kwargs["record_video_dir"] = str(video_dir)

    window = os.getenv("WINDOW_SIZE")
    if window and "x" in window:
        w, h = window.lower().split("x")[:2]
        session_kwargs["window_size"] = {"width": int(w), "height": int(h)}

    agent = Agent(
        task=get_task(),
        llm=build_llm(),
        use_vision=USE_VISION,
        browser_session=BrowserSession(**session_kwargs),
        generate_gif=str(gif_path),
    )

    history = await agent.run(max_steps=MAX_STEPS)
    save_screenshots(history)

    result = history.final_result() or "(no result produced)"
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(result, encoding="utf-8")

    print("\n==================== RESULT ====================")
    print(result)
    print("================================================")
    print(f"Report:      {report_path}")
    if gif_path.exists():
        print(f"Run GIF:     {gif_path}")
    print(f"Screenshots: {OUTPUT_DIR / 'screenshots'}/")
    if RECORD_VIDEO:
        videos = sorted(video_dir.glob("*.webm"))
        print(f"Video:       {videos[-1] if videos else '(none produced)'}")
    print()


def save_screenshots(history):
    """Write each step's screenshot (base64 in history) to output/screenshots/."""
    shots_dir = OUTPUT_DIR / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, shot in enumerate(history.screenshots(), start=1):
        if not shot:
            continue
        if shot.startswith("data:"):
            shot = shot.split(",", 1)[1]
        (shots_dir / f"step_{i:02d}.png").write_bytes(base64.b64decode(shot))
        saved += 1
    print(f"[info] Saved {saved} screenshot(s) to {shots_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
