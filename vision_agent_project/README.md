# Vision-Based UI Agent

**Sprint 1 — Computer Use Tooling**  
DOM-independent browser navigation via the Observe → Reason → Plan → Act loop.

---

## Overview

This agent drives a real Chromium browser using **screenshots + a Vision-Language Model (VLM)** — no CSS selectors, no XPath, no DOM introspection.

```
┌──────────────┐   PNG bytes   ┌──────────────────┐  base64 + task  ┌──────────────────┐
│  Playwright  │ ────────────▶ │   VisionAgent    │ ───────────────▶ │  LiteLLM Proxy   │
│  Chromium    │               │  (Observe-Reason  │ ◀─────────────── │  (Kimi K2.5 /    │
│  (headed /   │ ◀──────────── │   -Plan-Act loop) │  JSON action     │   GPT-4o / etc.) │
│  Xvfb)       │  click/type/  └──────────────────┘                  └──────────────────┘
│              │  scroll/nav
└──────────────┘
```

### Loop stages

| # | Stage       | What happens |
|---|-------------|--------------|
| 1 | **Observe** | Full-page PNG screenshot captured via Playwright |
| 2 | **Reason**  | Screenshot + task POSTed to VLM through LiteLLM proxy |
| 3 | **Plan**    | VLM JSON response parsed into a typed `Action` |
| 4 | **Act**     | Action executed; loop repeats until `done` or `MAX_STEPS` |

---

## Repository structure

```
vision_agent_project/
├── vision_agent/
│   ├── __init__.py      # Package exports
│   ├── agent.py         # Observe-Reason-Plan-Act loop + VLM calls
│   ├── browser.py       # Playwright/CDP controller (screenshot, click, type, scroll)
│   ├── recorder.py      # Screenshot persistence + GIF compilation
│   └── actions.py       # Typed Action dataclasses + JSON parsing
├── run.py               # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh        # Xvfb + ffmpeg + agent launcher
├── .env.example         # Configuration template
├── output/
│   ├── screenshots/     # step_001.png … step_NNN.png
│   ├── video/           # session.webm + screen_capture.mp4
│   ├── run.gif          # Animated GIF of all steps
│   └── report.md        # Agent's final result
└── README.md
```

---

## Quick start

### 1. Prerequisites

- Python 3.11
- Docker & docker-compose (for containerised runs)

### 2. Install (local)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
playwright install-deps chromium
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and fill in LITELLM_BASE_URL, LITELLM_API_KEY, DEFAULT_MODEL
```

### 4. Run locally

```bash
# Default task (Wikipedia Computer Vision article)
python run.py

# Custom task via CLI argument
python run.py "Go to https://news.ycombinator.com and list the top 5 headlines"

# Custom task via environment variable
TASK_PROMPT="Search DuckDuckGo for Playwright" python run.py
```

### 5. Run in Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Override the task
TASK_PROMPT="Go to example.com and describe the page" docker-compose up --build

# Run directly with docker
docker build -t vision-agent .
docker run --rm --env-file .env \
  -v "$(pwd)/output:/app/output" \
  vision-agent "Your task here"
```

---

## Configuration reference

All settings are controlled via environment variables (`.env` file or shell):

| Variable            | Default                            | Description |
|---------------------|------------------------------------|-------------|
| `GOOGLE_API_KEY`    | *(required)*                       | Google AI Studio / Gemini API key |
| `DEFAULT_MODEL`     | `gemini-2.5-flash`                 | Gemini model string |
| `TASK_PROMPT`       | Wikipedia Computer Vision          | Natural-language goal |
| `MAX_STEPS`         | `30`                               | Maximum Observe→Act iterations |
| `HEADLESS`          | `false`                            | Headless Chromium (true for CI) |
| `WINDOW_SIZE`       | `1280x900`                         | Browser viewport WxH |
| `CHROME_PATH`       | *(auto)*                           | Explicit Chromium path |
| `RECORD_VIDEO`      | `true`                             | Enable Playwright video recording |
| `GIF_FPS`           | `1`                                | Animated GIF frame rate |
| `OUTPUT_DIR`        | `output`                           | Root directory for all outputs |
| `LOG_LEVEL`         | `INFO`                             | `DEBUG` / `INFO` / `WARNING` |

---

## Output artefacts

After a run completes, the `output/` directory contains:

```
output/
├── screenshots/
│   ├── step_001.png     ← browser state before step 1
│   ├── step_002.png     ← browser state before step 2
│   └── …
├── video/
│   ├── session.webm     ← Playwright-recorded video
│   └── screen_capture.mp4  ← ffmpeg Xvfb capture (Docker only)
├── run.gif              ← Animated GIF compiled from screenshots
└── report.md            ← Agent's final result text
```

---

## Architecture details

### `vision_agent/agent.py` — VisionAgent

The core class. Owns a `BrowserController` and a `Recorder`, and implements the four loop stages plus the LiteLLM HTTP call.

Key method: `async def run() -> str`

### `vision_agent/browser.py` — BrowserController

Async context-manager wrapping a single Playwright `Page`. All browser I/O is isolated here so the agent loop stays clean.

Actions supported: `click(x, y)`, `type(text)`, `scroll(x, y, delta_y)`, `navigate(url)`, `wait(seconds)`.

### `vision_agent/actions.py` — Action / ActionType

Typed dataclasses for every VLM-producible action. Includes two factory methods:
- `Action.from_dict(data)` — from a parsed JSON dict
- `Action.from_vlm_text(text)` — extracts the first JSON object from free-form VLM output (handles markdown fences, preamble prose, etc.)

### `vision_agent/recorder.py` — Recorder

Handles all file I/O. Saves PNG frames to `screenshots/`, compiles them into an animated GIF using Pillow, and writes the final report.

### `entrypoint.sh`

Docker entrypoint that:
1. Starts Xvfb (virtual X display) so Chromium can run headed inside Docker
2. Launches ffmpeg to record the display to MP4
3. Runs `python run.py "$@"`
4. Stops ffmpeg cleanly on exit

---

## Extending the agent

### Add a new action type

1. Add a new `ActionType` value in `actions.py`
2. Add a branch in `BrowserController.execute()` in `browser.py`
3. Update the `SYSTEM_PROMPT` in `agent.py` to instruct the VLM

### Swap the VLM

Change `DEFAULT_MODEL` in `.env` to any supported Gemini vision model:
- `gemini-2.5-flash` — fast and cost-efficient (default)
- `gemini-2.5-pro` — highest capability
- `gemini-2.0-flash` — previous generation
- `gemini-1.5-pro` — stable long-context

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `LITELLM_BASE_URL … must be set` | Missing env vars | Copy `.env.example` → `.env` and fill in keys |
| `No valid action JSON found` | VLM not following prompt | Lower `temperature`, try a different model |
| `Chromium not found` | Playwright browsers not installed | `playwright install chromium` |
| Black screenshots in Docker | Xvfb not started | Ensure you run via `entrypoint.sh` |
| GIF not generated | Pillow not installed | `pip install Pillow` |

---

## License

Internal tooling — Sprint 1, Computer Use Tooling project.  
© 2026 Sprints AI. All rights reserved.
