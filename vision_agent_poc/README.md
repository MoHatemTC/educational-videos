# Vision-based UI Agent — Browser Automation PoC

A minimal proof of concept for browser automation driven by **vision**, not by
brittle HTML selectors. You give the agent a goal in plain English; the model
decides what to click and type. No CSS, no XPath.

Built with **[browser-use](https://github.com/browser-use/browser-use)** (LLMs
driving a real Chromium over CDP) and routed through your **LiteLLM** proxy.

## How it maps to the Observe → Reason → Plan → Act loop

| Step | What happens | Who does it |
|------|--------------|-------------|
| **Observe** | Screenshot (or DOM/accessibility tree) of the current page | browser-use + Chromium |
| **Reason**  | The state is sent to a model via your LiteLLM proxy | your multimodal model |
| **Plan**    | The model picks the next single step toward the goal | the model |
| **Act**     | The decision becomes a real click / type / scroll | browser-use → Chromium (CDP) |

The loop repeats until the goal is met or `max_steps` is hit.

## The task

The task is plain English — no selectors. The built-in default opens Wikipedia,
searches for an article, and reports its first paragraph. Override it without
touching the code:

```bash
python agent_poc.py "Go to https://news.ycombinator.com and list the top 5 story titles."
# or set TASK_PROMPT in .env
```

## Outputs

Every run writes to `output/` (mounted to your host in Docker):

| File | What it is |
|------|------------|
| `output/report.md` | The agent's final answer / report |
| `output/video/process.mp4` | **Continuous video of the whole run** (see Recording) |
| `output/run.gif` | Annotated GIF: each step's screenshot + the agent's goal |
| `output/screenshots/step_NN.png` | What the agent saw at each step |

## Recording the full process to video

browser-use 0.12 drives Chromium over CDP and **cannot** record a video via its
`record_video_dir` option (that's a Playwright-context feature). So the Docker
image records the *real* way: it runs Chromium **headful inside a virtual display
(Xvfb)** and captures that display continuously with **ffmpeg** → `process.mp4`
(H.264). This is orchestrated by `entrypoint.sh`.

Bonus: because this uses a genuine headful browser, it also gets past anti-bot
checks that block the headless browser.

## Vision mode

This is a *vision* agent, so point `DEFAULT_MODEL` at a multimodal
(vision-language) model exposed by your proxy. Then:

- **`USE_VISION=true` (default)** → the agent reasons over real screenshots.
- **`USE_VISION=false`** → falls back to the accessibility/DOM tree; only needed
  if you switch `DEFAULT_MODEL` to a text-only model.

## Configuration (.env)

```bash
cp .env.example .env      # then fill in your proxy URL, key, and model
```

| Variable | Purpose |
|----------|---------|
| `LITELLM_API_KEY` | Auth for your LiteLLM proxy |
| `LITELLM_BASE_URL` | Your proxy URL |
| `DEFAULT_MODEL` | Model name as registered in the proxy (multimodal for vision) |
| `TASK_PROMPT` | Task in plain English (overridden by a CLI arg) |
| `USE_VISION` | `true` = send screenshots (needs multimodal); `false` = DOM mode |
| `HEADLESS` | `false` = show the browser locally; `true` = no window (Docker) |
| `MAX_STEPS` | Max Observe-Reason-Plan-Act steps before stopping |
| `CHROME_PATH` | Optional path to a Chrome/Chromium binary (set automatically in Docker) |

> If requests fail with a 404, try appending `/v1` to `LITELLM_BASE_URL`.

## Option A — Run locally

Requires **Python 3.11** and a Chrome/Chromium installed on your machine
(browser-use 0.12 drives it over CDP; it auto-detects a system Chrome, or set
`CHROME_PATH` in `.env` to point at a specific binary).

```bash
cd vision_agent_poc
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # adjust USE_VISION / DEFAULT_MODEL if needed
python agent_poc.py
```

With `HEADLESS=false` a Chromium window opens so you can watch it work.

## Option B — Run in Docker ✅ (verified)

The image bundles Python, Chromium (`/usr/bin/chromium`), Xvfb, ffmpeg, and the
OS libraries the browser needs. `entrypoint.sh` runs Chromium headful inside a
virtual display and records the whole run to `output/video/process.mp4`. The
browser runs with the sandbox disabled (required for root in a container); your
secrets come from `.env` at runtime, not baked into the image.

```bash
cd vision_agent_poc
cp .env.example .env               # make sure it's filled in

# Build and run
docker compose up --build
```

Expected tail of a successful run:

```
vision-agent-1  | 📝 Report:       output/report.md
vision-agent-1  | 🎞  Run GIF:      output/run.gif
vision-agent-1  | 🖼  Screenshots:  output/screenshots/
vision-agent-1  | 🎥 Full-process video: output/video/process.mp4
vision-agent-1 exited with code 0
```

To run again later without rebuilding:

```bash
docker compose up
```

To tear down:

```bash
docker compose down
```

### Plain docker (no compose)

```bash
docker build -t vision-agent .
docker run --rm --env-file .env -e HEADLESS=true --shm-size=2g vision-agent
```

> `CHROME_PATH=/usr/bin/chromium` is already baked into the image, so you don't
> need to pass it.

> `--shm-size=2g` (and `shm_size` in compose) prevents Chromium from crashing on
> Docker's tiny default `/dev/shm`.

## Notes & gotchas

- **Cost:** in vision mode each step sends a screenshot to the model, so a run
  is a handful of calls. `MAX_STEPS` (default 30) caps a runaway loop.
- **Pinned version:** verified against `browser-use==0.12.9` (LLM wrappers in
  `browser_use.llm`, `BrowserSession` for browser config, Chromium driven over
  CDP — no Playwright CLI). If you bump the version and something breaks, check
  the [browser-use docs](https://docs.browser-use.com).
- **The built-in judge** may print a `FAIL` verdict even on a correct result
  (e.g. when Wikipedia redirects a search straight to the article, so there's no
  separate "click result" step). It's advisory; the `Final Result` is what
  matters.
