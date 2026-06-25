# VLM Render Mapper

Translates a **vision-agent browser-action session log** into a structured, reproducible **Render Plan JSON** for downstream programmatic video rendering via FFmpeg or Remotion.

---

## Architecture

```
session log (JSON / JSONL)
        │
        ▼
  ┌─────────────┐
  │   parser.py  │  Parse & normalise raw events
  └──────┬──────┘
         │  list[SessionEvent]
         ▼
  ┌─────────────┐
  │   timing.py  │  Compute durations, cursor easing, keyframes
  └──────┬──────┘
         │  list[FrameTiming] + list[CursorKeyframe]
         ▼
  ┌─────────────┐
  │   mapper.py  │  Build frames, highlights, zoom, captions, transitions
  └──────┬──────┘
         │  RenderPlan (Pydantic v2)
         ▼
  ┌─────────────┐
  │   schema.py  │  Validate & serialise → render_plan.json
  └─────────────┘
```

---

## Project Structure

```
vlm_render_mapper/
├── render_plan_schema.json        # JSON Schema (draft-07)
├── sample_session.json            # Example input session log
├── sample_render_plan.json        # Example output render plan
├── pyproject.toml
├── conftest.py
├── src/
│   └── vlm_render_mapper/
│       ├── __init__.py
│       ├── schema.py              # Pydantic v2 models
│       ├── parser.py              # Session log parser
│       ├── timing.py              # Duration & cursor-path computation
│       ├── mapper.py              # Action → frame mapper
│       └── cli.py                 # CLI entry point
└── tests/
    ├── test_parser.py
    ├── test_mapper.py
    └── test_timing.py
```

---

## Installation

```bash
# From project root — three equivalent options:

# Option 1: editable install (recommended for development)
pip install -e .

# Option 2: with dev dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"

# Option 3: install runtime deps only, run tests without installing
pip install pydantic>=2.0 Pillow
PYTHONPATH=src pytest
```

**Requirements:** Python 3.11+, Pydantic v2, Pillow, pytest

---

## Quick Start

### Map a session log → render plan

```bash
vlm-render-mapper --session sample_session.json --output render_plan.json --pretty
```

### With options

```bash
vlm-render-mapper \
  --session session.jsonl \
  --output out/plan.json \
  --fps 60 \
  --width 1920 --height 1080 \
  --speed 1.5 \
  --target remotion \
  --caption-position top \
  --click-zoom 2.0 \
  --pretty
```

### Validate an existing render plan

```bash
vlm-render-mapper --validate render_plan.json
```

### Python API

```python
from vlm_render_mapper import parse_session_file, RenderMapper, MapperConfig
from vlm_render_mapper.timing import TimingConfig

events = parse_session_file("sample_session.json")

cfg = MapperConfig(
    frame_rate=30,
    viewport_width=1280,
    viewport_height=720,
    timing=TimingConfig(speed_multiplier=1.0),
)

plan = RenderMapper(cfg).map(events)
print(plan.model_dump_json(indent=2))
```

---

## Session Log Format

Accepts **JSON array** or **JSON Lines** (`.jsonl`).  
Each event must have a `timestamp` field; all others are optional.

```json
[
  {
    "timestamp": 1700000000.0,
    "action": "navigate",
    "value": "https://example.com",
    "screenshot": "frame_000.png"
  },
  {
    "timestamp": 1700000001.5,
    "action": "click",
    "x": 640, "y": 360,
    "target": "#submit-btn",
    "screenshot": "frame_001.png"
  }
]
```

### Supported action types

| Raw value | Normalised to |
|---|---|
| `click`, `leftclick`, `left_click` | `click` |
| `double_click`, `dblclick` | `double_click` |
| `right_click`, `rightclick` | `right_click` |
| `hover`, `mouseover`, `mousemove` | `hover` |
| `type`, `input` | `type` |
| `key_press`, `keydown`, `keyboard` | `key_press` |
| `navigate`, `goto` | `navigate` |
| `page_load`, `load` | `page_load` |
| `scroll` | `scroll` |
| `drag` | `drag` |
| `screenshot`, `snap`, `capture` | `screenshot` |
| `wait`, `pause`, `sleep`, `delay` | `wait` |
| `focus` | `focus` |
| `blur` | `blur` |

---

## Render Plan Schema

The output `render_plan.json` conforms to `render_plan_schema.json` (JSON Schema draft-07)
and is also validated by Pydantic v2 `RenderPlan` on every run.

### Top-level fields

| Field | Description |
|---|---|
| `version` | Schema version (semver) |
| `metadata` | Session ID, timestamps, FPS, resolution, render target |
| `viewport` | Browser viewport size & DPR |
| `frames` | Ordered list of render frames |
| `timeline` | Segmented timeline referencing frame IDs |
| `cursor_path` | Interpolated cursor keyframes with easing |
| `captions` | Timed caption list |
| `transitions` | Inter-frame transitions (fade, cut, dissolve, …) |

---

## FFmpeg Usage Example

```bash
# Concatenate screenshots using the render plan timings
ffmpeg -f concat -safe 0 -i concat_list.txt \
       -vf "drawtext=fontfile=Arial.ttf:text='%{metadata\:caption}':..." \
       -r 30 output.mp4
```

## Remotion Usage Example

```tsx
// In your Remotion composition, load the render plan:
import planJson from "./render_plan.json";
// Drive <Sequence> durations from plan.frames[i].duration_ms
```

---

## Running Tests

```bash
pytest                          # all tests
pytest tests/test_parser.py     # parser only
pytest -v --tb=long             # verbose
pytest --cov=vlm_render_mapper  # with coverage
```

---

## CLI Reference

```
usage: vlm-render-mapper [-h] (--session FILE | --validate PLAN_FILE)
                         [--output FILE] [--fps N] [--width PX] [--height PX]
                         [--target {ffmpeg,remotion,both}] [--speed X]
                         [--min-frame-ms MS] [--max-gap-ms MS]
                         [--no-captions] [--caption-position POS]
                         [--no-zoom] [--no-highlight] [--click-zoom SCALE]
                         [--session-id ID] [--pretty]
```

---

## License

MIT
