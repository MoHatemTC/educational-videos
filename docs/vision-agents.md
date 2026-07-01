# Vision agent ownership

`app/services/pipeline/vision/` is the canonical vision-agent package for this repository.

It owns the integrated browser-vision pieces used by the video pipeline:

- URL capture and screenshot description for `web_explainer` jobs.
- DOM-independent recovery planning for popups, cookie banners, login walls, captchas, layout shifts, and navigation errors.
- The small async-compatible agent wrapper that calls recovery after observe/action steps.

## Canonical package

Use imports from `app.services.pipeline.vision` or its submodules:

```python
from app.services.pipeline.vision import RecoveryManager, VisionComputerUseAgent
```

The graduated recovery code consumes shared app infrastructure:

- configuration from `app/core/config.py`
- structured logs from `app/core/logging.py`
- lowercase underscore log events such as `invalid_vision_recovery_model_output`

Configuration is env-driven through `VISION_RECOVERY_*` settings. The old `code_and_log/agent_config.yaml` path has been removed.

## Legacy folders

The top-level `vision_agent_poc/` and `vision_agent_project/` folders are fenced historical prototypes. They are not import targets for new product work and should not receive new features. New work should either extend `app/services/pipeline/vision/` or be deleted after its useful pieces are migrated.

## Async posture

The canonical wrapper exposes an async `run_step` boundary so async browser loops can await screenshots and actions. `RecoveryManager` itself intentionally stays synchronous because it performs deterministic local work: screenshot diffing, retry accounting, JSONL event writing, safe action selection, and injected vision-client calls. In production, it runs inside the worker-owned browser automation process rather than the FastAPI event loop.
