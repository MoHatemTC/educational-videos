"""Per-stage observability for the video pipeline.

Each LLM call records a trace row (stage, model, token usage, estimated USD cost,
latency) into the job's ``artifacts['traces']`` list so the Streamlit "Traces &
Cost" page and ``GET /videos/jobs/{id}/traces`` can surface it. Cost is always an
estimate derived from a configurable per-million-token price table.
"""

from app.core.config import settings
from app.core.logging import logger
from app.services.video_store import video_store


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts using the configured price table."""
    cost = (prompt_tokens / 1_000_000) * settings.LLM_PRICE_INPUT_PER_M + (
        completion_tokens / 1_000_000
    ) * settings.LLM_PRICE_OUTPUT_PER_M
    return round(cost, 6)


def record_trace(
    job_id: str,
    stage: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
) -> None:
    """Append a trace row to the job's artifacts (read-modify-write).

    Safe for the pipeline because stages run sequentially within one job. Never
    raises — observability must not break the pipeline.
    """
    try:
        job = video_store.get_job(job_id)
        if job is None:
            return
        traces = list((job.artifacts or {}).get("traces", []))
        traces.append(
            {
                "stage": stage,
                "model": model,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
                "est_cost_usd": estimate_cost_usd(prompt_tokens, completion_tokens),
                "latency_ms": int(latency_ms),
            }
        )
        video_store.update_job(job_id, artifacts_merge={"traces": traces})
        logger.info(
            "pipeline_trace_recorded",
            job_id=job_id,
            stage=stage,
            total_tokens=int(prompt_tokens) + int(completion_tokens),
            latency_ms=int(latency_ms),
        )
    except Exception as exc:  # noqa: BLE001 - observability must never break the run
        logger.warning("pipeline_trace_record_failed", job_id=job_id, stage=stage, error=str(exc))
