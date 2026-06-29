"""Pipeline LLM helper — Kimi K2.6 via the LiteLLM proxy (OpenAI SDK).

A thin synchronous wrapper used by the generation agents. Unlike
``app.core.llm_client.LLMClient`` (which only returns text), this captures token
usage and latency and records an observability trace per call. Kimi is a
reasoning model: the visible answer is in ``message.content`` (chain-of-thought
goes to ``reasoning_content``), so callers get clean output.
"""

import time

from app.core.traced_openai import create_openai_client

from app.core.config import settings
from app.core.logging import logger
from app.services.pipeline.observability import record_trace


class PipelineLLM:
    """Synchronous Kimi client that records token/cost/latency traces."""

    def __init__(self, job_id: str | None = None) -> None:
        """Create a client bound to the LiteLLM proxy.

        Args:
            job_id: When set, every call records a trace row on this job.
        """
        self.client = create_openai_client(base_url=settings.LITELLM_BASE_URL, api_key=settings.LITELLM_API_KEY)
        self.model = settings.LITELLM_MODEL
        self.job_id = job_id

    def _metadata(self, stage: str) -> dict[str, object]:
        """Return Langfuse metadata for this pipeline LLM call."""
        tags = ["video-pipeline", stage[:200], settings.ENVIRONMENT.value]
        metadata: dict[str, object] = {
            "pipeline_stage": stage,
            "environment": settings.ENVIRONMENT.value,
            "langfuse_tags": tags,
        }
        if self.job_id:
            metadata["job_id"] = self.job_id
            metadata["langfuse_session_id"] = self.job_id
        return metadata

    def complete(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_object: bool = False,
    ) -> str:
        """Run one chat completion and return the assistant's text.

        Args:
            stage: Pipeline stage name (used as the trace label).
            system: System prompt.
            user: User prompt.
            temperature: Override sampling temperature.
            max_tokens: Override max output tokens.
            json_object: Request ``response_format={'type':'json_object'}``.

        Returns:
            The stripped assistant message content.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            "name": f"video_pipeline.{stage}",
            "metadata": self._metadata(stage),
        }
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - started) * 1000)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        if self.job_id:
            record_trace(self.job_id, stage, self.model, prompt_tokens, completion_tokens, latency_ms)

        content = response.choices[0].message.content or ""
        if not content.strip():
            logger.warning("pipeline_llm_empty_content", stage=stage, finish_reason=response.choices[0].finish_reason)
        return content.strip()
