"""Describe a web page from its screenshot(s) using Kimi vision.

Sends the screenshot(s) as base64 image_url content to the LiteLLM proxy
(OpenAI-compatible) and returns a factual, ordered description used as the
"research" context for the narration script. Records a cost trace.
"""

import base64
import time
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import settings
from app.core.logging import logger
from app.services.pipeline.observability import record_trace

_SYSTEM = (
    "You are a meticulous web-page analyst. Given screenshot(s) of a web page, describe what is actually visible "
    "section by section, top to bottom: brand/logo, navigation, search, hero/banner, main content (products, "
    "articles, cards) with any visible names and prices, and footer. Only report what you can see. Be concrete."
)


def describe_screenshots(screenshots: list[str], url: str, job_id: str | None = None) -> str:
    """Return a factual, ordered description of the page from its screenshot(s)."""
    client = OpenAI(base_url=settings.LITELLM_BASE_URL, api_key=settings.LITELLM_API_KEY)

    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Page URL: {url}\nDescribe this page section by section, top to bottom."}
    ]
    for shot in screenshots:
        b64 = base64.b64encode(Path(shot).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    started = time.monotonic()
    messages = cast(
        list[ChatCompletionMessageParam],
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": content}],
    )
    resp = client.chat.completions.create(
        model=settings.LITELLM_MODEL,
        temperature=0.0,
        max_tokens=1500,
        messages=messages,
    )
    latency_ms = int((time.monotonic() - started) * 1000)

    usage = getattr(resp, "usage", None)
    if job_id:
        record_trace(
            job_id,
            "vision_describe",
            settings.LITELLM_MODEL,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            latency_ms,
        )

    text = (resp.choices[0].message.content or "").strip()
    logger.info("page_described", url=url, screenshots=len(screenshots), chars=len(text))
    return text
