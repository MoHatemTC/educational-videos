"""LLM model registry — all models routed through the LiteLLM proxy."""

from typing import (
    Any,
    Dict,
    List,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.core.logging import logger

_API_KEY = SecretStr(settings.LITELLM_API_KEY or "litellm")
_BASE_URL = settings.LITELLM_BASE_URL
_MODEL = settings.LITELLM_MODEL


def _make_kimi(**overrides: Any) -> ChatOpenAI:
    """Build a ChatOpenAI bound to the LiteLLM proxy (Kimi K2.6).

    Args:
        **overrides: Per-call parameter overrides (e.g. temperature).

    Returns:
        A configured ``ChatOpenAI`` talking to the LiteLLM proxy.
    """
    params: Dict[str, Any] = {
        "model": _MODEL,
        "api_key": _API_KEY,
        "base_url": _BASE_URL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }
    params.update(overrides)
    return ChatOpenAI(**params)


class LLMRegistry:
    """Registry of available LLM models (single LiteLLM/Kimi backend).

    Kept as a registry (rather than a single instance) so the circular-fallback
    ``LLMService`` keeps working unchanged; with one backend it simply retries
    the same proxy model.
    """

    LLMS: List[Dict[str, Any]] = [
        {"name": _MODEL, "llm": _make_kimi()},
    ]

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """Get an LLM by name, falling back to the Kimi backend if unknown.

        Args:
            model_name: Requested model name. Unknown names resolve to Kimi
                (single-provider setup), so callers configured for the old
                gpt-5* names keep working.
            **kwargs: Optional per-call overrides; when present a fresh instance
                is returned, leaving the shared registry entry untouched.

        Returns:
            A ``BaseChatModel`` instance routed through the LiteLLM proxy.
        """
        model_entry = next((e for e in cls.LLMS if e["name"] == model_name), None)

        if model_entry is None:
            logger.debug("llm_model_not_registered_using_kimi", requested=model_name)
            return _make_kimi(**kwargs) if kwargs else cls.LLMS[0]["llm"]

        if kwargs:
            logger.debug("creating_llm_with_custom_args", model_name=model_name, custom_args=list(kwargs.keys()))
            return _make_kimi(**kwargs)

        return model_entry["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Return all registered model names in order."""
        return [e["name"] for e in cls.LLMS]

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """Return the model entry at an index, wrapping to 0 if out of range."""
        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]
        return cls.LLMS[0]
