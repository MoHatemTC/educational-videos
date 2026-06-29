"""Pyright-safe factory for Langfuse-traced OpenAI-compatible clients."""

from importlib import import_module
from typing import Any, cast


def create_openai_client(*, base_url: str | None, api_key: str | None) -> Any:
    """Create a Langfuse-traced OpenAI client without private import typing errors."""
    langfuse_openai = cast(Any, import_module("langfuse.openai"))
    return langfuse_openai.OpenAI(base_url=base_url, api_key=api_key)
