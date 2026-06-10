"""
LLM client wrapper for the structured-outputs project.

This file is responsible for talking to an OpenAI-compatible API.

The rest of the project should NOT directly call OpenAI.
Instead, other files should call:

    llm_client.generate_json(prompt)

or:

    llm_client.generate_json(prompt, schema=Timeline.model_json_schema())

Environment variables used:

    OPENAI_API_KEY   Required. Your API key or compatible provider token.
    OPENAI_MODEL     Optional. Defaults to gpt-4o-mini.
    OPENAI_BASE_URL  Optional. Use only for OpenAI-compatible providers.

Example official OpenAI setup:

    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini

Example OpenAI-compatible setup:

    OPENAI_API_KEY=your_provider_token
    OPENAI_MODEL=provider_model_name
    OPENAI_BASE_URL=https://example.com/openai/v1/
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class LLMClientError(Exception):
    """Raised when the LLM client cannot be configured or called."""


class LLMClient:
    """
    Small wrapper around the OpenAI Python SDK.

    Main job:
        Given a prompt, return JSON text.

    Why this exists:
        It keeps API details isolated in one file.
        prompt_chain.py should not care about OpenAI syntax.
    """

    def __init__(
            self,
            model: str | None = None,
            api_key: str | None = None,
            base_url: str | None = None,
            temperature: float = 0.0,
    ) -> None:
        """
        Create an LLM client.

        Args:
            model: Model name. If None, reads OPENAI_MODEL.
            api_key: API key. If None, reads OPENAI_API_KEY.
            base_url: Optional OpenAI-compatible base URL.
            temperature: Lower values make output more deterministic.
        """
        if load_dotenv is not None:
            load_dotenv()

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.temperature = temperature

        if not self.api_key:
            raise LLMClientError(
                "OPENAI_API_KEY is missing. "
                "Create a .env file or set it in your environment."
            )

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    def generate_json(
            self,
            prompt: str,
            schema: dict[str, Any] | None = None,
            schema_name: str = "structured_output",
    ) -> str:
        """
        Generate JSON text from the LLM.

        Uses a simple single-message format for better compatibility with
        OpenAI-compatible providers such as Puter.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMClientError("Prompt must be a non-empty string.")

        full_prompt = (
            "You are a strict JSON generation engine.\n"
            "Return JSON only.\n"
            "Do not include markdown fences.\n"
            "Do not include explanations.\n"
            "Do not include comments.\n\n"
            f"{prompt}"
        )

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            "temperature": self.temperature,
        }

        # Official OpenAI supports response_format=json_schema.
        # Some OpenAI-compatible providers may not fully support it.
        use_response_format = os.getenv("OPENAI_USE_RESPONSE_FORMAT", "true").lower()

        if use_response_format == "true":
            if schema is not None:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                }
            else:
                request_kwargs["response_format"] = {
                    "type": "json_object",
                }

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as error:
            raise LLMClientError(f"LLM API call failed: {error}") from error

        content = response.choices[0].message.content

        if not content:
            raise LLMClientError("LLM returned an empty response.")

        return content.strip()

    def generate_json_dict(
            self,
            prompt: str,
            schema: dict[str, Any] | None = None,
            schema_name: str = "structured_output",
    ) -> dict[str, Any]:
        """
        Generate JSON and parse it into a Python dictionary.

        This is a convenience method. The main pipeline can still use
        generate_json() when it wants raw text for validation/repair.
        """
        raw_output = self.generate_json(
            prompt=prompt,
            schema=schema,
            schema_name=schema_name,
        )

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise LLMClientError(
                f"LLM output was not valid JSON: {error}\n\n"
                f"Raw output:\n{raw_output}"
            ) from error

        if not isinstance(parsed, dict):
            raise LLMClientError(
                "LLM output must be a JSON object at the top level."
            )

        return parsed
