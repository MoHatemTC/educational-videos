"""LiteLLM client wrapper for the structured-outputs project.

This file is responsible for calling the shared Sprint LiteLLM endpoint.

The rest of the project should not directly call the OpenAI SDK.
Instead, use:

    llm_client.generate_json(prompt)

or:

    llm_client.generate_json(prompt, schema=Timeline.model_json_schema())

Environment variables expected in .env:

    LITELLM_BASE_URL   Required. Base URL for the LiteLLM proxy.
    LITELLM_API_KEY    Required. Shared API key from Sprints.
    DEFAULT_MODEL      Required. Azure/LiteLLM model name from Sprints.
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
    """Small wrapper around an OpenAI-compatible LiteLLM endpoint.

    Main responsibility:
        Given a prompt, return JSON text from the model.

    This keeps API details isolated from prompt_chain.py, eval_harness.py,
    and validate_repair.py.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        """Initialize the LiteLLM client.

        Args:
            model: Optional model override. Defaults to DEFAULT_MODEL.
            api_key: Optional key override. Defaults to LITELLM_API_KEY.
            base_url: Optional base URL override. Defaults to LITELLM_BASE_URL.
            temperature: Low temperature keeps JSON output more stable.
        """
        if load_dotenv is not None:
            load_dotenv()

        self.model = model or os.getenv("DEFAULT_MODEL")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL")
        self.temperature = temperature

        missing: list[str] = []

        if not self.model:
            missing.append("DEFAULT_MODEL")

        if not self.api_key:
            missing.append("LITELLM_API_KEY")

        if not self.base_url:
            missing.append("LITELLM_BASE_URL")

        if missing:
            raise LLMClientError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        schema_name: str = "structured_output",
    ) -> str:
        """Generate JSON text from the LLM.

        Args:
            prompt: Full prompt sent to the model.
            schema: Optional JSON Schema for schema-constrained decoding.
            schema_name: Name used for the response_format schema.

        Returns:
            Raw JSON string returned by the model.

        Raises:
            LLMClientError: If prompt is empty, API call fails, or response is empty.
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

        request_kwargs = self._build_request_kwargs(
            prompt=full_prompt,
            schema=schema,
            schema_name=schema_name,
            use_response_format=self._should_use_response_format(),
        )

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as first_error:
            if schema is None:
                raise LLMClientError(
                    f"LLM API call failed: {first_error}"
                ) from first_error

            fallback_kwargs = self._build_request_kwargs(
                prompt=full_prompt,
                schema=None,
                schema_name=schema_name,
                use_response_format=False,
            )

            try:
                response = self.client.chat.completions.create(**fallback_kwargs)
            except Exception as second_error:
                raise LLMClientError(
                    "LLM API call failed with schema-constrained decoding, "
                    "then failed again without response_format.\n\n"
                    f"First error: {first_error}\n"
                    f"Second error: {second_error}"
                ) from second_error

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
        """Generate JSON and parse it into a Python dictionary.

        This is a convenience method. The main pipeline usually keeps raw text
        so validate_repair.py can handle JSON parsing and repair.
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
            raise LLMClientError("LLM output must be a JSON object.")

        return parsed

    def _build_request_kwargs(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        schema_name: str,
        use_response_format: bool,
    ) -> dict[str, Any]:
        """Build kwargs for the OpenAI-compatible chat completion request."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
        }

        if use_response_format:
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

        return request_kwargs

    @staticmethod
    def _should_use_response_format() -> bool:
        """Decide whether to send response_format to LiteLLM.

        Defaults to true because the sprint requires schema-constrained decoding.
        Set LITELLM_USE_RESPONSE_FORMAT=false only if the backend rejects it.
        """
        value = os.getenv("LITELLM_USE_RESPONSE_FORMAT", "true")
        return value.strip().lower() not in {"false", "0", "no"}