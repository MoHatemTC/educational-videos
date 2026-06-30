"""Langfuse tracing helpers for LangGraph, OpenAI-compatible calls, and pipeline spans."""

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.core.config import settings
from app.core.logging import logger


def _clean_tags(tags: Sequence[str] | None) -> list[str]:
    """Return Langfuse-safe tags."""
    if not tags:
        return []
    return [str(tag)[:200] for tag in tags if str(tag).strip()]


def _configure_langfuse_environment() -> None:
    """Expose settings through the environment variables used by Langfuse."""
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.LANGFUSE_BASE_URL)
    os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_BASE_URL)
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", str(settings.LANGFUSE_TRACING_ENABLED).lower())


def langfuse_enabled() -> bool:
    """Return whether Langfuse tracing is configured."""
    return bool(settings.LANGFUSE_TRACING_ENABLED and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


def get_langfuse_client() -> Any | None:
    """Return the global Langfuse client, or ``None`` when tracing is disabled."""
    if not langfuse_enabled():
        return None

    try:
        _configure_langfuse_environment()
        return get_client()
    except Exception as exc:  # noqa: BLE001 - observability must not break app boot
        logger.warning("langfuse_client_unavailable", error=str(exc))
        return None


def langfuse_init() -> None:
    """Initialize Langfuse and verify credentials once during app startup."""
    client = get_langfuse_client()
    if client is None:
        logger.info("langfuse_disabled_or_missing_credentials")
        return

    try:
        if client.auth_check():
            logger.info(
                "langfuse_auth_success",
                host=settings.LANGFUSE_BASE_URL,
                environment=settings.LANGFUSE_ENVIRONMENT,
                release=settings.LANGFUSE_RELEASE,
            )
        else:
            logger.warning("langfuse_auth_failure", host=settings.LANGFUSE_BASE_URL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_auth_check_failed", error=str(exc))


def flush_langfuse() -> None:
    """Flush queued Langfuse events before application shutdown."""
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
        logger.info("langfuse_flushed")
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_flush_failed", error=str(exc))


def get_langfuse_callback_handler() -> CallbackHandler | None:
    """Return a Langfuse LangChain callback handler when tracing is enabled."""
    if not langfuse_enabled():
        return None

    try:
        _configure_langfuse_environment()
        return CallbackHandler()
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_callback_unavailable", error=str(exc))
        return None


def langfuse_callbacks() -> list[BaseCallbackHandler]:
    """Return callbacks for LangChain/LangGraph invocations."""
    handler = get_langfuse_callback_handler()
    return [handler] if handler is not None else []


@contextmanager
def langfuse_trace(
    *,
    name: str,
    as_type: str = "span",
    input_data: Any | None = None,
    output_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: Sequence[str] | None = None,
) -> Iterator[Any | None]:
    """Create a Langfuse observation and never mask errors from the wrapped code."""
    client = get_langfuse_client()
    clean_tags = _clean_tags(tags)

    if client is None:
        yield None
        return

    stack = ExitStack()
    try:
        observation = stack.enter_context(
            client.start_as_current_observation(
                as_type=as_type,
                name=name,
                input=input_data,
                metadata=metadata,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_trace_failed_open", name=name, error=str(exc))
        yield None
        return

    try:
        try:
            observation.update_trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                tags=clean_tags,
                metadata=metadata,
                output=output_data,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse_trace_update_failed", name=name, error=str(exc))

        try:
            stack.enter_context(
                propagate_attributes(
                    user_id=user_id,
                    session_id=session_id,
                    tags=clean_tags,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse_attribute_propagation_failed", name=name, error=str(exc))

        try:
            yield observation
        except Exception as exc:
            try:
                observation.update(output={"status": "error", "error": str(exc)})
            except Exception as update_exc:  # noqa: BLE001
                logger.warning("langfuse_error_update_failed", name=name, error=str(update_exc))
            raise
    finally:
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse_trace_close_failed", name=name, error=str(exc))
