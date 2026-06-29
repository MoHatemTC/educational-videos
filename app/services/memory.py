"""Long-term memory service using mem0 and pgvector with optional cache layer."""

from inspect import isawaitable
from typing import Any, cast

from mem0 import AsyncMemory

from app.core.cache import (
    cache_key,
    cache_service,
)
from app.core.config import settings
from app.core.logging import logger


class MemoryService:
    """Service for managing long-term memory using mem0 and pgvector."""

    def __init__(self) -> None:
        """Initialize the memory service."""
        self._memory: AsyncMemory | None = None

    async def _get_memory(self) -> AsyncMemory:
        """Get or create the shared mem0 AsyncMemory instance.

        Returns:
            Cached AsyncMemory instance.
        """
        memory = self._memory

        if memory is None:
            raw_memory: Any = AsyncMemory.from_config(
                config_dict={
                    "vector_store": {
                        "provider": "pgvector",
                        "config": {
                            "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                            "dbname": settings.POSTGRES_DB,
                            "user": settings.POSTGRES_USER,
                            "password": settings.POSTGRES_PASSWORD,
                            "host": settings.POSTGRES_HOST,
                            "port": settings.POSTGRES_PORT,
                        },
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": settings.LITELLM_MODEL,
                            "api_key": settings.LITELLM_API_KEY,
                            "openai_base_url": settings.LITELLM_BASE_URL,
                        },
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
                            "api_key": settings.LITELLM_API_KEY,
                            "openai_base_url": settings.LITELLM_BASE_URL,
                        },
                    },
                }
            )

            if isawaitable(raw_memory):
                raw_memory = await raw_memory

            memory = cast(AsyncMemory, raw_memory)
            self._memory = memory

        return memory

    async def initialize(self) -> None:
        """Pre-warm the mem0 AsyncMemory instance and its pgvector connection pool.

        Call once at startup so the first search() or add() does not pay the
        from_config and pgvector cold-init cost.
        """
        await self._get_memory()
        logger.info("memory_service_initialized")

    async def search(self, user_id: str | None, query: str) -> str:
        """Search relevant memories for a user.

        Checks cache first. On miss, queries mem0 and caches the result.

        Args:
            user_id: User identifier, or None for anonymous sessions.
            query: Search query.

        Returns:
            Formatted memory string, or an empty string on failure/no user.
        """
        if user_id is None:
            return ""

        try:
            key = cache_key("memory", str(user_id), query)
            cached = await cache_service.get(key)

            if cached is not None:
                logger.debug("memory_search_cache_hit", user_id=user_id)
                return cached

            memory = await self._get_memory()
            results = await memory.search(user_id=str(user_id), query=query)
            result = "\n".join(
                [f"* {r['memory']}" for r in results["results"]])

            if result:
                await cache_service.set(key, result)

            return result
        except Exception as exc:
            logger.error(
                "failed_to_get_relevant_memory",
                error=str(exc),
                user_id=user_id,
                query=query,
            )
            return ""

    async def add(
        self,
        user_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add messages to long-term memory for a user.

        Args:
            user_id: User identifier, or None for anonymous sessions.
            messages: Messages to store.
            metadata: Optional metadata to attach.
        """
        if user_id is None:
            return

        try:
            memory = await self._get_memory()
            await memory.add(messages, user_id=str(user_id), metadata=metadata)
            logger.info("long_term_memory_updated_successfully",
                        user_id=user_id)
        except Exception as exc:
            logger.exception(
                "failed_to_update_long_term_memory",
                user_id=user_id,
                error=str(exc),
            )


memory_service = MemoryService()
