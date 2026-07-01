"""Environment-backed configuration for the integrated RAG stack."""

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings as app_settings

DEFAULT_EMBEDDING_PROVIDER: Final[str] = "sentence_transformers"
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHROMA_PERSIST_DIR: Final[str] = ".chroma"
DEFAULT_CHROMA_COLLECTION: Final[str] = "technical_docs"
DEFAULT_TOP_K: Final[int] = 5
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.35


class RagSettings(BaseModel):
    """Typed runtime settings for ingestion and retrieval."""

    embedding_provider: str = Field(default=DEFAULT_EMBEDDING_PROVIDER)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    chroma_persist_dir: Path = Field(default=Path(DEFAULT_CHROMA_PERSIST_DIR))
    chroma_collection: str = Field(default=DEFAULT_CHROMA_COLLECTION)
    default_top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    default_similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    hf_token: str | None = None
    disable_hf_symlink_warning: bool = True

    @field_validator("embedding_provider", "embedding_model", "chroma_collection")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Validate string settings.

        Args:
            value: Raw setting value.

        Returns:
            Stripped setting value.

        Raises:
            ValueError: If the setting is empty.
        """
        normalized_value = value.strip()

        if not normalized_value:
            msg = "Setting value must be non-empty."
            raise ValueError(msg)

        return normalized_value


@lru_cache(maxsize=1)
def get_settings() -> RagSettings:
    """Load typed RAG settings from application settings.

    Returns:
        Validated RAG settings.
    """
    return RagSettings(
        embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
        embedding_model=app_settings.EMBEDDING_MODEL,
        chroma_persist_dir=Path(app_settings.RAG_CHROMA_PERSIST_DIR),
        chroma_collection=app_settings.RAG_CHROMA_COLLECTION,
        default_top_k=app_settings.RAG_TOP_K,
        default_similarity_threshold=app_settings.RAG_SIMILARITY_THRESHOLD,
        hf_token=app_settings.HF_TOKEN or None,
    )
