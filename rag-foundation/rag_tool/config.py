"""Environment-backed configuration for the RAG foundation stack."""

from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

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
    """Load typed RAG settings from environment variables.

    Returns:
        Validated RAG settings.
    """
    load_dotenv()

    import os

    return RagSettings(
        embedding_provider=os.getenv(
            "EMBEDDING_PROVIDER",
            DEFAULT_EMBEDDING_PROVIDER,
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        chroma_persist_dir=Path(
            os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_PERSIST_DIR),
        ),
        chroma_collection=os.getenv(
            "CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", str(DEFAULT_TOP_K))),
        default_similarity_threshold=float(
            os.getenv(
                "DEFAULT_SIMILARITY_THRESHOLD",
                str(DEFAULT_SIMILARITY_THRESHOLD),
            )
        ),
        hf_token=os.getenv("HF_TOKEN") or None,
        disable_hf_symlink_warning=os.getenv(
            "HF_HUB_DISABLE_SYMLINKS_WARNING",
            "1",
        )
        == "1",
    )
