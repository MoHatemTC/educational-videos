"""Embedding utilities for the RAG ingestion pipeline."""

import os
from typing import Final

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing_extensions import override
from rag_tool.config import DEFAULT_EMBEDDING_MODEL, get_settings


DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: Final[int] = 32

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class SentenceTransformerEmbeddingFunction(Embeddings):
    """LangChain-compatible embedding function using sentence-transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the embedding function.

        Args:
            model_name: Sentence-transformers model name.
            batch_size: Number of texts to embed per batch.
            normalize_embeddings: Whether to L2-normalize embeddings.

        Raises:
            ValueError: If model name or batch size is invalid.
        """
        if not model_name:
            msg = "model_name must be a non-empty string."
            raise ValueError(msg)

        if batch_size <= 0:
            msg = "batch_size must be greater than 0."
            raise ValueError(msg)

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None

    @property
    @property
    def model(self) -> SentenceTransformer:
        """Load and return the sentence-transformers model lazily.

        Returns:
            Loaded sentence-transformers model.
        """
        model = self._model

        if model is None:
            settings = get_settings()

            if settings.disable_hf_symlink_warning:
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

            model = SentenceTransformer(
                self.model_name,
                token=settings.hf_token,
            )
            self._model = model

        return model

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: Text chunks to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        cleaned_texts = [text if text.strip() else " " for text in texts]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    @override
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query text.

        Returns:
            Query embedding vector.

        Raises:
            ValueError: If query text is empty.
        """
        if not text.strip():
            msg = "Query text must be non-empty."
            raise ValueError(msg)

        return self.embed_documents([text])[0]


def get_embedding_function(
    model_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Embeddings:
    """Create the shared embedding function for ingestion and retrieval.

    Args:
        model_name: Optional embedding model name.
        batch_size: Number of texts to embed per batch.

    Returns:
        LangChain-compatible embedding function.
    """
    settings = get_settings()
    resolved_model_name = model_name or settings.embedding_model

    return SentenceTransformerEmbeddingFunction(
        model_name=resolved_model_name,
        batch_size=batch_size,
    )
