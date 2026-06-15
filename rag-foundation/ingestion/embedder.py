"""Embedding utilities for the RAG ingestion pipeline."""

import os
from typing import Final

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing_extensions import override

DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: Final[int] = 32


class SentenceTransformerEmbeddingFunction(Embeddings):
    """LangChain-compatible embedding function using sentence-transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the embedding function."""
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
    def model(self) -> SentenceTransformer:
        """Load and return the sentence-transformers model lazily."""
        model = self._model

        if model is None:
            model = SentenceTransformer(self.model_name)
            self._model = model

        return model

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
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
        """Embed a single query string."""
        if not text.strip():
            msg = "Query text must be non-empty."
            raise ValueError(msg)

        return self.embed_documents([text])[0]


def get_embedding_function(
    model_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Embeddings:
    """Create the shared embedding function for ingestion and retrieval."""
    resolved_model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL

    return SentenceTransformerEmbeddingFunction(
        model_name=resolved_model_name,
        batch_size=batch_size,
    )
