"""Embedding model wrapper using Sentence Transformers (all-MiniLM-L6-v2, 384-dim)."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config.logging_config import logger
from config.settings import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embed_model)
        _model = SentenceTransformer(settings.embed_model)
    return _model


class EmbeddingClient:
    """Wraps Sentence Transformers with L2-normalized output for cosine similarity via dot product."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embed_model
        # Trigger load at init time so the first call isn't slow
        self._model = SentenceTransformer(self._model_name)
        logger.info("EmbeddingClient ready: %s (dim=%d)", self._model_name, settings.embed_dim)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of strings. Returns L2-normalized array of shape (N, embed_dim)."""
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns 1D array of shape (embed_dim,)."""
        return self.embed([text])[0]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors (= dot product)."""
        return float(np.dot(a, b))
