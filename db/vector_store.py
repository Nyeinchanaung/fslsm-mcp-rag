"""FAISS in-memory vector store — loaded from PostgreSQL chunk embeddings at runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.logging_config import logger
from config.settings import settings

_INDEX_PATH = Path(settings.data_dir) / "processed" / "chunks" / "faiss.index"
_index: Optional[faiss.Index] = None
_chunk_uids: list[str] = []  # positional mapping: faiss row i → chunk_uid


def load_index(index_path: str | Path = _INDEX_PATH) -> faiss.Index:
    """Load FAISS index from disk into memory. Call once per experiment run."""
    global _index
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {path}. Run scripts/ingest_d2l.py first."
        )
    _index = faiss.read_index(str(path))
    logger.info("Loaded FAISS index with %d vectors from %s", _index.ntotal, path)
    return _index


def load_chunk_uids(uids_path: str | Path | None = None) -> list[str]:
    """Load the chunk_uid list that maps FAISS row indices to database chunk_uids."""
    global _chunk_uids
    path = Path(uids_path) if uids_path else _INDEX_PATH.with_suffix(".uids.txt")
    if not path.exists():
        raise FileNotFoundError(f"Chunk UID list not found at {path}.")
    _chunk_uids = path.read_text().splitlines()
    logger.info("Loaded %d chunk UIDs", len(_chunk_uids))
    return _chunk_uids


def search(
    query_vector: np.ndarray,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Retrieve top-k chunks by cosine similarity.
    Returns list of (chunk_uid, score) tuples, highest score first.
    """
    if _index is None:
        raise RuntimeError("FAISS index not loaded. Call load_index() first.")

    vec = query_vector.astype(np.float32).reshape(1, -1)
    scores, indices = _index.search(vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS padding for incomplete results
        uid = _chunk_uids[idx] if _chunk_uids else str(idx)
        results.append((uid, float(score)))
    return results
