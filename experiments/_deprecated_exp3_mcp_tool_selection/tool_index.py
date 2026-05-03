"""FAISS IndexFlatIP over the 15 tool descriptions.

Uses OpenAI `text-embedding-3-small` for embeddings (1536-d, normalised so that
inner-product == cosine). Mirrors the load pattern at
`src/tutor/retrieval_agent.py:126-149` for consistency with Exp1/Exp2.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from experiments.exp3_mcp_tool_selection.tool_registry import MCPTool, TOOL_REGISTRY


def _embed(texts: list[str], model: str) -> np.ndarray:
    """Embed a list of texts with OpenAI; returns L2-normalised float32 array."""
    from openai import OpenAI
    client = OpenAI()  # picks up OPENAI_API_KEY from env
    resp = client.embeddings.create(model=model, input=texts)
    vecs = np.array([r.embedding for r in resp.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


class ToolIndex:
    """Wraps a FAISS index over tool descriptions plus tool metadata."""

    def __init__(self, embed_model: str = "text-embedding-3-small"):
        self.embed_model = embed_model
        self._index: faiss.Index | None = None
        self._ids: list[int] = []  # tool_id at each FAISS row

    # -- build / save / load --------------------------------------------- #

    def build(self) -> None:
        """One-time: embed all tool descriptions and populate the FAISS index."""
        texts = [f"{t.name}. {t.description}" for t in TOOL_REGISTRY]
        vecs = _embed(texts, self.embed_model)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self._index = index
        self._ids = [t.tool_id for t in TOOL_REGISTRY]

    def save(self, index_path: Path | str, meta_path: Path | str) -> None:
        assert self._index is not None, "call build() first"
        faiss.write_index(self._index, str(index_path))
        Path(meta_path).write_text(json.dumps({
            "embed_model": self.embed_model,
            "tool_ids": self._ids,
        }, indent=2))

    def load(self, index_path: Path | str, meta_path: Path | str) -> None:
        self._index = faiss.read_index(str(index_path))
        meta = json.loads(Path(meta_path).read_text())
        self.embed_model = meta["embed_model"]
        self._ids = meta["tool_ids"]

    # -- query ------------------------------------------------------------ #

    def retrieve(self, query: str, k: int = 1) -> list[tuple[MCPTool, float]]:
        """Return top-k (tool, cosine_score) for a query string."""
        assert self._index is not None, "call build() or load() first"
        vec = _embed([query], self.embed_model)
        scores, idxs = self._index.search(vec, k)
        out: list[tuple[MCPTool, float]] = []
        for score, row in zip(scores[0], idxs[0]):
            tool_id = self._ids[row]
            tool = next(t for t in TOOL_REGISTRY if t.tool_id == tool_id)
            out.append((tool, float(score)))
        return out
