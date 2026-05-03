"""Builds a FAISS index over the 15 tool descriptions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
import re
import hashlib

from experiments.exp3_mcp_runtime.config import EMBED_DIM, TOOL_INDEX_PATH, TOOL_META_PATH
from experiments.exp3_mcp_runtime.tools.tool_registry import MCPTool, TOOL_BY_ID, TOOL_REGISTRY


def _stable_bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % EMBED_DIM


def _embed(texts: List[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), EMBED_DIM), dtype="float32")
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vectors[row, _stable_bucket(token)] += 1.0
    faiss.normalize_L2(vectors)
    return vectors


class ToolIndex:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.tools: List[MCPTool] = []

    def build(self, tools: List[MCPTool] = TOOL_REGISTRY) -> None:
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.tools = tools
        descs = [t.description for t in tools]
        print(f"[ToolIndex] Embedding {len(descs)} tool descriptions...")
        vecs = _embed(descs)
        self.index.add(vecs)
        print(f"[ToolIndex] Built. Vectors: {self.index.ntotal}")

    def save(
        self,
        idx_path: str | Path = TOOL_INDEX_PATH,
        meta_path: str | Path = TOOL_META_PATH,
    ) -> None:
        Path(idx_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(idx_path))
        meta = [{"tool_id": t.tool_id, "name": t.name} for t in self.tools]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ToolIndex] Saved → {idx_path}")

    def load(
        self,
        idx_path: str | Path = TOOL_INDEX_PATH,
        meta_path: str | Path = TOOL_META_PATH,
    ) -> None:
        self.index = faiss.read_index(str(idx_path))
        if self.index.d != EMBED_DIM:
            raise ValueError(
                f"Tool index dimension mismatch: expected {EMBED_DIM}, found {self.index.d}. "
                "Rebuild the index with scripts/01_build_tool_index.py."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        self.tools = [TOOL_BY_ID[m["tool_id"]] for m in meta]
        print(f"[ToolIndex] Loaded {self.index.ntotal} vectors")

    def retrieve(self, query: str, k: int = 1) -> List[Tuple[MCPTool, float]]:
        vec = _embed([query])
        scores, indices = self.index.search(vec, k)
        return [
            (self.tools[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]


if __name__ == "__main__":
    idx = ToolIndex()
    idx.build()
    idx.save()

    tests = [
        "Can you draw a computation graph for backpropagation?",
        "Walk me through gradient descent step by step.",
        "Compare ResNet and VGG architectures.",
        "Quiz me on batch normalization.",
        "Summarize the attention mechanism chapter.",
        "What are the latest transformer developments?",
    ]
    print("\n── Retrieval sanity check ──")
    for q in tests:
        hits = idx.retrieve(q, k=3)
        print(f"\n  Q: {q}")
        for i, (tool, score) in enumerate(hits):
            print(f"    {i+1}. [{tool.tool_id}] {tool.name:<35} score={score:.3f}")
