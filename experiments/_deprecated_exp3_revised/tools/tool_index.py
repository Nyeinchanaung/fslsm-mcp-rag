"""
Builds FAISS IndexFlatIP over all 15 tool descriptions.
Descriptions include domain-bridging vocabulary (Phase 1).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from openai import OpenAI

from experiments.exp3_revised.tools.tool_registry import MCPTool, TOOL_REGISTRY, TOOL_BY_ID
from experiments.exp3_revised.config import (
    EMBED_MODEL, EMBED_DIM, TOOL_INDEX_PATH, TOOL_META_PATH
)


def _embed(texts: List[str]) -> np.ndarray:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    delays = [2, 4, 8]
    for attempt, delay in enumerate(delays, 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=60)
            vecs = np.array([r.embedding for r in resp.data], dtype="float32")
            faiss.normalize_L2(vecs)
            return vecs
        except Exception as e:
            print(f"[embed_retry] attempt {attempt} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=60)
    vecs = np.array([r.embedding for r in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


class ToolIndex:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.tools: List[MCPTool] = []

    def build(self, tools: List[MCPTool] = TOOL_REGISTRY) -> None:
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
