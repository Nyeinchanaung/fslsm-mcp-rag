from __future__ import annotations

import json
import re
from typing import Any

from experiments.exp3_mcp_runtime.config import CHUNKS_PATH


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class D2LRetriever:
    def __init__(self) -> None:
        self._chunks = json.loads(CHUNKS_PATH.read_text())

    def retrieve(self, query: str, k: int = 5) -> dict[str, Any]:
        query_tokens = _tokenize(query)
        scored = []
        for chunk in self._chunks:
            chunk_tokens = _tokenize(chunk.get("text", ""))
            score = len(query_tokens & chunk_tokens)
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for _, chunk in scored[:k]] or self._chunks[:k]
        evidence = [
            {
                "chunk_id": chunk["chunk_id"],
                "chapter": chunk.get("chapter"),
                "heading": chunk.get("heading"),
                "text": chunk.get("text", ""),
            }
            for chunk in top_chunks
        ]
        return {
            "chunk_ids": [chunk["chunk_id"] for chunk in top_chunks],
            "evidence": evidence,
            "combined_text": "\n\n".join(chunk["text"] for chunk in evidence if chunk["text"]),
        }
