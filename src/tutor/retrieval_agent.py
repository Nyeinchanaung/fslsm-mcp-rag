"""
Phase 2 — Retrieval Agent for Experiment 2 (FSLSM-Based Tutor Personalization).

Takes a reasoning plan (Phase 1) and a raw student question, produces a
reformulated query optimised for FSLSM-aligned chunk retrieval from FAISS.

Implements the R0 vs R1 retrieval switch:
  R0 (Control)      — raw FAISS similarity, no reformulation, no reranking
  R1 (Experimental) — reformulate query + FAISS + FSLSM-aware reranking
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import settings

# ---------------------------------------------------------------------------
# Content-type tags used for FSLSM-aware reranking (Task 2.3)
# ---------------------------------------------------------------------------

# Keywords mapped to reranking tags — used for runtime heuristic tagging
_TAG_KEYWORDS: dict[str, list[str]] = {
    "visual_content": [
        "diagram", "figure", "chart", "graph", "plot", "illustration",
        "visualization", "table", "image", "fig.", "shown in",
    ],
    "diagrams": [
        "diagram", "flowchart", "flow chart", "architecture",
        "block diagram", "schematic",
    ],
    "step_by_step": [
        "step 1", "step 2", "step 3", "first,", "second,", "third,",
        "next,", "then,", "finally,", "procedure", "algorithm:",
    ],
    "sequential": [
        "sequentially", "in order", "one by one", "linearly",
    ],
    "concrete_examples": [
        "example", "for instance", "e.g.", "such as", "consider",
        "suppose", "let's say", "given that", "specifically",
    ],
    "procedures": [
        "procedure", "method", "technique", "recipe", "how to",
    ],
    "conceptual_overview": [
        "overview", "introduction", "summary", "in general",
        "broadly", "the idea", "high-level", "big picture",
    ],
    "holistic": [
        "overall", "framework", "architecture", "landscape",
    ],
    "abstract_theory": [
        "theorem", "proof", "lemma", "corollary", "formally",
        "theoretical", "mathematically", "derivation",
    ],
    "principles": [
        "principle", "fundamental", "axiom", "law", "property",
    ],
    "exercises": [
        "exercise", "problem", "practice", "try", "implement",
        "compute", "calculate", "verify",
    ],
    "interactive": [
        "interactive", "hands-on", "experiment", "try it",
    ],
    "reflective": [
        "reflect", "consider why", "think about", "analyze",
        "compare and contrast", "evaluate",
    ],
    "analytical": [
        "analysis", "analytical", "reasoning", "critical",
    ],
    "verbal_text": [
        "explain", "describe", "discuss", "narrative",
        "definition", "meaning", "interpretation",
    ],
    "definitions": [
        "definition", "defined as", "refers to", "is called",
        "known as", "terminology",
    ],
    "passive_reading": [
        "read the following", "note that", "observe that",
    ],
}


def _tag_chunk(text: str) -> list[str]:
    """Assign content-type tags to a chunk based on keyword matching."""
    text_lower = text.lower()
    tags = []
    for tag, keywords in _TAG_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# RetrievalAgent
# ---------------------------------------------------------------------------

class RetrievalAgent:
    """FSLSM-aware retrieval with R0/R1 switch for Experiment 2."""

    def __init__(
        self,
        chunks_path: str | Path = "d2l/output/d2l_corpus_chunks.json",
        faiss_index_path: str | Path | None = None,
        faiss_uids_path: str | Path | None = None,
    ):
        """
        Args:
            chunks_path: Path to the D2L corpus chunks JSON.
            faiss_index_path: Path to faiss.index. Defaults to settings.
            faiss_uids_path: Path to faiss.uids.txt. Defaults to sibling of index.
        """
        self._chunks_path = Path(chunks_path)
        self._faiss_index_path = Path(faiss_index_path) if faiss_index_path else (
            Path(settings.data_dir) / "processed" / "chunks" / "faiss.index"
        )
        self._faiss_uids_path = Path(faiss_uids_path) if faiss_uids_path else (
            self._faiss_index_path.with_suffix(".uids.txt")
        )

        # Lazy-loaded resources
        self._index = None
        self._uids: list[str] = []
        self._chunk_lookup: dict[str, dict] = {}
        self._embed_client = None

    # -- lazy loaders -------------------------------------------------------

    def _ensure_index(self):
        if self._index is not None:
            return
        import faiss
        self._index = faiss.read_index(str(self._faiss_index_path))
        self._uids = self._faiss_uids_path.read_text().splitlines()

    def _ensure_chunks(self):
        if self._chunk_lookup:
            return
        with open(self._chunks_path) as f:
            chunks = json.load(f)
        self._chunk_lookup = {c["chunk_id"]: c for c in chunks}

    def _ensure_embedder(self):
        if self._embed_client is not None:
            return
        from src.utils.embedding_client import EmbeddingClient
        self._embed_client = EmbeddingClient()

    # -- public API ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
        reasoning_plan: Optional[dict] = None,
        personalized: bool = True,
        k: int = 5,
    ) -> dict:
        """
        Main entry point — R0/R1 switch.

        Args:
            question: Raw student question.
            reasoning_plan: Output of ProfileAgent.generate_reasoning_plan().
                            Required when personalized=True.
            personalized: False for R0 (control), True for R1 (experimental).
            k: Number of chunks to retrieve.

        Returns:
            Dict with reformulated_query, retrieved_chunks, chunk_ids,
            reranked flag, and scores.
        """
        if personalized and reasoning_plan is None:
            raise ValueError("reasoning_plan is required when personalized=True")

        self._ensure_index()
        self._ensure_chunks()
        self._ensure_embedder()

        if personalized:
            query = self.reformulate_query(question, reasoning_plan)
        else:
            query = question

        chunks_with_scores = self._faiss_search(query, k)

        if personalized:
            chunks_with_scores = self.rerank_chunks(chunks_with_scores, reasoning_plan)

        retrieved_chunks = []
        chunk_ids = []
        scores = []
        for chunk, score in chunks_with_scores:
            retrieved_chunks.append(chunk)
            chunk_ids.append(chunk["chunk_id"])
            scores.append(round(score, 4))

        return {
            "reformulated_query": query,
            "retrieved_chunks": retrieved_chunks,
            "chunk_ids": chunk_ids,
            "reranked": personalized,
            "scores": scores,
        }

    def reformulate_query(self, question: str, reasoning_plan: dict) -> str:
        """
        Append retrieval_directive to the raw question for richer embedding.

        Example:
          "How does backpropagation work?" ->
          "How does backpropagation work? [Retrieve chunks containing diagrams...]"
        """
        directive = reasoning_plan.get("retrieval_directive", "")
        if directive:
            return f"{question} [{directive}]"
        return question

    def rerank_chunks(
        self,
        chunks_with_scores: list[tuple[dict, float]],
        reasoning_plan: dict,
        boost: float = 0.05,
    ) -> list[tuple[dict, float]]:
        """
        FSLSM-aware reranking: boost/demote chunks based on content tags.

        Uses runtime keyword heuristics to tag chunk content, then adjusts
        cosine similarity scores by +/- boost per matching tag.
        """
        bias_tags = set(reasoning_plan.get("reranking_bias", []))
        depri_tags = set(reasoning_plan.get("deprioritize", []))

        reranked = []
        for chunk, score in chunks_with_scores:
            chunk_tags = set(_tag_chunk(chunk.get("text", "")))
            boost_count = len(chunk_tags & bias_tags)
            demote_count = len(chunk_tags & depri_tags)
            adjusted = score + (boost_count * boost) - (demote_count * boost)
            reranked.append((chunk, adjusted))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    # -- internal -----------------------------------------------------------

    def _faiss_search(self, query: str, k: int) -> list[tuple[dict, float]]:
        """Embed query, search FAISS, return (chunk_dict, score) pairs."""
        query_vec = self._embed_client.embed_one(query).reshape(1, -1)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            uid = self._uids[idx]
            chunk = self._chunk_lookup.get(uid, {"chunk_id": uid, "text": ""})
            results.append((dict(chunk), float(score)))
        return results
