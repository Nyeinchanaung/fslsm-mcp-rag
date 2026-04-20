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
        decompose_client=None,
    ):
        """
        Args:
            chunks_path: Path to the D2L corpus chunks JSON.
            faiss_index_path: Path to faiss.index. Defaults to settings.
            faiss_uids_path: Path to faiss.uids.txt. Defaults to sibling of index.
            decompose_client: LLMClient for query decomposition (multi-hop).
                              If None, multi-query retrieval is disabled.
        """
        self._chunks_path = Path(chunks_path)
        self._faiss_index_path = Path(faiss_index_path) if faiss_index_path else (
            Path(settings.data_dir) / "processed" / "chunks" / "faiss.index"
        )
        self._faiss_uids_path = Path(faiss_uids_path) if faiss_uids_path else (
            self._faiss_index_path.with_suffix(".uids.txt")
        )

        self._decompose_client = decompose_client

        # Lazy-loaded resources
        self._index = None
        self._uids: list[str] = []
        self._chunk_lookup: dict[str, dict] = {}
        self._embed_client = None
        self._bm25 = None
        self._bm25_ids: list[str] = []

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

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi
        self._ensure_chunks()
        corpus = [c.get("text", "").lower().split() for c in self._chunk_lookup.values()]
        self._bm25_ids = list(self._chunk_lookup.keys())
        self._bm25 = BM25Okapi(corpus)

    # -- public API ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
        reasoning_plan: Optional[dict] = None,
        personalized: bool = True,
        k: int = 10,
    ) -> dict:
        """
        Main entry point — R0/R1 switch.

        Multi-hop questions span several topics, so a single query often
        misses gold chunks (CR@5 ≈ 0.10). When a decompose_client is
        available, the question is split into sub-queries, each searched
        independently, and results are merged (union-then-rank).  This
        applies to *both* R0 and R1 since it improves base retrieval.

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
        self._ensure_bm25()

        # --- Hybrid multi-query retrieval ---
        sub_queries = self._decompose_question(question)
        per_query_k = max(k, 10)  # fetch enough per sub-query
        merged = self._hybrid_search(sub_queries, per_query_k)

        if personalized:
            # R1 augmentation: add reformulated query to broaden coverage.
            # The retrieval_directive adds FSLSM-relevant keywords that may
            # surface additional gold chunks matching both topic AND style.
            reformulated = self.reformulate_query(question, reasoning_plan)
            if reformulated != question:
                aug_results = self._hybrid_search([reformulated], per_query_k)
                seen: dict[str, tuple[dict, float]] = {}
                for chunk, score in merged + aug_results:
                    cid = chunk["chunk_id"]
                    if cid not in seen or score > seen[cid][1]:
                        seen[cid] = (chunk, score)
                merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)

        chunks_with_scores = merged[:k]

        retrieved_chunks = []
        chunk_ids = []
        scores = []
        for chunk, score in chunks_with_scores:
            retrieved_chunks.append(chunk)
            chunk_ids.append(chunk["chunk_id"])
            scores.append(round(score, 4))

        return {
            "reformulated_query": " | ".join(sub_queries),
            "retrieved_chunks": retrieved_chunks,
            "chunk_ids": chunk_ids,
            "reranked": personalized,
            "scores": scores,
        }

    # -- query decomposition ------------------------------------------------

    _DECOMPOSE_PROMPT = (
        "Break the following multi-hop question into 2-4 short, independent "
        "search queries. Each query should target ONE specific concept or "
        "topic mentioned in the question. Return ONLY the queries, one per "
        "line, no numbering or bullets.\n\nQuestion: {question}"
    )

    def _decompose_question(self, question: str) -> list[str]:
        """
        Split a multi-hop question into sub-queries for better retrieval.

        Falls back to [question] if no decompose_client is available or
        if the LLM call fails.
        """
        if self._decompose_client is None:
            return [question]
        try:
            resp = self._decompose_client.chat(
                system="You are a search query generator.",
                user=self._DECOMPOSE_PROMPT.format(question=question),
                max_tokens=200,
                temperature=0.0,
            )
            lines = [ln.strip() for ln in resp.content.strip().splitlines() if ln.strip()]
            # Always include the original question as the first sub-query
            # so we don't lose the holistic match.
            sub_queries = [question] + lines[:4]
            return sub_queries
        except Exception:
            return [question]

    def _multi_query_search(
        self,
        queries: list[str],
        per_query_k: int,
    ) -> list[tuple[dict, float]]:
        """
        Search FAISS for each sub-query, merge results by max-score
        per chunk, return sorted descending.
        """
        best: dict[str, tuple[dict, float]] = {}
        for q in queries:
            results = self._faiss_search(q, per_query_k)
            for chunk, score in results:
                cid = chunk["chunk_id"]
                if cid not in best or score > best[cid][1]:
                    best[cid] = (chunk, score)
        merged = sorted(best.values(), key=lambda x: x[1], reverse=True)
        return merged

    def _bm25_search(self, query: str, k: int) -> list[tuple[dict, float]]:
        """BM25 sparse search. Returns (chunk_dict, score) pairs."""
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top_k = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_k:
            if scores[idx] > 0:
                uid = self._bm25_ids[idx]
                chunk = self._chunk_lookup.get(uid, {"chunk_id": uid, "text": ""})
                results.append((dict(chunk), float(scores[idx])))
        return results

    def _hybrid_search(
        self,
        queries: list[str],
        per_query_k: int,
    ) -> list[tuple[dict, float]]:
        """
        Hybrid BM25 + FAISS with Reciprocal Rank Fusion (RRF).

        For each sub-query, retrieves from both dense (FAISS) and sparse
        (BM25) indexes, then fuses rankings using RRF scores.
        """
        rrf_k = 60  # RRF constant
        chunk_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for q in queries:
            # Dense results
            dense_results = self._faiss_search(q, per_query_k)
            for rank, (chunk, _score) in enumerate(dense_results):
                cid = chunk["chunk_id"]
                chunk_scores[cid] = chunk_scores.get(cid, 0) + 1.0 / (rrf_k + rank + 1)
                chunk_data[cid] = chunk

            # Sparse results
            bm25_results = self._bm25_search(q, per_query_k)
            for rank, (chunk, _score) in enumerate(bm25_results):
                cid = chunk["chunk_id"]
                chunk_scores[cid] = chunk_scores.get(cid, 0) + 1.0 / (rrf_k + rank + 1)
                chunk_data[cid] = chunk

        # Sort by RRF score descending
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_data[cid], score) for cid, score in ranked]

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
        boost: float = 0.002,
        max_adjustment: float = 0.004,
    ) -> list[tuple[dict, float]]:
        """
        FSLSM-aware reranking: boost/demote chunks based on content tags.

        Uses runtime keyword heuristics to tag chunk content, then adjusts
        cosine similarity scores by +/- boost per matching tag, capped at
        ±max_adjustment total so a single chunk can't swing too far.
        """
        bias_tags = set(reasoning_plan.get("reranking_bias", []))
        depri_tags = set(reasoning_plan.get("deprioritize", []))

        reranked = []
        for chunk, score in chunks_with_scores:
            chunk_tags = set(_tag_chunk(chunk.get("text", "")))
            boost_count = len(chunk_tags & bias_tags)
            demote_count = len(chunk_tags & depri_tags)
            raw_adj = (boost_count * boost) - (demote_count * boost)
            capped_adj = max(-max_adjustment, min(max_adjustment, raw_adj))
            adjusted = score + capped_adj
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
