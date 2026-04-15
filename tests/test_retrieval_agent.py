"""Unit tests for Phase 2 — RetrievalAgent (Experiment 2).

Tests are split into:
  - Unit tests (no FAISS/embedding needed): reformulation, reranking, tagging
  - Integration tests (need FAISS index + embedding model): full retrieve()
"""

import pytest

from src.tutor.profile_agent import ProfileAgent
from src.tutor.retrieval_agent import RetrievalAgent, _tag_chunk

PROFILES_PATH = "data/fslsm/profiles.json"


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def profile_agent():
    return ProfileAgent(profiles_path=PROFILES_PATH)


@pytest.fixture
def retrieval_agent():
    return RetrievalAgent()


# -------------------------------------------------------------------
# _tag_chunk tests (pure unit tests — no IO)
# -------------------------------------------------------------------

class TestTagChunk:
    def test_visual_tags(self):
        text = "As shown in the diagram below, the architecture of a neural network..."
        tags = _tag_chunk(text)
        assert "visual_content" in tags
        assert "diagrams" in tags

    def test_step_by_step_tags(self):
        text = "Step 1: Initialize weights. Step 2: Forward pass. Step 3: Compute loss."
        tags = _tag_chunk(text)
        assert "step_by_step" in tags

    def test_concrete_examples_tags(self):
        text = "For example, consider a neural network with 3 layers."
        tags = _tag_chunk(text)
        assert "concrete_examples" in tags

    def test_abstract_theory_tags(self):
        text = "Theorem 3.1: Formally, the gradient can be derived mathematically as follows."
        tags = _tag_chunk(text)
        assert "abstract_theory" in tags

    def test_overview_tags(self):
        text = "This section provides a high-level overview of the training process."
        tags = _tag_chunk(text)
        assert "conceptual_overview" in tags

    def test_empty_text(self):
        assert _tag_chunk("") == []

    def test_no_matching_tags(self):
        tags = _tag_chunk("Hello world")
        assert tags == []


# -------------------------------------------------------------------
# reformulate_query tests (no IO)
# -------------------------------------------------------------------

class TestReformulateQuery:
    def test_appends_directive(self, retrieval_agent):
        plan = {
            "retrieval_directive": "Retrieve chunks containing diagrams and visual walkthroughs."
        }
        result = retrieval_agent.reformulate_query("How does backprop work?", plan)
        assert "How does backprop work?" in result
        assert "diagrams" in result
        assert result.startswith("How does backprop work? [")

    def test_empty_directive(self, retrieval_agent):
        plan = {"retrieval_directive": ""}
        result = retrieval_agent.reformulate_query("What is SGD?", plan)
        assert result == "What is SGD?"

    def test_missing_directive_key(self, retrieval_agent):
        result = retrieval_agent.reformulate_query("What is SGD?", {})
        assert result == "What is SGD?"


# -------------------------------------------------------------------
# rerank_chunks tests (no IO)
# -------------------------------------------------------------------

class TestRerankChunks:
    def _make_chunks(self):
        """Create mock chunks with known content for tag matching."""
        return [
            ({"chunk_id": "a", "text": "Theorem 3.1: Formally, the proof shows..."}, 0.90),
            ({"chunk_id": "b", "text": "As shown in the diagram, the architecture..."}, 0.88),
            ({"chunk_id": "c", "text": "Step 1: Initialize. Step 2: Forward pass."}, 0.86),
            ({"chunk_id": "d", "text": "For example, consider a simple case."}, 0.84),
            ({"chunk_id": "e", "text": "This section provides a high-level overview."}, 0.82),
        ]

    def test_visual_learner_boosts_diagram_chunk(self, retrieval_agent):
        chunks = self._make_chunks()
        plan = {
            "reranking_bias": ["visual_content", "diagrams", "concrete_examples"],
            "deprioritize": ["abstract_theory"],
        }
        reranked = retrieval_agent.rerank_chunks(chunks, plan)
        ids = [c["chunk_id"] for c, _ in reranked]
        # Chunk "b" (diagram) should be boosted; chunk "a" (theorem) should be demoted
        assert ids.index("b") < ids.index("a")

    def test_scores_adjusted(self, retrieval_agent):
        chunks = self._make_chunks()
        plan = {
            "reranking_bias": ["visual_content", "diagrams"],
            "deprioritize": [],
        }
        reranked = retrieval_agent.rerank_chunks(chunks, plan)
        # Chunk "b" has both visual_content and diagrams tags -> +0.10
        b_score = next(s for c, s in reranked if c["chunk_id"] == "b")
        assert b_score > 0.88  # original was 0.88, should be boosted

    def test_empty_bias_no_change_order(self, retrieval_agent):
        chunks = self._make_chunks()
        plan = {"reranking_bias": [], "deprioritize": []}
        reranked = retrieval_agent.rerank_chunks(chunks, plan)
        ids = [c["chunk_id"] for c, _ in reranked]
        assert ids == ["a", "b", "c", "d", "e"]  # original order preserved

    def test_returns_same_count(self, retrieval_agent):
        chunks = self._make_chunks()
        plan = {"reranking_bias": ["visual_content"], "deprioritize": ["abstract_theory"]}
        reranked = retrieval_agent.rerank_chunks(chunks, plan)
        assert len(reranked) == len(chunks)


# -------------------------------------------------------------------
# Integration tests — full retrieve() pipeline (requires FAISS + embedder)
# -------------------------------------------------------------------

class TestRetrieveIntegration:
    """These tests load the real FAISS index and embedding model."""

    def test_r0_retrieval_returns_valid_schema(self, retrieval_agent):
        result = retrieval_agent.retrieve(
            question="How does backpropagation work?",
            personalized=False,
            k=5,
        )
        assert "reformulated_query" in result
        assert "retrieved_chunks" in result
        assert "chunk_ids" in result
        assert "reranked" in result
        assert "scores" in result
        assert result["reranked"] is False
        assert len(result["chunk_ids"]) == 5
        assert len(result["scores"]) == 5

    def test_r1_retrieval_returns_valid_schema(self, retrieval_agent, profile_agent):
        vec = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": 1}
        plan = profile_agent.generate_reasoning_plan(vec)
        result = retrieval_agent.retrieve(
            question="How does backpropagation work?",
            reasoning_plan=plan,
            personalized=True,
            k=5,
        )
        assert result["reranked"] is True
        assert len(result["chunk_ids"]) == 5
        assert "[" in result["reformulated_query"]  # directive appended

    def test_r0_and_r1_may_differ_in_ordering(self, retrieval_agent, profile_agent):
        question = "Explain gradient descent optimization"
        r0 = retrieval_agent.retrieve(question, personalized=False, k=5)

        # Visual-Global learner
        vec = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": 1}
        plan = profile_agent.generate_reasoning_plan(vec)
        r1 = retrieval_agent.retrieve(question, reasoning_plan=plan, personalized=True, k=5)

        # R1 scores may differ due to reranking (not guaranteed to differ in order
        # for every query, but scores should be adjusted)
        assert r0["scores"] != r1["scores"] or r0["chunk_ids"] != r1["chunk_ids"]

    def test_cr5_computable(self, retrieval_agent):
        result = retrieval_agent.retrieve(
            question="How does backpropagation work?",
            personalized=False,
            k=5,
        )
        gold_ids = ["fake_gold_1", "fake_gold_2", result["chunk_ids"][0]]
        retrieved = set(result["chunk_ids"])
        gold = set(gold_ids)
        cr5 = len(retrieved & gold) / len(gold) if gold else 0.0
        assert 0.0 <= cr5 <= 1.0
        assert cr5 > 0  # at least one overlap (we inserted one)

    def test_r1_requires_reasoning_plan(self, retrieval_agent):
        with pytest.raises(ValueError, match="reasoning_plan is required"):
            retrieval_agent.retrieve("What is SGD?", personalized=True)
