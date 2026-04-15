"""Unit and integration tests for Phase 3 — TutorAgent (Experiment 2).

Unit tests (no API calls): context building, engagement parsing, session result schema.
Integration tests (require OpenAI API key): full R0/R1 end-to-end smoke test.
"""

import json
import pytest

from src.tutor.tutor_agent import TutorAgent, R0_SYSTEM_PROMPT
from src.tutor.profile_agent import ProfileAgent
from src.tutor.retrieval_agent import RetrievalAgent
from src.utils.llm_client import LLMClient

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


@pytest.fixture
def tutor_agent(profile_agent, retrieval_agent):
    """TutorAgent with real LLM clients (requires OPENAI_API_KEY)."""
    tutor_client = LLMClient("gpt-4.1-mini", temperature=0.3)
    student_client = LLMClient("gpt-4.1-mini", temperature=0.0)
    return TutorAgent(
        tutor_client=tutor_client,
        student_client=student_client,
        profile_agent=profile_agent,
        retrieval_agent=retrieval_agent,
    )


@pytest.fixture
def sample_session_r0():
    return {
        "agent_id": "test_agent_001",
        "fslsm_vector": {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1},
        "question": "What is gradient descent?",
        "question_id": "test_q_001",
        "mode": "R0",
    }


@pytest.fixture
def sample_session_r1():
    return {
        "agent_id": "test_agent_002",
        "fslsm_vector": {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": 1},
        "question": "What is gradient descent?",
        "question_id": "test_q_001",
        "mode": "R1",
    }


# -------------------------------------------------------------------
# Unit tests — _build_context_prompt (no IO / API)
# -------------------------------------------------------------------

class TestBuildContextPrompt:
    def test_formats_evidence_blocks(self):
        chunks = [
            {"chunk_id": "a", "heading": "SGD", "text": "Stochastic gradient descent..."},
            {"chunk_id": "b", "heading": "Loss", "text": "The loss function measures..."},
        ]
        prompt = TutorAgent._build_context_prompt(chunks, "What is SGD?", max_chars=10000)
        assert "[Evidence 1] (SGD)" in prompt
        assert "[Evidence 2] (Loss)" in prompt
        assert "Student Question: What is SGD?" in prompt

    def test_truncates_long_chunks(self):
        chunks = [
            {"chunk_id": "a", "text": "x" * 5000},
            {"chunk_id": "b", "text": "y" * 5000},
            {"chunk_id": "c", "text": "z" * 5000},
        ]
        prompt = TutorAgent._build_context_prompt(chunks, "Q?", max_chars=8000)
        # Should not exceed max_chars for evidence portion
        assert len(prompt) < 10000  # some overhead for formatting

    def test_empty_chunks(self):
        prompt = TutorAgent._build_context_prompt([], "What is ML?", max_chars=8000)
        assert "Student Question: What is ML?" in prompt

    def test_chunks_without_heading(self):
        chunks = [{"chunk_id": "a", "text": "Some content here."}]
        prompt = TutorAgent._build_context_prompt(chunks, "Q?", max_chars=8000)
        assert "[Evidence 1]:" in prompt


# -------------------------------------------------------------------
# Unit tests — _parse_engagement_score (no IO / API)
# -------------------------------------------------------------------

class TestParseEngagementScore:
    def test_direct_integer(self):
        assert TutorAgent._parse_engagement_score("4") == 4

    def test_integer_with_whitespace(self):
        assert TutorAgent._parse_engagement_score("  3  ") == 3

    def test_integer_in_sentence(self):
        assert TutorAgent._parse_engagement_score("I would rate this a 4.") == 4

    def test_defaults_to_3_on_garbage(self):
        assert TutorAgent._parse_engagement_score("no number here") == 3

    def test_all_valid_scores(self):
        for i in range(1, 6):
            assert TutorAgent._parse_engagement_score(str(i)) == i

    def test_picks_first_valid_digit(self):
        assert TutorAgent._parse_engagement_score("Rating: 5 out of 5") == 5


# -------------------------------------------------------------------
# Unit tests — session result JSON-serializability
# -------------------------------------------------------------------

class TestSessionResultSchema:
    """Validate that mock session results are JSON-safe."""

    def test_mock_result_is_json_serializable(self):
        result = {
            "agent_id": "agent_007",
            "question_id": "q_042",
            "mode": "R1",
            "response": "Here is the explanation...",
            "system_prompt_used": "You are an expert AI Tutor...",
            "retrieved_chunk_ids": ["a", "b", "c", "d", "e"],
            "reformulated_query": "How does backprop work? [with visuals]",
            "engagement_score": 4,
            "latency_ms": 1240,
            "token_count": 312,
            "tutor_cost": 0.001,
        }
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized == result


# -------------------------------------------------------------------
# Integration tests — full pipeline (require OpenAI API key)
# -------------------------------------------------------------------

class TestTutorIntegration:
    """End-to-end smoke tests. Require OPENAI_API_KEY."""

    REQUIRED_KEYS = {
        "agent_id", "question_id", "mode", "response",
        "system_prompt_used", "retrieved_chunk_ids",
        "reformulated_query", "engagement_score",
        "latency_ms", "token_count", "tutor_cost",
    }

    def test_r0_session_end_to_end(self, tutor_agent, sample_session_r0):
        result = tutor_agent.run_session(sample_session_r0)

        # All required keys present
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

        assert result["mode"] == "R0"
        assert result["system_prompt_used"] == R0_SYSTEM_PROMPT
        assert len(result["response"]) > 10
        assert len(result["retrieved_chunk_ids"]) == 5
        assert result["engagement_score"] in {1, 2, 3, 4, 5}
        assert result["latency_ms"] > 0
        assert result["token_count"] > 0

    def test_r1_session_end_to_end(self, tutor_agent, sample_session_r1):
        result = tutor_agent.run_session(sample_session_r1)

        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

        assert result["mode"] == "R1"
        assert "Active-Sensing-Visual-Global" in result["system_prompt_used"]
        assert len(result["response"]) > 10
        assert len(result["retrieved_chunk_ids"]) == 5
        assert result["engagement_score"] in {1, 2, 3, 4, 5}
        assert "[" in result["reformulated_query"]  # directive appended

    def test_result_is_json_serializable(self, tutor_agent, sample_session_r0):
        result = tutor_agent.run_session(sample_session_r0)
        serialized = json.dumps(result)
        assert len(serialized) > 0
