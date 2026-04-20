"""
Phase 3 — Tutor Agent for Experiment 2 (FSLSM-Based Tutor Personalization).

LLM Orchestrator that wires ProfileAgent + RetrievalAgent + LLM calls to
produce the final tutor response.  Also collects engagement scores from
virtual student agents.

Pipeline per session:
  1. Profile resolution  (R1 only)
  2. Retrieval           (FAISS ± FSLSM reranking)
  3. Build LLM context   (evidence blocks + question)
  4. Call tutor LLM      (GPT-4.1 mini)
  5. Get engagement score (virtual student agent, 1-5)
  6. Return session_result
"""

from __future__ import annotations

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

from src.tutor.profile_agent import ProfileAgent
from src.tutor.retrieval_agent import RetrievalAgent
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

R0_SYSTEM_PROMPT = "You are a helpful tutor."

# Max context chars for chunk evidence (approximate — character-based heuristic).
# System prompt is passed via the `system` parameter (separate budget),
# so R0 and R1 evidence budgets should be equal.
_MAX_CONTEXT_CHARS = 16_000   # ~4000 tokens


class TutorAgent:
    """Orchestrates the PersonaRAG pipeline for a single tutoring session."""

    def __init__(
        self,
        tutor_client: LLMClient,
        student_client: LLMClient,
        profile_agent: ProfileAgent,
        retrieval_agent: RetrievalAgent,
        mode: str = "R1",
        max_tokens: int = 1000,
        student_profiles_path: str | Path = "data/fslsm/profiles.json",
    ):
        """
        Args:
            tutor_client: LLMClient for tutor generation (GPT-4.1 mini).
            student_client: LLMClient for student engagement scoring.
            profile_agent: Phase 1 ProfileAgent instance.
            retrieval_agent: Phase 2 RetrievalAgent instance.
            mode: Default mode ("R0" or "R1"), can be overridden per session.
            max_tokens: Max output tokens for tutor response.
            student_profiles_path: Path to FSLSM profiles with behavioral_instructions.
        """
        self.tutor_client = tutor_client
        self.student_client = student_client
        self.profile_agent = profile_agent
        self.retrieval_agent = retrieval_agent
        self.default_mode = mode
        self.max_tokens = max_tokens
        self._student_profiles = self._load_student_profiles(student_profiles_path)

    @staticmethod
    def _load_student_profiles(path: str | Path) -> dict[tuple, dict]:
        """Load FSLSM profiles keyed by dimension tuple for student persona lookup."""
        with open(path) as f:
            profiles = json.load(f)
        lookup = {}
        for p in profiles:
            dims = p.get("dimensions", {})
            if dims.get("act_ref") == 0:
                continue  # skip baseline
            key = (dims["act_ref"], dims["sen_int"], dims["vis_ver"], dims["seq_glo"])
            lookup[key] = p
        return lookup

    def run_session(self, session: dict) -> dict:
        """
        Execute a single tutoring session (R0 or R1).

        Args:
            session: Dict with agent_id, fslsm_vector, question,
                     question_id, and mode.

        Returns:
            session_result dict with all fields from the output contract.
        """
        mode = session.get("mode", self.default_mode)
        fslsm_vector = session["fslsm_vector"]
        question = session["question"]

        t_start = time.time()

        # Step 1 — Profile resolution
        if mode == "R1":
            plan = self.profile_agent.generate_reasoning_plan(fslsm_vector)
            system_prompt = self.profile_agent.generate_system_prompt(plan)
        else:
            plan = None
            system_prompt = R0_SYSTEM_PROMPT

        # Step 2 — Retrieval
        retrieval_result = self.retrieval_agent.retrieve(
            question=question,
            reasoning_plan=plan,
            personalized=(mode == "R1"),
        )

        # Step 3 — Build LLM context
        max_chars = _MAX_CONTEXT_CHARS
        context_prompt = self._build_context_prompt(
            retrieval_result["retrieved_chunks"], question, max_chars
        )

        # Step 4 — Call tutor LLM
        tutor_resp = self.tutor_client.chat(
            system=system_prompt,
            user=context_prompt,
            max_tokens=self.max_tokens,
        )

        latency_ms = int((time.time() - t_start) * 1000)

        # Step 5 — Get engagement score from virtual student agent
        engagement_score = self._get_engagement_score(
            tutor_resp.content, fslsm_vector
        )

        # Step 6 — Assemble result
        return {
            "agent_id": session.get("agent_id", ""),
            "question_id": session.get("question_id", ""),
            "mode": mode,
            "response": tutor_resp.content,
            "system_prompt_used": system_prompt,
            "retrieved_chunk_ids": retrieval_result["chunk_ids"],
            "reformulated_query": retrieval_result["reformulated_query"],
            "engagement_score": engagement_score,
            "latency_ms": latency_ms,
            "token_count": tutor_resp.total_tokens,
            "tutor_cost": tutor_resp.cost,
        }

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _build_context_prompt(
        chunks: list[dict], question: str, max_chars: int
    ) -> str:
        """
        Format retrieved chunks as numbered evidence blocks + student question.

        Truncates if total character count exceeds max_chars.
        """
        evidence_parts = []
        total_chars = 0
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            heading = chunk.get("heading", "")
            block = f"[Evidence {i}]"
            if heading:
                block += f" ({heading})"
            block += f": {text}"

            if total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    block = block[:remaining] + "..."
                    evidence_parts.append(block)
                break
            evidence_parts.append(block)
            total_chars += len(block)

        evidence_text = "\n\n".join(evidence_parts)
        return (
            f"Use the following evidence to answer the student's question.\n\n"
            f"{evidence_text}\n\n"
            f"Student Question: {question}"
        )

    def _get_engagement_score(
        self, response: str, fslsm_vector: dict
    ) -> int:
        """
        Call virtual student agent to rate tutor response engagement (1-5).

        Uses the student's FSLSM behavioral instructions as the persona.
        """
        key = (
            fslsm_vector["act_ref"],
            fslsm_vector["sen_int"],
            fslsm_vector["vis_ver"],
            fslsm_vector["seq_glo"],
        )
        profile = self._student_profiles.get(key, {})
        instructions = profile.get("behavioral_instructions", {})
        label = profile.get("label", "Unknown")

        # Build student persona from behavioral instructions
        persona_parts = [
            f"You are a student with the following learning style: {label}.",
        ]
        for dim, text in instructions.items():
            if text:
                persona_parts.append(text)
        student_system = " ".join(persona_parts)

        user_prompt = (
            f"A tutor gave you this response:\n\n"
            f'"""\n{response}\n"""\n\n'
            f"Rate how engaging and well-suited this response is for your "
            f"learning style on a scale of 1-5.\n"
            f"1 = completely mismatched with your learning preferences\n"
            f"2 = mostly mismatched\n"
            f"3 = neutral / partially suited\n"
            f"4 = well suited to your learning style\n"
            f"5 = perfectly aligned with your learning preferences\n\n"
            f"Respond with ONLY a single integer (1-5)."
        )

        try:
            resp = self.student_client.chat(
                system=student_system,
                user=user_prompt,
                max_tokens=5,
                temperature=0.0,
            )
            return self._parse_engagement_score(resp.content)
        except Exception as e:
            logger.warning("Engagement scoring failed: %s — defaulting to 3", e)
            return 3

    @staticmethod
    def _parse_engagement_score(text: str) -> int:
        """Extract integer 1-5 from LLM response. Defaults to 3 on failure."""
        text = text.strip()
        # Try direct int parse
        if text in ("1", "2", "3", "4", "5"):
            return int(text)
        # Try finding first digit 1-5
        match = re.search(r"[1-5]", text)
        if match:
            return int(match.group())
        logger.warning("Could not parse engagement score from: %r — defaulting to 3", text)
        return 3
