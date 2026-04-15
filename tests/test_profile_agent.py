"""Unit tests for Phase 1 — ProfileAgent (Experiment 2)."""

import re
from itertools import product
from pathlib import Path

import pytest

from src.tutor.profile_agent import FSLSM_STYLE_MAP, ProfileAgent

PROFILES_PATH = Path("data/fslsm/profiles.json")
ALL_VECTORS = [
    {"act_ref": ar, "sen_int": si, "vis_ver": vv, "seq_glo": sg}
    for ar, si, vv, sg in product((-1, 1), repeat=4)
]

REQUIRED_PLAN_KEYS = {
    "profile_code",
    "style_label",
    "retrieval_directive",
    "generation_directive",
    "reranking_bias",
    "deprioritize",
}


@pytest.fixture
def agent():
    return ProfileAgent(profiles_path=PROFILES_PATH)


@pytest.fixture
def agent_no_graf():
    return ProfileAgent(profiles_path=None)


# -------------------------------------------------------------------
# FSLSM_STYLE_MAP tests
# -------------------------------------------------------------------

class TestFSLSMStyleMap:
    def test_has_16_entries(self):
        assert len(FSLSM_STYLE_MAP) == 16

    def test_all_binary_combos_present(self):
        for vec in ALL_VECTORS:
            key = (vec["act_ref"], vec["sen_int"], vec["vis_ver"], vec["seq_glo"])
            assert key in FSLSM_STYLE_MAP, f"Missing key {key}"

    def test_each_entry_has_required_keys(self):
        for key, entry in FSLSM_STYLE_MAP.items():
            for rk in REQUIRED_PLAN_KEYS:
                assert rk in entry, f"Key {key} missing field '{rk}'"

    def test_directives_non_empty(self):
        for key, entry in FSLSM_STYLE_MAP.items():
            assert len(entry["retrieval_directive"]) > 20, f"{key}: retrieval_directive too short"
            assert len(entry["generation_directive"]) > 20, f"{key}: generation_directive too short"

    def test_reranking_bias_is_list(self):
        for key, entry in FSLSM_STYLE_MAP.items():
            assert isinstance(entry["reranking_bias"], list)
            assert len(entry["reranking_bias"]) >= 4


# -------------------------------------------------------------------
# ProfileAgent.generate_reasoning_plan tests
# -------------------------------------------------------------------

class TestGenerateReasoningPlan:
    def test_all_16_profiles_produce_valid_plan(self, agent):
        for vec in ALL_VECTORS:
            plan = agent.generate_reasoning_plan(vec)
            for rk in REQUIRED_PLAN_KEYS:
                assert rk in plan, f"Plan for {vec} missing '{rk}'"
            assert plan["style_descriptor_graf"], f"Missing graf descriptor for {vec}"

    def test_plan_without_graf(self, agent_no_graf):
        vec = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
        plan = agent_no_graf.generate_reasoning_plan(vec)
        assert "style_descriptor_graf" not in plan

    def test_active_contains_hands_on(self, agent):
        for vec in ALL_VECTORS:
            if vec["act_ref"] == -1:
                plan = agent.generate_reasoning_plan(vec)
                combined = plan["retrieval_directive"] + " " + plan["generation_directive"]
                assert "hands-on" in combined.lower() or "interactive" in combined.lower(), (
                    f"Active profile {vec} should mention hands-on/interactive"
                )

    def test_visual_contains_diagram(self, agent):
        for vec in ALL_VECTORS:
            if vec["vis_ver"] == -1:
                plan = agent.generate_reasoning_plan(vec)
                combined = plan["retrieval_directive"] + " " + plan["generation_directive"]
                assert "diagram" in combined.lower() or "visual" in combined.lower(), (
                    f"Visual profile {vec} should mention diagram/visual"
                )

    def test_sequential_contains_step(self, agent):
        for vec in ALL_VECTORS:
            if vec["seq_glo"] == -1:
                plan = agent.generate_reasoning_plan(vec)
                combined = plan["retrieval_directive"] + " " + plan["generation_directive"]
                assert "step" in combined.lower() or "sequential" in combined.lower(), (
                    f"Sequential profile {vec} should mention step/sequential"
                )

    def test_sensing_contains_concrete(self, agent):
        for vec in ALL_VECTORS:
            if vec["sen_int"] == -1:
                plan = agent.generate_reasoning_plan(vec)
                combined = plan["retrieval_directive"] + " " + plan["generation_directive"]
                assert "concrete" in combined.lower() or "fact" in combined.lower(), (
                    f"Sensing profile {vec} should mention concrete/fact"
                )


# -------------------------------------------------------------------
# Validation tests
# -------------------------------------------------------------------

class TestValidation:
    def test_missing_dimension(self, agent):
        with pytest.raises(ValueError, match="missing"):
            agent.generate_reasoning_plan({"act_ref": -1, "sen_int": -1, "vis_ver": -1})

    def test_extra_dimension(self, agent):
        with pytest.raises(ValueError, match="unexpected"):
            agent.generate_reasoning_plan(
                {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1, "extra": 1}
            )

    def test_invalid_value_zero(self, agent):
        with pytest.raises(ValueError, match="must be -1 or \\+1"):
            agent.generate_reasoning_plan(
                {"act_ref": 0, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
            )

    def test_invalid_value_string(self, agent):
        with pytest.raises(ValueError):
            agent.generate_reasoning_plan(
                {"act_ref": "active", "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
            )

    def test_not_a_dict(self, agent):
        with pytest.raises(ValueError, match="must be a dict"):
            agent.generate_reasoning_plan([1, -1, -1, -1])


# -------------------------------------------------------------------
# generate_system_prompt tests
# -------------------------------------------------------------------

class TestGenerateSystemPrompt:
    def test_prompt_references_style_label(self, agent):
        vec = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": 1}
        plan = agent.generate_reasoning_plan(vec)
        prompt = agent.generate_system_prompt(plan)
        assert "Active-Sensing-Visual-Global" in prompt

    def test_prompt_at_least_3_sentences(self, agent):
        for vec in ALL_VECTORS:
            plan = agent.generate_reasoning_plan(vec)
            prompt = agent.generate_system_prompt(plan)
            sentences = [s.strip() for s in re.split(r'[.!?]+', prompt) if s.strip()]
            assert len(sentences) >= 3, (
                f"Prompt for {plan['style_label']} has only {len(sentences)} sentences"
            )

    def test_prompt_mentions_tutor_role(self, agent):
        vec = {"act_ref": 1, "sen_int": 1, "vis_ver": 1, "seq_glo": 1}
        plan = agent.generate_reasoning_plan(vec)
        prompt = agent.generate_system_prompt(plan)
        assert "tutor" in prompt.lower()

    def test_prompt_includes_generation_directive(self, agent):
        vec = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
        plan = agent.generate_reasoning_plan(vec)
        prompt = agent.generate_system_prompt(plan)
        assert "hands-on" in prompt.lower() or "experimentation" in prompt.lower()
        assert "visual" in prompt.lower() or "diagram" in prompt.lower()
