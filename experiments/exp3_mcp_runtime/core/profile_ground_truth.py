from __future__ import annotations

from typing import Any

from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_label, profile_to_tuple

PROFILE_ELIGIBLE_FAMILIES = {"concept_explain"}

EXPLICIT_TASK_FAMILIES = {
    "adapt_retrieved_text",
    "analogy",
    "compare",
    "concept_map",
    "derive_steps",
    "diagram",
    "external_search",
    "hands_on_exercise",
    "locate_d2l_content",
    "quiz",
    "reflective_discussion",
    "style_transfer",
    "summarize",
    "worked_example",
}

CONCEPT_EXPLAIN_PROFILE_TARGETS: dict[tuple[str, str, str, str], int] = {
    ("Active", "Sensing", "Visual", "Sequential"): 4,
    ("Active", "Sensing", "Visual", "Global"): 7,
    ("Active", "Sensing", "Verbal", "Sequential"): 1,
    ("Active", "Sensing", "Verbal", "Global"): 7,
    ("Active", "Intuitive", "Visual", "Sequential"): 4,
    ("Active", "Intuitive", "Visual", "Global"): 7,
    ("Active", "Intuitive", "Verbal", "Sequential"): 5,
    ("Active", "Intuitive", "Verbal", "Global"): 5,
    ("Reflective", "Sensing", "Visual", "Sequential"): 4,
    ("Reflective", "Sensing", "Visual", "Global"): 13,
    ("Reflective", "Sensing", "Verbal", "Sequential"): 1,
    ("Reflective", "Sensing", "Verbal", "Global"): 13,
    ("Reflective", "Intuitive", "Visual", "Sequential"): 4,
    ("Reflective", "Intuitive", "Visual", "Global"): 7,
    ("Reflective", "Intuitive", "Verbal", "Sequential"): 10,
    ("Reflective", "Intuitive", "Verbal", "Global"): 10,
}


def is_profile_eval_eligible(question_family: str) -> bool:
    return question_family in PROFILE_ELIGIBLE_FAMILIES


def profile_target_tool_id(
    *,
    question_family: str,
    target_tool_id: int,
    profile: dict[str, Any],
) -> int:
    if not is_profile_eval_eligible(question_family):
        return target_tool_id
    if question_family == "concept_explain":
        return CONCEPT_EXPLAIN_PROFILE_TARGETS.get(profile_to_tuple(profile), target_tool_id)
    return target_tool_id


def build_profile_target_tool_ids(
    *,
    question_family: str,
    target_tool_id: int,
    profiles: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        profile_to_label(profile["fslsm_vector"]): profile_target_tool_id(
            question_family=question_family,
            target_tool_id=target_tool_id,
            profile=profile["fslsm_vector"],
        )
        for profile in profiles
    }
