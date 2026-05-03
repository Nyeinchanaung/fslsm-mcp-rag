"""
Ground truth optimal tool assignment.
Uses (question_type × profile) to assign optimal_tool_id.
Covers all 15 tools across 4 R2a question types + 4 R2b question types × 16 profiles.
"""
import json
from functools import lru_cache
from typing import Any

from experiments.exp3_mcp_runtime.config import CORE_ANSWER_KEY_PATH
from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_label
from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_tuple

# (question_type, act, sns, vis, seq) → tool_id
# All 15 tools are reachable through this mapping.

GROUND_TRUTH_MAP_FULL = {

    # ── explain_relationship (21 questions) ────────────────────
    ("explain_relationship", "Active",     "Sensing",    "Visual",  "Sequential"): 4,
    ("explain_relationship", "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("explain_relationship", "Active",     "Sensing",    "Verbal",  "Sequential"): 1,
    ("explain_relationship", "Active",     "Sensing",    "Verbal",  "Global"):     7,
    ("explain_relationship", "Active",     "Intuitive",  "Visual",  "Sequential"): 4,
    ("explain_relationship", "Active",     "Intuitive",  "Visual",  "Global"):     7,
    ("explain_relationship", "Active",     "Intuitive",  "Verbal",  "Sequential"): 5,
    ("explain_relationship", "Active",     "Intuitive",  "Verbal",  "Global"):     5,
    ("explain_relationship", "Reflective", "Sensing",    "Visual",  "Sequential"): 4,
    ("explain_relationship", "Reflective", "Sensing",    "Visual",  "Global"):     13,
    ("explain_relationship", "Reflective", "Sensing",    "Verbal",  "Sequential"): 1,
    ("explain_relationship", "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("explain_relationship", "Reflective", "Intuitive",  "Visual",  "Sequential"): 4,
    ("explain_relationship", "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("explain_relationship", "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("explain_relationship", "Reflective", "Intuitive",  "Verbal",  "Global"):     10,

    # ── synthesize_workflow (20 questions) ─────────────────────
    ("synthesize_workflow",  "Active",     "Sensing",    "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Active",     "Sensing",    "Visual",  "Global"):     11,
    ("synthesize_workflow",  "Active",     "Sensing",    "Verbal",  "Sequential"): 2,
    ("synthesize_workflow",  "Active",     "Sensing",    "Verbal",  "Global"):     3,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Visual",  "Global"):     11,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Verbal",  "Sequential"): 2,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Verbal",  "Global"):     5,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Visual",  "Global"):     7,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Verbal",  "Sequential"): 3,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Verbal",  "Global"):     10,

    # ── trace_evolution (16 questions) ─────────────────────────
    ("trace_evolution",      "Active",     "Sensing",    "Visual",  "Sequential"): 2,
    ("trace_evolution",      "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("trace_evolution",      "Active",     "Sensing",    "Verbal",  "Sequential"): 2,
    ("trace_evolution",      "Active",     "Sensing",    "Verbal",  "Global"):     1,
    ("trace_evolution",      "Active",     "Intuitive",  "Visual",  "Sequential"): 4,
    ("trace_evolution",      "Active",     "Intuitive",  "Visual",  "Global"):     5,
    ("trace_evolution",      "Active",     "Intuitive",  "Verbal",  "Sequential"): 5,
    ("trace_evolution",      "Active",     "Intuitive",  "Verbal",  "Global"):     5,
    ("trace_evolution",      "Reflective", "Sensing",    "Visual",  "Sequential"): 4,
    ("trace_evolution",      "Reflective", "Sensing",    "Visual",  "Global"):     7,
    ("trace_evolution",      "Reflective", "Sensing",    "Verbal",  "Sequential"): 1,
    ("trace_evolution",      "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("trace_evolution",      "Reflective", "Intuitive",  "Visual",  "Sequential"): 4,
    ("trace_evolution",      "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("trace_evolution",      "Reflective", "Intuitive",  "Verbal",  "Sequential"): 6,
    ("trace_evolution",      "Reflective", "Intuitive",  "Verbal",  "Global"):     10,

    # ── compare (15 questions) ─────────────────────────────────
    ("compare",              "Active",     "Sensing",    "Visual",  "Sequential"): 6,
    ("compare",              "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("compare",              "Active",     "Sensing",    "Verbal",  "Sequential"): 6,
    ("compare",              "Active",     "Sensing",    "Verbal",  "Global"):     6,
    ("compare",              "Active",     "Intuitive",  "Visual",  "Sequential"): 4,
    ("compare",              "Active",     "Intuitive",  "Visual",  "Global"):     5,
    ("compare",              "Active",     "Intuitive",  "Verbal",  "Sequential"): 6,
    ("compare",              "Active",     "Intuitive",  "Verbal",  "Global"):     5,
    ("compare",              "Reflective", "Sensing",    "Visual",  "Sequential"): 6,
    ("compare",              "Reflective", "Sensing",    "Visual",  "Global"):     7,
    ("compare",              "Reflective", "Sensing",    "Verbal",  "Sequential"): 6,
    ("compare",              "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("compare",              "Reflective", "Intuitive",  "Visual",  "Sequential"): 4,
    ("compare",              "Reflective", "Intuitive",  "Visual",  "Global"):     5,
    ("compare",              "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("compare",              "Reflective", "Intuitive",  "Verbal",  "Global"):     13,

    # ── Coverage question types (R2b — 16 questions) ───────────

    # practice
    ("practice",             "Active",     "Sensing",    "Visual",  "Sequential"): 11,
    ("practice",             "Active",     "Sensing",    "Visual",  "Global"):     11,
    ("practice",             "Active",     "Sensing",    "Verbal",  "Sequential"): 12,
    ("practice",             "Active",     "Sensing",    "Verbal",  "Global"):     12,
    ("practice",             "Active",     "Intuitive",  "Visual",  "Sequential"): 11,
    ("practice",             "Active",     "Intuitive",  "Visual",  "Global"):     11,
    ("practice",             "Active",     "Intuitive",  "Verbal",  "Sequential"): 12,
    ("practice",             "Active",     "Intuitive",  "Verbal",  "Global"):     12,
    ("practice",             "Reflective", "Sensing",    "Visual",  "Sequential"): 10,
    ("practice",             "Reflective", "Sensing",    "Visual",  "Global"):     10,
    ("practice",             "Reflective", "Sensing",    "Verbal",  "Sequential"): 10,
    ("practice",             "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("practice",             "Reflective", "Intuitive",  "Visual",  "Sequential"): 10,
    ("practice",             "Reflective", "Intuitive",  "Visual",  "Global"):     10,
    ("practice",             "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("practice",             "Reflective", "Intuitive",  "Verbal",  "Global"):     13,

    # style_adapt (PersonaRAG Adapter, FSLSM Styler)
    ("style_adapt",          "Active",     "Sensing",    "Visual",  "Sequential"): 8,
    ("style_adapt",          "Active",     "Sensing",    "Visual",  "Global"):     8,
    ("style_adapt",          "Active",     "Sensing",    "Verbal",  "Sequential"): 8,
    ("style_adapt",          "Active",     "Sensing",    "Verbal",  "Global"):     8,
    ("style_adapt",          "Active",     "Intuitive",  "Visual",  "Sequential"): 9,
    ("style_adapt",          "Active",     "Intuitive",  "Visual",  "Global"):     9,
    ("style_adapt",          "Active",     "Intuitive",  "Verbal",  "Sequential"): 9,
    ("style_adapt",          "Active",     "Intuitive",  "Verbal",  "Global"):     9,
    ("style_adapt",          "Reflective", "Sensing",    "Visual",  "Sequential"): 8,
    ("style_adapt",          "Reflective", "Sensing",    "Visual",  "Global"):     8,
    ("style_adapt",          "Reflective", "Sensing",    "Verbal",  "Sequential"): 8,
    ("style_adapt",          "Reflective", "Sensing",    "Verbal",  "Global"):     8,
    ("style_adapt",          "Reflective", "Intuitive",  "Visual",  "Sequential"): 9,
    ("style_adapt",          "Reflective", "Intuitive",  "Visual",  "Global"):     9,
    ("style_adapt",          "Reflective", "Intuitive",  "Verbal",  "Sequential"): 9,
    ("style_adapt",          "Reflective", "Intuitive",  "Verbal",  "Global"):     9,

    # search
    ("search",               "Active",     "Sensing",    "Visual",  "Sequential"): 15,
    ("search",               "Active",     "Sensing",    "Visual",  "Global"):     15,
    ("search",               "Active",     "Sensing",    "Verbal",  "Sequential"): 14,
    ("search",               "Active",     "Sensing",    "Verbal",  "Global"):     15,
    ("search",               "Active",     "Intuitive",  "Visual",  "Sequential"): 15,
    ("search",               "Active",     "Intuitive",  "Visual",  "Global"):     15,
    ("search",               "Active",     "Intuitive",  "Verbal",  "Sequential"): 15,
    ("search",               "Active",     "Intuitive",  "Verbal",  "Global"):     15,
    ("search",               "Reflective", "Sensing",    "Visual",  "Sequential"): 14,
    ("search",               "Reflective", "Sensing",    "Visual",  "Global"):     14,
    ("search",               "Reflective", "Sensing",    "Verbal",  "Sequential"): 14,
    ("search",               "Reflective", "Sensing",    "Verbal",  "Global"):     14,
    ("search",               "Reflective", "Intuitive",  "Visual",  "Sequential"): 15,
    ("search",               "Reflective", "Intuitive",  "Visual",  "Global"):     15,
    ("search",               "Reflective", "Intuitive",  "Verbal",  "Sequential"): 15,
    ("search",               "Reflective", "Intuitive",  "Verbal",  "Global"):     15,

    # summarize
    ("summarize",            "Active",     "Sensing",    "Visual",  "Sequential"): 7,
    ("summarize",            "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("summarize",            "Active",     "Sensing",    "Verbal",  "Sequential"): 1,
    ("summarize",            "Active",     "Sensing",    "Verbal",  "Global"):     13,
    ("summarize",            "Active",     "Intuitive",  "Visual",  "Sequential"): 7,
    ("summarize",            "Active",     "Intuitive",  "Visual",  "Global"):     7,
    ("summarize",            "Active",     "Intuitive",  "Verbal",  "Sequential"): 5,
    ("summarize",            "Active",     "Intuitive",  "Verbal",  "Global"):     13,
    ("summarize",            "Reflective", "Sensing",    "Visual",  "Sequential"): 7,
    ("summarize",            "Reflective", "Sensing",    "Visual",  "Global"):     7,
    ("summarize",            "Reflective", "Sensing",    "Verbal",  "Sequential"): 13,
    ("summarize",            "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("summarize",            "Reflective", "Intuitive",  "Visual",  "Sequential"): 7,
    ("summarize",            "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("summarize",            "Reflective", "Intuitive",  "Verbal",  "Sequential"): 13,
    ("summarize",            "Reflective", "Intuitive",  "Verbal",  "Global"):     13,
}


def get_optimal_tool_id(question_type: str, profile: dict) -> int:
    """Return expert-defined optimal tool for (question_type, profile).

    Falls back to Concept Explainer (1) if no mapping found.
    """
    key = (question_type, *profile_to_tuple(profile))
    return GROUND_TRUTH_MAP_FULL.get(key, 1)


@lru_cache(maxsize=1)
def load_core_answer_key() -> dict[str, dict[str, Any]]:
    if CORE_ANSWER_KEY_PATH.exists():
        return json.loads(CORE_ANSWER_KEY_PATH.read_text())
    return {}


def get_core_task_tool_id(question_id: str) -> int | None:
    answer = load_core_answer_key().get(question_id)
    if not answer:
        return None
    return int(answer["target_tool_id"])


def get_core_profile_tool_id(question_id: str, profile: dict[str, Any]) -> int | None:
    answer = load_core_answer_key().get(question_id)
    if not answer:
        return None
    profile_label = profile_to_label(profile)
    profile_targets = answer.get("profile_target_tool_ids", {}) or {}
    if profile_label in profile_targets:
        return int(profile_targets[profile_label])

    # Backward-compatible support for earlier draft answer keys.
    overrides = answer.get("profile_target_overrides", {}) or {}
    if profile_label in overrides:
        return int(overrides[profile_label])

    dims = set(profile_to_tuple(profile))
    for dim in sorted(dims):
        if dim in overrides:
            return int(overrides[dim])

    return int(answer["target_tool_id"])


def get_core_profile_eval_eligible(question_id: str) -> bool:
    answer = load_core_answer_key().get(question_id)
    if not answer:
        return False
    return bool(answer.get("profile_eval_eligible", False))


def get_core_answer_key_tool_id(question_id: str, profile: dict[str, Any] | None = None) -> int | None:
    """Backward-compatible alias for the primary task-intent tool label."""
    return get_core_task_tool_id(question_id)


def verify_coverage() -> bool:
    """Verify all 15 tools (1–15) are reachable in the map."""
    assigned = set(GROUND_TRUTH_MAP_FULL.values())
    all_tools = set(range(1, 16))
    missing = all_tools - assigned
    if missing:
        print(f"WARNING: Tools {missing} are never assigned as optimal!")
        return False
    print(f"All 15 tools are reachable in GROUND_TRUTH_MAP_FULL")
    print(f"  Total entries: {len(GROUND_TRUTH_MAP_FULL)}")
    print(f"  Unique tools: {len(assigned)}")
    return True


if __name__ == "__main__":
    verify_coverage()
