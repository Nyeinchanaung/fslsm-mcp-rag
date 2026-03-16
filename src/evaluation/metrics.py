"""Metrics computation for Experiment 1 (agent fidelity)."""
from __future__ import annotations

import numpy as np

from config.constants import FSLSM_DIM_LABELS, FSLSM_DIMENSIONS


def profile_recovery_accuracy(results: list[dict]) -> dict:
    """
    PRA per dimension and overall (4D).

    Ties (detected == 0) count as mismatches.

    Args:
        results: list of dicts with 'assigned' and 'detected' keys,
                 each mapping dimension names to ±1 (or 0 for ties).

    Returns:
        {"per_dimension": {dim: float}, "overall_4d": float,
         "ties_per_dimension": {dim: int}}
    """
    dim_matches: dict[str, list[int]] = {d: [] for d in FSLSM_DIMENSIONS}
    ties: dict[str, int] = {d: 0 for d in FSLSM_DIMENSIONS}

    for r in results:
        for d in FSLSM_DIMENSIONS:
            detected = r["detected"][d]
            assigned = r["assigned"][d]
            if detected == 0:
                ties[d] += 1
                dim_matches[d].append(0)
            else:
                dim_matches[d].append(int(assigned == detected))

    pra_per_dim = {d: float(np.mean(dim_matches[d])) for d in FSLSM_DIMENSIONS}
    pra_4d = float(np.mean(list(pra_per_dim.values())))

    return {
        "per_dimension": pra_per_dim,
        "overall_4d": pra_4d,
        "ties_per_dimension": ties,
    }


def pra_by_knowledge_level(results: list[dict]) -> dict:
    """Slice PRA by knowledge_level group."""
    grouped: dict[str, list[dict]] = {}
    for r in results:
        level = r.get("knowledge_level") or "general"
        grouped.setdefault(level, []).append(r)
    return {
        level: profile_recovery_accuracy(group)
        for level, group in grouped.items()
    }


def cost_summary(results: list[dict]) -> dict:
    """Aggregate cost from LiteLLM per-call tracking."""
    costs = [r.get("cost_usd", 0) for r in results]
    return {
        "total_usd": sum(costs),
        "mean_per_agent_trial_usd": float(np.mean(costs)) if costs else 0.0,
        "num_records": len(costs),
    }


def baseline_pra_vs_all_profiles(
    baseline_results: list[dict],
    all_profiles: list[dict],
) -> dict:
    """
    Compute how well baseline agents match each of the 16 FSLSM profiles.

    For each baseline result, count dimension matches against each profile.
    Report: overall average PRA, per-profile PRA, best-matching profile,
    and per-dimension natural bias.

    Args:
        baseline_results: list of dicts with "detected" dimension poles
        all_profiles: list of profile dicts with "dimensions" {dim: -1/+1}
    """
    per_profile_scores: dict[str, float] = {}
    for profile in all_profiles:
        code = profile["profile_code"]
        if code == "P00_Baseline":
            continue
        assigned = profile["dimensions"]
        matches = []
        for r in baseline_results:
            detected = r["detected"]
            non_tie_dims = [d for d in FSLSM_DIMENSIONS if detected[d] != 0]
            if non_tie_dims:
                dim_match_count = sum(
                    1 for d in non_tie_dims if detected[d] == assigned[d]
                )
                matches.append(dim_match_count / len(non_tie_dims))
            else:
                matches.append(0.5)  # all ties → coin flip
        per_profile_scores[code] = float(np.mean(matches))

    all_pra_values = list(per_profile_scores.values())
    best_code = max(per_profile_scores, key=per_profile_scores.get)

    # Natural dimension bias
    dim_bias: dict[str, list] = {d: [] for d in FSLSM_DIMENSIONS}
    for r in baseline_results:
        for d in FSLSM_DIMENSIONS:
            dim_bias[d].append(r["raw_scores"][d])

    return {
        "avg_pra_vs_all": float(np.mean(all_pra_values)),
        "std_pra_vs_all": float(np.std(all_pra_values)),
        "per_profile_pra": per_profile_scores,
        "best_match_profile": best_code,
        "best_match_pra": per_profile_scores[best_code],
        "dimension_bias": {
            d: {
                "mean_score": float(np.mean(dim_bias[d])),
                "detected_pole": int(np.sign(np.mean(dim_bias[d]))) or 0,
                "pole_label": (
                    FSLSM_DIM_LABELS[d][0] if np.mean(dim_bias[d]) < 0
                    else FSLSM_DIM_LABELS[d][1] if np.mean(dim_bias[d]) > 0
                    else "Neutral"
                ),
            }
            for d in FSLSM_DIMENSIONS
        },
    }


def dimension_alignment_score(
    agent_embeddings: np.ndarray,
    trait_embeddings: np.ndarray,
) -> float:
    """
    DAS = cosine similarity between agent's aggregated response embeddings
    and style descriptor embeddings. Both inputs must be L2-normalized.
    """
    return float(np.dot(agent_embeddings, trait_embeddings))


def compute_das_for_results(results: list[dict]) -> list[dict]:
    """
    Compute per-dimension and overall DAS for each FSLSM agent-trial.

    Formula: DAS_d = (raw_score_d × assigned_d + 11) / 22
    - raw_score_d ∈ [-11, +11] (ILS dimension score from results)
    - assigned_d ∈ {-1, +1} (assigned profile pole)
    - 1.0 = perfect alignment, 0.5 = neutral, 0.0 = perfect misalignment

    Skips baseline agents (profile_code == BASELINE_PROFILE_CODE).

    Returns list of dicts with keys:
      agent_uid, trial, profile_code, knowledge_level,
      das_act_ref, das_sen_int, das_vis_ver, das_seq_glo, das_overall
    """
    from config.constants import BASELINE_PROFILE_CODE

    das_rows: list[dict] = []

    for r in results:
        profile_code = r.get("profile_code", "")
        assigned = r.get("assigned")
        if profile_code == BASELINE_PROFILE_CODE or not assigned:
            continue

        dim_das = {
            d: (r["raw_scores"][d] * assigned[d] + 11) / 22
            for d in FSLSM_DIMENSIONS
        }

        das_rows.append({
            "agent_uid": r["agent_uid"],
            "trial": r["trial"],
            "profile_code": profile_code,
            "knowledge_level": r.get("knowledge_level") or "general",
            "das_act_ref": dim_das["act_ref"],
            "das_sen_int": dim_das["sen_int"],
            "das_vis_ver": dim_das["vis_ver"],
            "das_seq_glo": dim_das["seq_glo"],
            "das_overall": float(np.mean(list(dim_das.values()))),
        })

    return das_rows


def compute_baseline_das(
    baseline_results: list[dict],
    all_profiles: list[dict],
) -> list[dict]:
    """
    DAS for baseline agents: computed vs ALL 16 FSLSM profiles, then averaged.

    By mathematical symmetry (8 profiles have assigned=-1, 8 have +1 per dim),
    the mean DAS = exactly 0.5 for any raw_score. This is the expected result —
    confirming baseline agents show chance-level alignment, not innate FSLSM bias.

    Returns same structure as compute_das_for_results().
    """
    from config.constants import BASELINE_PROFILE_CODE

    profiles_16 = [p for p in all_profiles if p["profile_code"] != BASELINE_PROFILE_CODE]

    das_rows: list[dict] = []
    for r in baseline_results:
        dim_das = {}
        for d in FSLSM_DIMENSIONS:
            raw_s = r["raw_scores"][d]
            dim_das[d] = float(np.mean(
                [(raw_s * p["dimensions"][d] + 11) / 22 for p in profiles_16]
            ))

        das_rows.append({
            "agent_uid": r["agent_uid"],
            "trial": r["trial"],
            "profile_code": BASELINE_PROFILE_CODE,
            "knowledge_level": r.get("knowledge_level") or "general",
            "das_act_ref": dim_das["act_ref"],
            "das_sen_int": dim_das["sen_int"],
            "das_vis_ver": dim_das["vis_ver"],
            "das_seq_glo": dim_das["seq_glo"],
            "das_overall": float(np.mean(list(dim_das.values()))),
        })

    return das_rows
