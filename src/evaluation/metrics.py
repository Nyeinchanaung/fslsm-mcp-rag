"""Metrics computation for Experiment 1 (agent fidelity) and Experiment 2 (tutor personalization)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

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


def compute_per_question_alignment(
    model: str,
    results: list[dict],
    raw_responses_dir: Path,
    questionnaire: list[dict],
) -> list[dict]:
    """
    Per-question alignment rate across all 44 ILS items for a given model.

    For each question, determines the expected answer based on the agent's
    assigned profile pole, then checks whether the actual answer matched.
    Baseline agents (no assigned profile) are automatically excluded.

    Args:
        model: short model name (e.g. "gpt-4.1-mini")
        results: records from {model}_results.json (FSLSM agents only)
        raw_responses_dir: path to results/exp1/raw_responses/
        questionnaire: list of 44 question dicts from ils_questionnaire.json

    Returns:
        264 rows (44 × 6 models), each with:
        {model, q_num, dimension, aligned_count, total_count, alignment_rate}
    """
    # agent_uid → assigned poles (from FSLSM results — no baseline entries)
    assigned_lookup = {r["agent_uid"]: r["assigned"] for r in results}

    # q_num → question dict
    q_lookup = {q["q_num"]: q for q in questionnaire}

    counts: dict[int, dict[str, int]] = {
        q["q_num"]: {"aligned": 0, "total": 0} for q in questionnaire
    }

    for raw_file in Path(raw_responses_dir).glob("*.json"):
        data = json.loads(raw_file.read_text())
        if data["model"] != model:
            continue
        agent_uid = data["agent_uid"]
        if agent_uid not in assigned_lookup:
            continue  # skip baseline files
        assigned = assigned_lookup[agent_uid]

        for item in data["raw"]:
            qn = item["q_num"]
            actual = item.get("answer")
            if actual not in ("a", "b"):
                continue
            q = q_lookup[qn]
            assigned_pole = assigned[q["dimension"]]
            expected = "a" if q["option_a"]["pole"] == assigned_pole else "b"
            counts[qn]["aligned"] += int(actual == expected)
            counts[qn]["total"] += 1

    return [
        {
            "model": model,
            "q_num": q["q_num"],
            "dimension": q["dimension"],
            "aligned_count": counts[q["q_num"]]["aligned"],
            "total_count": counts[q["q_num"]]["total"],
            "alignment_rate": (
                counts[q["q_num"]]["aligned"] / counts[q["q_num"]]["total"]
                if counts[q["q_num"]]["total"] > 0 else None
            ),
        }
        for q in questionnaire
    ]


def compute_baseline_natural_style(
    model: str,
    baseline_results: list[dict],
) -> dict:
    """
    Determine the natural learning style of an LLM (no persona) by
    aggregating raw ILS scores across all 5 baseline agents × 3 trials.

    Returns one dict per model with mean scores, detected poles, pole labels,
    and a combined style string (e.g. "Active-Sensing-Visual-Sequential").
    """
    dim_scores: dict[str, list[float]] = {d: [] for d in FSLSM_DIMENSIONS}
    for r in baseline_results:
        for d in FSLSM_DIMENSIONS:
            dim_scores[d].append(r["raw_scores"][d])

    row: dict = {"model": model}
    style_parts: list[str] = []
    for d in FSLSM_DIMENSIONS:
        mean_score = float(np.mean(dim_scores[d]))
        if mean_score < 0:
            label = FSLSM_DIM_LABELS[d][0]   # e.g. "Active"
            pole = -1
        elif mean_score > 0:
            label = FSLSM_DIM_LABELS[d][1]   # e.g. "Reflective"
            pole = 1
        else:
            label = "Neutral"
            pole = 0
        row[f"{d}_mean_score"] = round(mean_score, 3)
        row[f"{d}_detected_pole"] = pole
        row[f"{d}_label"] = label
        style_parts.append(label)

    row["detected_style"] = "-".join(style_parts)
    return row


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


# ============================================================
# EXPERIMENT 2 — Tutor Personalization Metrics
# ============================================================


def compute_scs(
    response: str,
    style_anchor: str,
    embed_model,
) -> float:
    """
    Style Conformance Score (legacy, whole-response) — cosine similarity
    between the tutor response embedding and the FSLSM style anchor embedding.

    Args:
        response: tutor response text
        style_anchor: composite FSLSM style description
        embed_model: object with .encode(text) -> np.ndarray

    Returns:
        cosine similarity in [-1, 1] (higher = better style alignment)
    """
    v_r = embed_model.encode(response)
    v_s = embed_model.encode(style_anchor)
    # L2 normalize
    v_r = v_r / (np.linalg.norm(v_r) + 1e-10)
    v_s = v_s / (np.linalg.norm(v_s) + 1e-10)
    return float(np.dot(v_r, v_s))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short fragments."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 15]


# Structural marker patterns per FSLSM dimension pole.
# Used by compute_scs_perdim to supplement embedding similarity.
_STYLE_MARKERS: dict[str, dict[int, list[str]]] = {
    "act_ref": {
        -1: [r"try\b", r"exercise", r"hands.on", r"your turn", r"practice", r"experiment with", r"implement"],
        1: [r"think about", r"consider\b", r"reflect", r"why do you", r"analyze", r"compare and contrast"],
    },
    "sen_int": {
        -1: [r"for example", r"for instance", r"specifically", r"in practice", r"\d+\.\d+", r"compute the", r"concrete"],
        1: [r"principle", r"theor", r"fundamental", r"framework", r"underlying", r"conceptual"],
    },
    "vis_ver": {
        -1: [r"diagram", r"table\b", r"figure", r"visual", r"\|.*\|", r"shown in", r"layout"],
        1: [r"in other words", r"analogy", r"think of .+ as", r"narrative", r"put it simply", r"walk.*through"],
    },
    "seq_glo": {
        -1: [r"step \d", r"first[,:]", r"next[,:]", r"then[,:]", r"finally[,:]", r"moving to"],
        1: [r"big picture", r"overview", r"overall", r"before diving", r"key takeaway", r"grand scheme", r"framework"],
    },
}


def _compute_marker_score(text: str, dim: str, pole: int) -> float:
    """Count FSLSM style markers in response text. Returns score in [0, 1]."""
    markers = _STYLE_MARKERS.get(dim, {}).get(pole, [])
    text_lower = text.lower()
    hits = sum(1 for m in markers if re.search(m, text_lower))
    return min(hits / 3.0, 1.0)


def compute_scs_perdim(
    response: str,
    dimension_anchors: dict[str, str],
    assigned_poles: dict[str, int],
    embed_model,
    top_k: int = 3,
    alpha: float = 0.5,
) -> dict:
    """
    Hybrid Per-Dimension Style Conformance Score.

    Combines two signals:
      1. Embedding similarity: top-K sentence-to-anchor cosine similarity
      2. Structural markers: regex-based detection of style-specific patterns

    SCS_dim = alpha * embed_sim + (1 - alpha) * marker_score

    Args:
        response: tutor response text
        dimension_anchors: mapping like {"act_ref=-1": "...", "act_ref=+1": "..."}
        assigned_poles: {dim_name: -1 or +1} e.g. {"act_ref": -1, "sen_int": 1, ...}
        embed_model: object with .encode(texts) supporting batch encoding
        top_k: number of top-matching sentences to average (default 3)
        alpha: weight for embedding similarity vs marker score (default 0.5)

    Returns:
        {"per_dim": {dim: float}, "overall": float}
    """
    sentences = _split_sentences(response)
    if not sentences:
        return {"per_dim": {d: 0.0 for d in assigned_poles}, "overall": 0.0}

    # Batch-encode all sentences at once
    sent_embs = embed_model.encode(sentences)
    sent_norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-10
    sent_embs = sent_embs / sent_norms

    dim_scores = {}
    for dim, pole in assigned_poles.items():
        pole_str = f"{dim}={'+1' if pole == 1 else '-1'}"
        anchor_text = dimension_anchors.get(pole_str)
        if anchor_text is None:
            dim_scores[dim] = 0.0
            continue
        anchor_emb = embed_model.encode(anchor_text)
        anchor_emb = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-10)

        # Cosine similarity of each sentence with the anchor
        sims = sent_embs @ anchor_emb  # shape (n_sentences,)
        # Take mean of top-K
        k = min(top_k, len(sims))
        embed_score = float(np.mean(np.sort(sims)[-k:]))

        # Structural marker score
        marker_score = _compute_marker_score(response, dim, pole)

        dim_scores[dim] = alpha * embed_score + (1 - alpha) * marker_score

    overall = float(np.mean(list(dim_scores.values()))) if dim_scores else 0.0
    return {"per_dim": dim_scores, "overall": overall}


def compute_rr(
    response: str,
    gold_answer: str,
    judge_client,
    student_query: str = "",
    source_chunks: list[dict] | None = None,
) -> int:
    """
    Response Relevance — LLM-as-a-Judge scoring (1–5).

    Args:
        response:       Tutor response text.
        gold_answer:    Ground truth answer string.
        judge_client:   LLMClient instance (GPT-4o, temperature=0.0).
        student_query:  The original student question (for judge context).
        source_chunks:  List of retrieved chunk dicts with 'text' keys.
                        Formatted inline for the judge to assess hallucination.

    Returns:
        Integer score 1–5. Defaults to 3 on parse failure.
    """
    from src.tutor.prompts.judge_prompts import RR_JUDGE_PROMPT, RR_JUDGE_SYSTEM

    # Format source chunks as a numbered list for the judge
    if source_chunks:
        chunks_text = "\n\n".join(
            f"[Chunk {i+1}]: {c.get('text', '')[:600]}"
            for i, c in enumerate(source_chunks[:5])
        )
    else:
        chunks_text = "(No source chunks provided)"

    prompt = RR_JUDGE_PROMPT.format(
        student_query=student_query or "(not provided)",
        gold_answer=gold_answer,
        source_chunks=chunks_text,
        response=response,
    )
    result = judge_client.chat(
        system=RR_JUDGE_SYSTEM,
        user=prompt,
    )
    text = result.content.strip()

    # Primary: match "Response Relevance Rating: [[N]]" or "Response Relevance Rating: N"
    match = re.search(r"Response Relevance Rating:\s*\[?\[?([1-5])\]?\]?", text)
    if match:
        return int(match.group(1))

    # Secondary: any "Rating: N" or "rating: N" near the end
    match = re.search(r"[Rr]ating[:\s]+\[?\[?([1-5])\]?\]?", text)
    if match:
        return int(match.group(1))

    # Last resort: last standalone digit 1-5 in the text (avoids numbered-list false positives)
    all_matches = list(re.finditer(r"\b([1-5])\b", text))
    if all_matches:
        return int(all_matches[-1].group(1))

    logger.warning("compute_rr: could not parse score from: %r — defaulting to 3", text[:200])
    return 3


def compute_cr5(
    retrieved_ids: list[str],
    gold_ids: list[str],
) -> float:
    """
    Chunk Recall@5 — fraction of gold chunks in top-5 retrieved.

    CR@5 = |retrieved[:5] ∩ gold| / |gold|
    """
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:5]) & set(gold_ids)) / len(set(gold_ids))


def compute_cr10(
    retrieved_ids: list[str],
    gold_ids: list[str],
) -> float:
    """
    Chunk Recall@10 — fraction of gold chunks in top-10 retrieved.

    CR@10 = |retrieved[:10] ∩ gold| / |gold|
    """
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:10]) & set(gold_ids)) / len(set(gold_ids))


def compute_essential_recall(
    retrieved_ids: list[str],
    essential_ids: list[str],
) -> float:
    """
    Essential Recall — fraction of essential (must-have) chunks retrieved.

    ER = |retrieved ∩ essential| / |essential|
    """
    if not essential_ids:
        return 0.0
    return len(set(retrieved_ids) & set(essential_ids)) / len(set(essential_ids))


# ---------------------------------------------------------------------------
# Pairwise pedagogical preference judge (Experiment 2 post-hoc evaluation)
# ---------------------------------------------------------------------------

# Maps the fslsm_vector dimension values to human-readable descriptions
_DIM_LABELS = {
    "act_ref": {-1: "Active", 0: "Balanced (Active/Reflective)", 1: "Reflective"},
    "sen_int": {-1: "Sensing", 0: "Balanced (Sensing/Intuitive)", 1: "Intuitive"},
    "vis_ver": {-1: "Visual", 0: "Balanced (Visual/Verbal)", 1: "Verbal"},
    "seq_glo": {-1: "Sequential", 0: "Balanced (Sequential/Global)", 1: "Global"},
}


def judge_pairwise(
    session: dict,
    swap: bool,
    judge_client,
    max_tokens: int = 200,
    response_token_cap: int = 1200,
) -> dict:
    """
    Call GPT-4o to judge one R0 vs R1 pair for pedagogical fit.

    Args:
        session:            Merged pair dict with keys: session_id, agent_id,
                            profile_label, fslsm_vector, question_text,
                            question_type, r0_response, r1_response.
        swap:               If True, assign A=R1/B=R0 (position debiasing).
        judge_client:       LLMClient(model="gpt-4o", temperature=0.0).
        max_tokens:         Max output tokens for the judge (default 200).
        response_token_cap: Approximate per-response character cap before
                            symmetrical truncation (~4 chars per token).

    Returns:
        Dict with: session_id, agent_id, profile_label, question_id,
        question_type, swap, verdict_raw, verdict_normalized, rationale,
        prompt_tokens, completion_tokens, cost, truncated.
        verdict_normalized is one of: R0_WIN, R1_WIN, TIE, PARSE_ERROR, API_ERROR.
    """
    from src.tutor.prompts.judge_prompts import (
        PAIRWISE_JUDGE_PROMPT,
        PAIRWISE_JUDGE_SYSTEM,
    )

    char_cap = response_token_cap * 4  # ~4 chars per token

    r0 = session["r0_response"]
    r1 = session["r1_response"]
    truncated = False

    if len(r0) > char_cap or len(r1) > char_cap:
        r0 = r0[:char_cap]
        r1 = r1[:char_cap]
        truncated = True

    response_a = r1 if swap else r0
    response_b = r0 if swap else r1

    # Build dimension description strings from fslsm_vector
    vec = session.get("fslsm_vector", {})
    processing_dim = _DIM_LABELS["act_ref"].get(vec.get("act_ref", 0), "Unknown")
    perception_dim = _DIM_LABELS["sen_int"].get(vec.get("sen_int", 0), "Unknown")
    input_dim = _DIM_LABELS["vis_ver"].get(vec.get("vis_ver", 0), "Unknown")
    understanding_dim = _DIM_LABELS["seq_glo"].get(vec.get("seq_glo", 0), "Unknown")

    user_prompt = PAIRWISE_JUDGE_PROMPT.format(
        profile_label=session.get("profile_label", "Unknown"),
        processing_dim=processing_dim,
        perception_dim=perception_dim,
        input_dim=input_dim,
        understanding_dim=understanding_dim,
        question_text=session.get("question_text", ""),
        response_a=response_a,
        response_b=response_b,
    )

    result = judge_client.chat(
        system=PAIRWISE_JUDGE_SYSTEM,
        user=user_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    text = result.content.strip()

    # Parse verdict: "Verdict: [[A]]" / "Verdict: [[B]]" / "Verdict: [[Tie]]"
    verdict_raw = ""
    verdict_normalized = "PARSE_ERROR"
    match = re.search(r"Verdict:\s*\[\[(A|B|Tie)\]\]", text, re.IGNORECASE)
    if match:
        tag = match.group(1).capitalize()
        verdict_raw = f"[[{tag}]]"
        if tag == "Tie":
            verdict_normalized = "TIE"
        elif tag == "A":
            verdict_normalized = "R1_WIN" if swap else "R0_WIN"
        else:  # B
            verdict_normalized = "R0_WIN" if swap else "R1_WIN"
    else:
        logger.warning(
            "judge_pairwise: could not parse verdict for %s (swap=%s): %r",
            session.get("session_id", "?"),
            swap,
            text[:200],
        )
        verdict_raw = text[:200]

    # Extract rationale (everything after "Rationale:")
    rationale = ""
    rat_match = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    if rat_match:
        rationale = rat_match.group(1).strip()

    return {
        "session_id": session.get("session_id", ""),
        "agent_id": session.get("agent_id", ""),
        "profile_label": session.get("profile_label", ""),
        "question_id": session.get("question_id", ""),
        "question_type": session.get("question_type", ""),
        "swap": swap,
        "verdict_raw": verdict_raw,
        "verdict_normalized": verdict_normalized,
        "rationale": rationale,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost": result.cost,
        "truncated": truncated,
    }
