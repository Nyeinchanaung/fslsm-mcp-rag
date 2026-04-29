"""
Experiment 2 — Phase 5: Evaluation & Results
==============================================
Loads JSONL session logs from Phase 4, computes all metrics
(SCS, RR, CR@5, CR@10, ER, Engagement), runs statistical significance tests,
and writes exp2_results_summary.json.

Usage:
  python experiments/exp2_tutor_personalization/evaluate_exp2.py
  python experiments/exp2_tutor_personalization/evaluate_exp2.py --skip-rr
  python experiments/exp2_tutor_personalization/evaluate_exp2.py --rr-workers 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.evaluation.metrics import compute_cr5, compute_cr10, compute_essential_recall, compute_rr, compute_scs_perdim
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path(__file__).resolve().parent / "results"
R0_JSONL = RESULTS_DIR / "raw_sessions_r0.jsonl"
R1_JSONL = RESULTS_DIR / "raw_sessions_r1.jsonl"
ANCHORS_PATH = PROJECT_ROOT / "data" / "exp2" / "scs_style_anchors.json"
OUTPUT_SUMMARY = RESULTS_DIR / "exp2_results_summary.json"
OUTPUT_METRICS_CSV = RESULTS_DIR / "exp2_session_metrics.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
CORPUS_PATH = PROJECT_ROOT / "d2l" / "output" / "d2l_corpus_chunks.json"


def load_corpus_index() -> dict[str, dict]:
    """Load corpus chunks indexed by chunk_id for fast lookup."""
    with open(CORPUS_PATH) as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_sessions(path: Path) -> list[dict]:
    """Load all sessions from a JSONL file."""
    sessions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    logger.info("Loaded %d sessions from %s", len(sessions), path.name)
    return sessions


def load_style_anchors() -> tuple[dict[str, str], dict[str, str]]:
    """Load composite AND dimension anchors."""
    with open(ANCHORS_PATH) as f:
        data = json.load(f)
    return data["composite_anchors"], data["dimension_anchors"]


def _profile_code_from_agent_id(agent_id: str) -> str:
    """Extract profile code from agent_id like P01_ActSenVisSeq_I02_int."""
    parts = agent_id.split("_")
    # Profile code is first two parts: P01_ActSenVisSeq
    return f"{parts[0]}_{parts[1]}"


# -------------------------------------------------------------------
# Metric computation
# -------------------------------------------------------------------

def compute_all_scs(
    sessions: list[dict],
    dim_anchors: dict[str, str],
) -> list[float]:
    """
    Compute per-dimension sentence-level SCS for all sessions.

    For each session, splits the response into sentences, computes cosine
    similarity between each sentence and the assigned pole's dimension anchor,
    takes the top-3 mean per dimension, then averages across 4 dimensions.
    This avoids topic-semantic dilution from embedding full responses.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    scores = []
    for session in tqdm(sessions, desc="Computing SCS"):
        fslsm = session.get("fslsm_vector")
        if not fslsm:
            scores.append(0.0)
            continue
        result = compute_scs_perdim(
            response=session["response"],
            dimension_anchors=dim_anchors,
            assigned_poles=fslsm,
            embed_model=model,
            top_k=3,
        )
        scores.append(result["overall"])
    return scores


def compute_all_rr(
    sessions: list[dict],
    max_workers: int = 3,
) -> list[int]:
    """Compute RR for all sessions using LLM-as-a-Judge (GPT-4o)."""
    judge_client = LLMClient("gpt-4o", temperature=0.0)

    # Check for existing RR checkpoint
    rr_checkpoint = RESULTS_DIR / "rr_scores_checkpoint.json"
    existing_rr = {}
    if rr_checkpoint.exists():
        existing_rr = json.loads(rr_checkpoint.read_text())
        logger.info("Loaded %d existing RR scores from checkpoint", len(existing_rr))

    scores = [0] * len(sessions)
    to_compute = []
    for i, session in enumerate(sessions):
        key = f"{session['agent_id']}_{session['question_id']}_{session['mode']}"
        if key in existing_rr:
            scores[i] = existing_rr[key]
        else:
            to_compute.append((i, session, key))

    if not to_compute:
        logger.info("All RR scores already computed")
        return scores

    logger.info("Computing RR for %d sessions (%d cached)", len(to_compute), len(existing_rr))

    corpus_index = load_corpus_index()
    logger.info("Loaded corpus index with %d chunks", len(corpus_index))

    def _compute_one(args):
        idx, session, key = args
        import time
        max_retries = 4
        chunk_ids = session.get("retrieved_chunk_ids", [])
        source_chunks = [corpus_index[cid] for cid in chunk_ids if cid in corpus_index]
        for attempt in range(max_retries):
            try:
                score = compute_rr(
                    session["response"],
                    session["gold_answer"],
                    judge_client,
                    student_query=session.get("question", ""),
                    source_chunks=source_chunks,
                )
                return idx, key, score
            except Exception as e:
                if attempt < max_retries - 1 and "RateLimitError" in str(type(e).__name__) or "Rate limit" in str(e):
                    wait = 2 ** attempt + 1  # 2, 3, 5, 9 seconds
                    time.sleep(wait)
                else:
                    logger.warning("RR failed for %s after %d attempts: %s", key, attempt + 1, e)
                    return idx, key, 3  # default neutral
        return idx, key, 3

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_one, item): item for item in to_compute}
        with tqdm(total=len(to_compute), desc="Computing RR (Judge)") as pbar:
            for future in as_completed(futures):
                idx, key, score = future.result()
                scores[idx] = score
                existing_rr[key] = score
                completed += 1
                # Checkpoint every 100
                if completed % 100 == 0:
                    rr_checkpoint.write_text(json.dumps(existing_rr))
                pbar.update(1)

    # Final checkpoint
    rr_checkpoint.write_text(json.dumps(existing_rr))
    return scores


def compute_all_cr5(sessions: list[dict]) -> list[float]:
    """Compute CR@5 (top-5) for all sessions."""
    return [
        compute_cr5(s["retrieved_chunk_ids"], s["gold_chunk_ids"])
        for s in sessions
    ]


def compute_all_cr10(sessions: list[dict]) -> list[float]:
    """Compute CR@10 (top-10) for all sessions."""
    return [
        compute_cr10(s["retrieved_chunk_ids"], s["gold_chunk_ids"])
        for s in sessions
    ]


def compute_all_er(sessions: list[dict]) -> list[float]:
    """Compute Essential Recall for all sessions."""
    return [
        compute_essential_recall(
            s["retrieved_chunk_ids"],
            s.get("essential_chunk_ids", s["gold_chunk_ids"]),
        )
        for s in sessions
    ]


def get_all_engagement(sessions: list[dict]) -> list[int]:
    """Extract engagement scores (already collected in Phase 4)."""
    return [s["engagement_score"] for s in sessions]


# -------------------------------------------------------------------
# Statistical tests
# -------------------------------------------------------------------

def paired_stats(r0_vals: list[float], r1_vals: list[float]) -> dict:
    """
    Paired statistical test (t-test or Wilcoxon) + Cohen's d.
    Sessions are matched by (agent_id, question_id).
    """
    from scipy.stats import shapiro, ttest_rel, wilcoxon

    r0 = np.array(r0_vals, dtype=float)
    r1 = np.array(r1_vals, dtype=float)
    diff = r1 - r0

    result = {
        "r0_mean": float(np.mean(r0)),
        "r0_std": float(np.std(r0, ddof=1)),
        "r1_mean": float(np.mean(r1)),
        "r1_std": float(np.std(r1, ddof=1)),
        "mean_diff": float(np.mean(diff)),
    }

    # Check normality on a sample (shapiro max 5000)
    sample = diff[:5000] if len(diff) > 5000 else diff
    try:
        _, p_normal = shapiro(sample)
    except Exception:
        p_normal = 0.0

    if p_normal > 0.05:
        stat, p_val = ttest_rel(r0, r1)
        result["test"] = "paired_t"
    else:
        stat, p_val = wilcoxon(diff, alternative="two-sided")
        result["test"] = "wilcoxon"

    result["statistic"] = float(stat)
    result["p_value"] = float(p_val)
    result["significant"] = bool(p_val < 0.05)

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(r0) - 1) * np.var(r0, ddof=1) + (len(r1) - 1) * np.var(r1, ddof=1))
        / (len(r0) + len(r1) - 2)
    )
    result["cohens_d"] = float((np.mean(r1) - np.mean(r0)) / pooled_std) if pooled_std > 0 else 0.0

    return result


# -------------------------------------------------------------------
# Per-dimension and per-profile analysis
# -------------------------------------------------------------------

def by_dimension_analysis(
    sessions_r0: list[dict],
    sessions_r1: list[dict],
    scs_r0: list[float],
    scs_r1: list[float],
    eng_r0: list[int],
    eng_r1: list[int],
) -> dict:
    """Aggregate SCS and Engagement by each FSLSM dimension pole."""
    from config.constants import FSLSM_DIM_LABELS, FSLSM_DIMENSIONS

    result = {}
    for dim in FSLSM_DIMENSIONS:
        neg_label, pos_label = FSLSM_DIM_LABELS[dim]
        neg_scs_r0, neg_scs_r1 = [], []
        pos_scs_r0, pos_scs_r1 = [], []
        neg_eng_r0, neg_eng_r1 = [], []
        pos_eng_r0, pos_eng_r1 = [], []

        for i, s in enumerate(sessions_r0):
            pole = s["fslsm_vector"][dim]
            if pole == -1:
                neg_scs_r0.append(scs_r0[i])
                neg_eng_r0.append(eng_r0[i])
            else:
                pos_scs_r0.append(scs_r0[i])
                pos_eng_r0.append(eng_r0[i])

        for i, s in enumerate(sessions_r1):
            pole = s["fslsm_vector"][dim]
            if pole == -1:
                neg_scs_r1.append(scs_r1[i])
                neg_eng_r1.append(eng_r1[i])
            else:
                pos_scs_r1.append(scs_r1[i])
                pos_eng_r1.append(eng_r1[i])

        result[dim] = {
            neg_label: {
                "SCS_R0": float(np.mean(neg_scs_r0)) if neg_scs_r0 else 0,
                "SCS_R1": float(np.mean(neg_scs_r1)) if neg_scs_r1 else 0,
                "Eng_R0": float(np.mean(neg_eng_r0)) if neg_eng_r0 else 0,
                "Eng_R1": float(np.mean(neg_eng_r1)) if neg_eng_r1 else 0,
                "n": len(neg_scs_r0),
            },
            pos_label: {
                "SCS_R0": float(np.mean(pos_scs_r0)) if pos_scs_r0 else 0,
                "SCS_R1": float(np.mean(pos_scs_r1)) if pos_scs_r1 else 0,
                "Eng_R0": float(np.mean(pos_eng_r0)) if pos_eng_r0 else 0,
                "Eng_R1": float(np.mean(pos_eng_r1)) if pos_eng_r1 else 0,
                "n": len(pos_scs_r0),
            },
        }
    return result


# -------------------------------------------------------------------
# Main orchestrator
# -------------------------------------------------------------------

def run_evaluation(skip_rr: bool = False, rr_workers: int = 3):
    """Run the full Exp2 evaluation pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    sessions_r0 = load_sessions(R0_JSONL)
    sessions_r1 = load_sessions(R1_JSONL)
    _composite_anchors, dim_anchors = load_style_anchors()

    print(f"Sessions: R0={len(sessions_r0)}, R1={len(sessions_r1)}")

    # Build matched pairs (agent_id, question_id) → index
    r0_map = {(s["agent_id"], s["question_id"]): i for i, s in enumerate(sessions_r0)}
    r1_map = {(s["agent_id"], s["question_id"]): i for i, s in enumerate(sessions_r1)}
    matched_keys = sorted(set(r0_map.keys()) & set(r1_map.keys()))
    print(f"Matched pairs: {len(matched_keys)}")

    # 1. SCS (per-dimension, sentence-level)
    print("\n--- Computing SCS (per-dimension sentence-level) ---")
    scs_r0 = compute_all_scs(sessions_r0, dim_anchors)
    scs_r1 = compute_all_scs(sessions_r1, dim_anchors)

    # 2. RR
    if skip_rr:
        print("\n--- Skipping RR (--skip-rr) ---")
        rr_r0 = [0] * len(sessions_r0)
        rr_r1 = [0] * len(sessions_r1)
    else:
        print("\n--- Computing RR (LLM-as-a-Judge) ---")
        rr_r0 = compute_all_rr(sessions_r0, max_workers=rr_workers)
        rr_r1 = compute_all_rr(sessions_r1, max_workers=rr_workers)

    # 3. CR@5 and CR@10
    print("\n--- Computing CR@5 and CR@10 ---")
    cr5_r0 = compute_all_cr5(sessions_r0)
    cr5_r1 = compute_all_cr5(sessions_r1)
    cr10_r0 = compute_all_cr10(sessions_r0)
    cr10_r1 = compute_all_cr10(sessions_r1)

    # 4. ER
    print("\n--- Computing ER ---")
    er_r0 = compute_all_er(sessions_r0)
    er_r1 = compute_all_er(sessions_r1)

    # 5. Engagement
    print("\n--- Extracting Engagement ---")
    eng_r0 = get_all_engagement(sessions_r0)
    eng_r1 = get_all_engagement(sessions_r1)

    # Build matched arrays for statistical tests
    matched_scs_r0 = [scs_r0[r0_map[k]] for k in matched_keys]
    matched_scs_r1 = [scs_r1[r1_map[k]] for k in matched_keys]
    matched_rr_r0 = [rr_r0[r0_map[k]] for k in matched_keys]
    matched_rr_r1 = [rr_r1[r1_map[k]] for k in matched_keys]
    matched_cr5_r0 = [cr5_r0[r0_map[k]] for k in matched_keys]
    matched_cr5_r1 = [cr5_r1[r1_map[k]] for k in matched_keys]
    matched_cr10_r0 = [cr10_r0[r0_map[k]] for k in matched_keys]
    matched_cr10_r1 = [cr10_r1[r1_map[k]] for k in matched_keys]
    matched_er_r0 = [er_r0[r0_map[k]] for k in matched_keys]
    matched_er_r1 = [er_r1[r1_map[k]] for k in matched_keys]
    matched_eng_r0 = [eng_r0[r0_map[k]] for k in matched_keys]
    matched_eng_r1 = [eng_r1[r1_map[k]] for k in matched_keys]

    # Statistical tests
    print("\n--- Statistical Tests ---")
    sig_results = {}
    sig_results["SCS"] = paired_stats(matched_scs_r0, matched_scs_r1)
    if not skip_rr:
        sig_results["RR"] = paired_stats(matched_rr_r0, matched_rr_r1)
    sig_results["CR@5"] = paired_stats(matched_cr5_r0, matched_cr5_r1)
    sig_results["CR@10"] = paired_stats(matched_cr10_r0, matched_cr10_r1)
    sig_results["ER"] = paired_stats(matched_er_r0, matched_er_r1)
    sig_results["Eng"] = paired_stats(matched_eng_r0, matched_eng_r1)

    # Per-dimension analysis
    print("\n--- Per-Dimension Analysis ---")
    dim_analysis = by_dimension_analysis(
        sessions_r0, sessions_r1, scs_r0, scs_r1, eng_r0, eng_r1
    )

    # Cost summary
    total_cost_r0 = sum(s.get("tutor_cost", 0) for s in sessions_r0)
    total_cost_r1 = sum(s.get("tutor_cost", 0) for s in sessions_r1)

    # Build summary
    summary = {
        "n_sessions_r0": len(sessions_r0),
        "n_sessions_r1": len(sessions_r1),
        "n_matched_pairs": len(matched_keys),
        "metrics": {
            "SCS": {"R0": {"mean": sig_results["SCS"]["r0_mean"], "std": sig_results["SCS"]["r0_std"]},
                     "R1": {"mean": sig_results["SCS"]["r1_mean"], "std": sig_results["SCS"]["r1_std"]}},
            "CR@5": {"R0": {"mean": sig_results["CR@5"]["r0_mean"], "std": sig_results["CR@5"]["r0_std"]},
                      "R1": {"mean": sig_results["CR@5"]["r1_mean"], "std": sig_results["CR@5"]["r1_std"]}},
            "CR@10": {"R0": {"mean": sig_results["CR@10"]["r0_mean"], "std": sig_results["CR@10"]["r0_std"]},
                       "R1": {"mean": sig_results["CR@10"]["r1_mean"], "std": sig_results["CR@10"]["r1_std"]}},
            "ER": {"R0": {"mean": sig_results["ER"]["r0_mean"], "std": sig_results["ER"]["r0_std"]},
                    "R1": {"mean": sig_results["ER"]["r1_mean"], "std": sig_results["ER"]["r1_std"]}},
            "Eng": {"R0": {"mean": sig_results["Eng"]["r0_mean"], "std": sig_results["Eng"]["r0_std"]},
                     "R1": {"mean": sig_results["Eng"]["r1_mean"], "std": sig_results["Eng"]["r1_std"]}},
        },
        "significance": sig_results,
        "by_dimension": dim_analysis,
        "cost": {
            "total_r0_usd": total_cost_r0,
            "total_r1_usd": total_cost_r1,
            "mean_per_session_usd": (total_cost_r0 + total_cost_r1) / (len(sessions_r0) + len(sessions_r1)),
        },
    }

    if not skip_rr:
        summary["metrics"]["RR"] = {
            "R0": {"mean": sig_results["RR"]["r0_mean"], "std": sig_results["RR"]["r0_std"]},
            "R1": {"mean": sig_results["RR"]["r1_mean"], "std": sig_results["RR"]["r1_std"]},
        }

    # Write summary
    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {OUTPUT_SUMMARY}")

    # Write per-session CSV for notebook analysis
    import csv
    with open(OUTPUT_METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "agent_id", "question_id", "mode", "profile_label",
            "act_ref", "sen_int", "vis_ver", "seq_glo",
            "scs", "rr", "cr5", "cr10", "er", "engagement",
            "latency_ms", "token_count", "tutor_cost",
        ])
        writer.writeheader()
        for sessions, scs_vals, rr_vals, cr5_vals, cr10_vals, er_vals, eng_vals in [
            (sessions_r0, scs_r0, rr_r0, cr5_r0, cr10_r0, er_r0, eng_r0),
            (sessions_r1, scs_r1, rr_r1, cr5_r1, cr10_r1, er_r1, eng_r1),
        ]:
            for i, s in enumerate(sessions):
                writer.writerow({
                    "agent_id": s["agent_id"],
                    "question_id": s["question_id"],
                    "mode": s["mode"],
                    "profile_label": s.get("profile_label", ""),
                    "act_ref": s["fslsm_vector"]["act_ref"],
                    "sen_int": s["fslsm_vector"]["sen_int"],
                    "vis_ver": s["fslsm_vector"]["vis_ver"],
                    "seq_glo": s["fslsm_vector"]["seq_glo"],
                    "scs": round(scs_vals[i], 4),
                    "rr": rr_vals[i],
                    "cr5": round(cr5_vals[i], 4),
                    "cr10": round(cr10_vals[i], 4),
                    "er": round(er_vals[i], 4),
                    "engagement": eng_vals[i],
                    "latency_ms": s.get("latency_ms", 0),
                    "token_count": s.get("token_count", 0),
                    "tutor_cost": s.get("tutor_cost", 0),
                })
    print(f"Per-session CSV saved to {OUTPUT_METRICS_CSV}")

    # Print summary table
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — RESULTS SUMMARY")
    print("=" * 60)
    for metric in ["SCS", "RR", "CR@5", "CR@10", "ER", "Eng"]:
        if metric not in summary["metrics"]:
            continue
        m = summary["metrics"][metric]
        sig = summary["significance"].get(metric, {})
        star = " *" if sig.get("significant") else ""
        d = sig.get("cohens_d", 0)
        print(f"  {metric:5s}  R0={m['R0']['mean']:.3f}±{m['R0']['std']:.3f}  "
              f"R1={m['R1']['mean']:.3f}±{m['R1']['std']:.3f}  "
              f"d={d:.3f}{star}")
    print("=" * 60)
    print(f"  Cost: R0=${summary['cost']['total_r0_usd']:.2f}  "
          f"R1=${summary['cost']['total_r1_usd']:.2f}  "
          f"Avg=${summary['cost']['mean_per_session_usd']:.4f}/session")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 2 — Evaluation")
    parser.add_argument("--skip-rr", action="store_true",
                        help="Skip RR computation (saves API cost)")
    parser.add_argument("--rr-workers", type=int, default=1,
                        help="Concurrent workers for RR judge calls (default: 1)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_evaluation(skip_rr=args.skip_rr, rr_workers=args.rr_workers)


if __name__ == "__main__":
    main()
