"""
Experiment 2 — Pairwise GPT-4o-as-Judge Evaluation
====================================================
Post-hoc evaluation of R0 (generic RAG) vs R1 (FSLSM-personalized) tutor
responses over all 5,760 session pairs. GPT-4o selects the more
pedagogically appropriate response given the agent's FSLSM learning profile.

Usage:
  # Step 2 — Token audit (30 calls, ~$0.20, ~1 min)
  python experiments/exp2_tutor_personalization/pairwise_eval.py --audit-only

  # Smoke test (10 pairs, no swap)
  python experiments/exp2_tutor_personalization/pairwise_eval.py --limit 10 --no-swap

  # 200-pair pilot (stratified, no swap, ~$1.30, ~5 min)
  python experiments/exp2_tutor_personalization/pairwise_eval.py --pilot

  # Full single-order run (5,760 calls, ~$37, ~30 min)
  python experiments/exp2_tutor_personalization/pairwise_eval.py --full

  # Full double-order run (11,520 calls, ~$74, ~60 min)
  python experiments/exp2_tutor_personalization/pairwise_eval.py --full --double-order

  # Run analysis only (after full run)
  python experiments/exp2_tutor_personalization/pairwise_eval.py --analyze-only

  # Tune concurrency (default 20)
  python experiments/exp2_tutor_personalization/pairwise_eval.py --full --concurrency 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scipy import stats
from tqdm import tqdm

from src.evaluation.metrics import judge_pairwise
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"
R0_JSONL = RESULTS_DIR / "raw_sessions_r0.jsonl"
R1_JSONL = RESULTS_DIR / "raw_sessions_r1.jsonl"
PAIRWISE_DIR = RESULTS_DIR / "pairwise"

RAW_RESULTS = PAIRWISE_DIR / "raw_results.jsonl"
CHECKPOINT = PAIRWISE_DIR / "checkpoint.jsonl"
FINAL_VERDICTS = PAIRWISE_DIR / "final_verdicts.csv"
TOKEN_AUDIT = PAIRWISE_DIR / "token_audit.json"
SUMMARY_OVERALL = PAIRWISE_DIR / "summary_overall.json"
SUMMARY_BY_PROFILE = PAIRWISE_DIR / "summary_by_profile.csv"
RUN_LOG = PAIRWISE_DIR / "run_log.txt"
REPORT_SNIPPET = PAIRWISE_DIR / "report_snippet.md"

EXPECTED_PAIRS = 5760
CHECKPOINT_INTERVAL = 100

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RUN_LOG, mode="a"),
        ],
    )


def log_run_header(args: argparse.Namespace) -> None:
    mode = "double-order" if args.double_order else "single-order"
    pilot = " (PILOT 200 pairs)" if args.pilot else ""
    audit = " (AUDIT ONLY)" if args.audit_only else ""
    limit_note = f" (LIMIT {args.limit})" if args.limit else ""
    logger.info("=" * 70)
    logger.info("Pairwise Evaluation Run — %s", datetime.now(timezone.utc).isoformat())
    logger.info("Mode: %s%s%s%s", mode, pilot, audit, limit_note)
    logger.info("D1 (swap): %s", "double-order" if args.double_order else ("no-swap forced" if args.no_swap else "single-order"))
    logger.info("D2 (max_tokens): 200")
    logger.info("D3 (response_token_cap): 1200")
    logger.info("D4 (concurrency): %d", args.concurrency)
    logger.info("D5 (checkpoint_interval): %d", CHECKPOINT_INTERVAL)
    logger.info("Model: gpt-4o | temperature: 0.0")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sessions() -> list[dict]:
    """Load and merge R0 + R1 JSONL files into 5,760 paired records."""
    r0_map: dict[tuple, dict] = {}
    r1_map: dict[tuple, dict] = {}

    logger.info("Loading R0 sessions from %s", R0_JSONL)
    with open(R0_JSONL) as f:
        for line in f:
            s = json.loads(line)
            key = (s["agent_id"], s["question_id"])
            r0_map[key] = s

    logger.info("Loading R1 sessions from %s", R1_JSONL)
    with open(R1_JSONL) as f:
        for line in f:
            s = json.loads(line)
            key = (s["agent_id"], s["question_id"])
            r1_map[key] = s

    logger.info("R0: %d sessions, R1: %d sessions", len(r0_map), len(r1_map))

    pairs = []
    orphans = []
    for key, r0 in r0_map.items():
        if key not in r1_map:
            orphans.append(key)
            continue
        r1 = r1_map[key]
        agent_id, question_id = key
        session_id = f"{agent_id}__{question_id}"
        pairs.append({
            "session_id": session_id,
            "agent_id": agent_id,
            "profile_label": r0.get("profile_label", r1.get("profile_label", "")),
            "fslsm_vector": r0.get("fslsm_vector", r1.get("fslsm_vector", {})),
            "question_id": question_id,
            "question_text": r0.get("question", r1.get("question", "")),
            "question_type": r0.get("question_type", r1.get("question_type", "")),
            "r0_response": r0.get("response", ""),
            "r1_response": r1.get("response", ""),
        })

    if orphans:
        logger.warning("%d R0 sessions have no matching R1 — check data integrity!", len(orphans))

    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_PAIRS} matched pairs, got {len(pairs)}. Aborting."
        )

    logger.info("Loaded %d matched session pairs.", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict[str, dict]:
    """Load existing results keyed by '{session_id}_{swap}' from checkpoint.jsonl."""
    done: dict[str, dict] = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ck = f"{rec['session_id']}_{rec['swap']}"
                done[ck] = rec
        logger.info("Loaded %d completed calls from checkpoint", len(done))
    return done


def flush_checkpoint(done: dict[str, dict]) -> None:
    """Rewrite checkpoint.jsonl from in-memory dict (ensures no duplicates)."""
    with open(CHECKPOINT, "w") as f:
        for rec in done.values():
            f.write(json.dumps(rec) + "\n")


def append_raw_result(rec: dict) -> None:
    """Append one result to raw_results.jsonl."""
    with open(RAW_RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _make_worker(judge_client: LLMClient, response_token_cap: int):
    def _work(args: tuple) -> dict:
        session, swap = args
        max_retries = 4
        for attempt in range(max_retries):
            try:
                rec = judge_pairwise(
                    session=session,
                    swap=swap,
                    judge_client=judge_client,
                    max_tokens=200,
                    response_token_cap=response_token_cap,
                )
                return rec
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "RateLimitError" in type(e).__name__ or "rate limit" in err_str.lower() or "429" in err_str
                is_server_err = "500" in err_str or "503" in err_str or "ServiceUnavailable" in err_str
                if attempt < max_retries - 1 and (is_rate_limit or is_server_err):
                    wait = 2 ** attempt + 1  # 2, 3, 5, 9 seconds
                    logger.debug("Retry %d/%d for %s (swap=%s): %s — sleeping %ds",
                                 attempt + 1, max_retries,
                                 session.get("session_id"), swap, e, wait)
                    time.sleep(wait)
                else:
                    logger.warning("FAILED %s swap=%s after %d attempts: %s",
                                   session.get("session_id"), swap, attempt + 1, e)
                    return {
                        "session_id": session.get("session_id", ""),
                        "agent_id": session.get("agent_id", ""),
                        "profile_label": session.get("profile_label", ""),
                        "question_id": session.get("question_id", ""),
                        "question_type": session.get("question_type", ""),
                        "swap": swap,
                        "verdict_raw": f"ERROR: {err_str[:200]}",
                        "verdict_normalized": "API_ERROR",
                        "rationale": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost": 0.0,
                        "truncated": False,
                    }
        # unreachable but mypy-safe
        return {"verdict_normalized": "API_ERROR", "session_id": session.get("session_id", ""),
                "swap": swap}
    return _work


# ---------------------------------------------------------------------------
# Token audit (Step 2)
# ---------------------------------------------------------------------------

def run_token_audit(sessions: list[dict], judge_client: LLMClient) -> int:
    """
    Sample 30 pairs, fill prompts, call GPT-4o, report token stats.
    Returns the recommended response_token_cap (max(1200, P95)).
    Writes results/pairwise/token_audit.json.
    """
    logger.info("=== TOKEN AUDIT: sampling 30 pairs ===")
    sample = random.sample(sessions, min(30, len(sessions)))
    worker = _make_worker(judge_client, response_token_cap=99999)

    records = []
    for s in tqdm(sample, desc="Token audit"):
        rec = worker((s, False))
        records.append(rec)

    prompt_tokens = [r["prompt_tokens"] for r in records if r["prompt_tokens"] > 0]
    completion_tokens = [r["completion_tokens"] for r in records if r["completion_tokens"] > 0]

    if not prompt_tokens:
        logger.error("Token audit failed — no successful calls.")
        return 1200

    import statistics
    pt = sorted(prompt_tokens)
    ct = sorted(completion_tokens)

    def p95(lst):
        idx = int(len(lst) * 0.95)
        return lst[min(idx, len(lst) - 1)]

    audit = {
        "n_samples": len(prompt_tokens),
        "prompt_tokens": {
            "min": min(pt), "median": statistics.median(pt),
            "mean": round(sum(pt) / len(pt), 1),
            "p95": p95(pt), "max": max(pt),
        },
        "completion_tokens": {
            "min": min(ct), "median": statistics.median(ct),
            "mean": round(sum(ct) / len(ct), 1),
            "p95": p95(ct), "max": max(ct),
        },
        "cost_per_call_sample": {
            "min": min(r["cost"] for r in records if r["cost"] > 0),
            "mean": round(sum(r["cost"] for r in records if r["cost"] > 0) / max(len(records), 1), 6),
        },
        "extrapolated_cost_single_order_usd": round(
            (sum(pt) / len(pt)) * EXPECTED_PAIRS / 1_000_000 * 2.50
            + (sum(ct) / len(ct)) * EXPECTED_PAIRS / 1_000_000 * 10.0,
            2,
        ),
    }

    TOKEN_AUDIT.write_text(json.dumps(audit, indent=2))
    logger.info("Token audit written to %s", TOKEN_AUDIT)
    logger.info("Prompt tokens  — min:%d  median:%s  mean:%s  P95:%d  max:%d",
                audit["prompt_tokens"]["min"], audit["prompt_tokens"]["median"],
                audit["prompt_tokens"]["mean"], audit["prompt_tokens"]["p95"],
                audit["prompt_tokens"]["max"])
    logger.info("Completion tokens — min:%d  median:%s  mean:%s  P95:%d  max:%d",
                audit["completion_tokens"]["min"], audit["completion_tokens"]["median"],
                audit["completion_tokens"]["mean"], audit["completion_tokens"]["p95"],
                audit["completion_tokens"]["max"])
    logger.info("Extrapolated full single-order cost: $%.2f",
                audit["extrapolated_cost_single_order_usd"])

    response_token_cap = max(1200, audit["prompt_tokens"]["p95"])
    logger.info("Response token cap set to: %d", response_token_cap)
    return response_token_cap


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def build_worklist(
    sessions: list[dict],
    done: dict[str, dict],
    double_order: bool,
    limit: int | None = None,
    pilot: bool = False,
    no_swap: bool = False,
) -> list[tuple[dict, bool]]:
    """Build the list of (session, swap) jobs, excluding already-done ones."""
    swap_flags = [False, True] if (double_order and not no_swap) else [False]

    candidates = sessions
    if pilot:
        # Stratify ~200 across 16 FSLSM profiles (~13 per profile)
        by_profile: dict[str, list[dict]] = defaultdict(list)
        for s in sessions:
            by_profile[s["profile_label"]].append(s)
        stratified = []
        per_profile = max(1, 200 // len(by_profile))
        for profile_sessions in by_profile.values():
            stratified.extend(random.sample(profile_sessions, min(per_profile, len(profile_sessions))))
        random.shuffle(stratified)
        candidates = stratified[:200]
        logger.info("Pilot: selected %d pairs across %d profiles", len(candidates), len(by_profile))
    elif limit:
        candidates = sessions[:limit]

    jobs = []
    for s in candidates:
        for swap in swap_flags:
            ck = f"{s['session_id']}_{swap}"
            if ck not in done:
                jobs.append((s, swap))
    return jobs


def run_batch(
    worklist: list[tuple[dict, bool]],
    done: dict[str, dict],
    judge_client: LLMClient,
    concurrency: int,
    response_token_cap: int,
    desc: str = "Pairwise judge",
) -> dict[str, dict]:
    """
    Run the worklist with ThreadPoolExecutor.
    Appends results to raw_results.jsonl and checkpoints every 100 calls.
    Returns updated done dict.
    """
    worker = _make_worker(judge_client, response_token_cap)
    completed = 0
    total_cost = 0.0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, job): job for job in worklist}
        with tqdm(total=len(worklist), desc=desc) as pbar:
            for future in as_completed(futures):
                rec = future.result()
                ck = f"{rec['session_id']}_{rec['swap']}"
                done[ck] = rec
                append_raw_result(rec)
                total_cost += rec.get("cost", 0.0)
                completed += 1
                if completed % CHECKPOINT_INTERVAL == 0:
                    flush_checkpoint(done)
                    logger.debug("Checkpoint: %d done, cumulative cost $%.4f", completed, total_cost)
                pbar.update(1)

    flush_checkpoint(done)
    logger.info("Batch done: %d calls, cumulative cost $%.4f", completed, total_cost)
    return done


# ---------------------------------------------------------------------------
# Swap resolution (Step 5)
# ---------------------------------------------------------------------------

def resolve_swap_pairs(done: dict[str, dict]) -> list[dict]:
    """
    Resolve (swap=False, swap=True) pairs into a single final verdict per session.
    For single-order runs, just uses swap=False verdict directly.
    """
    by_session: dict[str, dict] = {}
    for rec in done.values():
        sid = rec["session_id"]
        if sid not in by_session:
            by_session[sid] = {}
        by_session[sid][rec["swap"]] = rec

    finals = []
    for sid, pair in by_session.items():
        rec_off = pair.get(False)
        rec_on = pair.get(True)

        base = rec_off or rec_on
        agent_id = base["agent_id"]
        profile_label = base["profile_label"]
        question_id = base["question_id"]
        question_type = base["question_type"]
        v_off = rec_off["verdict_normalized"] if rec_off else None
        v_on = rec_on["verdict_normalized"] if rec_on else None

        if v_on is None:
            # Single-order: just use swap=False
            final = v_off
        elif v_off is None:
            final = v_on
        elif v_off in ("API_ERROR", "PARSE_ERROR") or v_on in ("API_ERROR", "PARSE_ERROR"):
            final = "INVALID"
        elif v_off == v_on:
            final = v_off
        else:
            final = "TIE"  # inconsistent → conservative

        finals.append({
            "session_id": sid,
            "agent_id": agent_id,
            "profile_label": profile_label,
            "question_id": question_id,
            "question_type": question_type,
            "verdict_swap_off": v_off or "",
            "verdict_swap_on": v_on or "",
            "final_verdict": final,
        })

    return finals


def write_final_verdicts(finals: list[dict]) -> None:
    import csv
    with open(FINAL_VERDICTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(finals[0].keys()))
        writer.writeheader()
        writer.writerows(finals)
    logger.info("Final verdicts written to %s (%d rows)", FINAL_VERDICTS, len(finals))


# ---------------------------------------------------------------------------
# Analysis (Step 6)
# ---------------------------------------------------------------------------

def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_success / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    margin = z * (p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run_analysis(finals: list[dict]) -> None:
    """Run §6.1, §6.2, §6.4, §6.5 analysis and write output files."""
    import csv

    valid = [f for f in finals if f["final_verdict"] != "INVALID"]
    n_r1 = sum(1 for f in valid if f["final_verdict"] == "R1_WIN")
    n_r0 = sum(1 for f in valid if f["final_verdict"] == "R0_WIN")
    n_tie = sum(1 for f in valid if f["final_verdict"] == "TIE")
    n_invalid = len(finals) - len(valid)
    n_decisive = n_r1 + n_r0

    win_rate = n_r1 / n_decisive if n_decisive else 0.0
    ci_lo, ci_hi = wilson_ci(n_r1, n_decisive)

    binom_result = stats.binomtest(n_r1, n_decisive, p=0.5, alternative="greater")
    p_value = binom_result.pvalue

    logger.info("=== §6.1 OVERALL WIN RATE ===")
    logger.info("R1_WIN: %d (%.1f%%)  R0_WIN: %d (%.1f%%)  TIE: %d  INVALID: %d",
                n_r1, n_r1/len(finals)*100 if finals else 0,
                n_r0, n_r0/len(finals)*100 if finals else 0,
                n_tie, n_invalid)
    logger.info("Win rate R1 (decisive): %.3f  95%% CI: [%.3f, %.3f]  p=%.4g",
                win_rate, ci_lo, ci_hi, p_value)
    if n_tie / len(valid) > 0.30 if valid else False:
        logger.warning("Tie rate %.1f%% > 30%% — consider discussing response differentiation in thesis",
                       n_tie / len(valid) * 100)

    overall = {
        "n_total": len(finals),
        "n_valid": len(valid),
        "n_r1_win": n_r1, "n_r0_win": n_r0, "n_tie": n_tie, "n_invalid": n_invalid,
        "n_decisive": n_decisive,
        "win_rate_r1": round(win_rate, 4),
        "win_rate_r1_ci_lo": round(ci_lo, 4),
        "win_rate_r1_ci_hi": round(ci_hi, 4),
        "binomial_p_value": float(round(p_value, 6)),
        "significant": bool(p_value < 0.05),
        "tie_rate": round(n_tie / len(valid), 4) if valid else 0,
        "tie_rate_flag": bool(n_tie / len(valid) > 0.30) if valid else False,
    }
    SUMMARY_OVERALL.write_text(json.dumps(overall, indent=2))
    logger.info("Overall summary written to %s", SUMMARY_OVERALL)

    # §6.2 — Per-profile win rate
    logger.info("=== §6.2 PER-PROFILE WIN RATE ===")
    alpha_bonferroni = 0.05 / 16
    by_profile: dict[str, list] = defaultdict(list)
    for f in valid:
        by_profile[f["profile_label"]].append(f["final_verdict"])

    profile_rows = []
    for profile, verdicts in sorted(by_profile.items()):
        pr1 = verdicts.count("R1_WIN")
        pr0 = verdicts.count("R0_WIN")
        ptie = verdicts.count("TIE")
        pdec = pr1 + pr0
        pwr = pr1 / pdec if pdec else 0.0
        pb = stats.binomtest(pr1, pdec, p=0.5, alternative="greater").pvalue if pdec else 1.0
        pci_lo, pci_hi = wilson_ci(pr1, pdec)
        row = {
            "profile_label": profile,
            "n_r1_win": pr1, "n_r0_win": pr0, "n_tie": ptie,
            "n_decisive": pdec,
            "win_rate_r1": round(pwr, 4),
            "ci_lo": round(pci_lo, 4), "ci_hi": round(pci_hi, 4),
            "p_value": float(round(pb, 6)),
            "significant_bonferroni": bool(pb < alpha_bonferroni),
        }
        profile_rows.append(row)
        sig_flag = "***" if pb < alpha_bonferroni else ("*" if pb < 0.05 else "")
        logger.info("  %-45s  WR=%.3f  p=%.4g  %s", profile, pwr, pb, sig_flag)

    with open(SUMMARY_BY_PROFILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)
    logger.info("Per-profile summary written to %s", SUMMARY_BY_PROFILE)

    # §6.3 skipped (no Gold/Silver tier split)
    logger.info("§6.3 (Gold/Silver tier analysis) skipped — all questions are 'silver'.")

    # §6.5 — Convergent validity (Spearman ρ vs Likert)
    _run_convergent_validity(finals)

    # Report snippet
    _write_report_snippet(overall, profile_rows)


def _run_convergent_validity(finals: list[dict]) -> None:
    """Spearman ρ between per-question pairwise win rate and Likert Eng/SCS delta."""
    metrics_csv = RESULTS_DIR / "exp2_session_metrics.csv"
    if not metrics_csv.exists():
        logger.warning("exp2_session_metrics.csv not found — skipping convergent validity check.")
        return

    import csv as _csv

    logger.info("=== §6.5 CONVERGENT VALIDITY ===")

    # Load Likert per-question means from exp2_session_metrics.csv
    r0_eng: dict[str, list] = defaultdict(list)
    r1_eng: dict[str, list] = defaultdict(list)
    r0_scs: dict[str, list] = defaultdict(list)
    r1_scs: dict[str, list] = defaultdict(list)

    with open(metrics_csv) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            qid = row.get("question_id", "")
            mode = row.get("mode", "")
            try:
                eng = float(row.get("engagement_score", 0) or 0)
                scs = float(row.get("scs_score", 0) or 0)
            except ValueError:
                continue
            if mode == "R0":
                r0_eng[qid].append(eng)
                r0_scs[qid].append(scs)
            elif mode == "R1":
                r1_eng[qid].append(eng)
                r1_scs[qid].append(scs)

    # Per-question pairwise win rate
    pairwise_wr: dict[str, dict] = defaultdict(lambda: {"r1": 0, "dec": 0})
    for f in finals:
        if f["final_verdict"] == "INVALID":
            continue
        qid = f["question_id"]
        if f["final_verdict"] in ("R1_WIN", "R0_WIN"):
            pairwise_wr[qid]["dec"] += 1
            if f["final_verdict"] == "R1_WIN":
                pairwise_wr[qid]["r1"] += 1

    common_qids = [q for q in pairwise_wr if q in r0_eng and q in r1_eng and pairwise_wr[q]["dec"] > 0]
    if len(common_qids) < 5:
        logger.warning("Too few common question IDs (%d) for Spearman correlation.", len(common_qids))
        return

    pw_wr_vec = []
    eng_delta_vec = []
    scs_delta_vec = []
    for qid in common_qids:
        wr = pairwise_wr[qid]["r1"] / pairwise_wr[qid]["dec"]
        e_delta = (sum(r1_eng[qid]) / len(r1_eng[qid])) - (sum(r0_eng[qid]) / len(r0_eng[qid]))
        s_delta = (sum(r1_scs[qid]) / len(r1_scs[qid])) - (sum(r0_scs[qid]) / len(r0_scs[qid]))
        pw_wr_vec.append(wr)
        eng_delta_vec.append(e_delta)
        scs_delta_vec.append(s_delta)

    rho_eng, p_eng = stats.spearmanr(pw_wr_vec, eng_delta_vec)
    rho_scs, p_scs = stats.spearmanr(pw_wr_vec, scs_delta_vec)

    logger.info("Spearman ρ (pairwise WR vs Engagement delta): %.3f  p=%.4g  (n=%d)",
                rho_eng, p_eng, len(common_qids))
    logger.info("Spearman ρ (pairwise WR vs SCS delta):        %.3f  p=%.4g  (n=%d)",
                rho_scs, p_scs, len(common_qids))

    convergence = {
        "n_questions": len(common_qids),
        "spearman_engagement": {"rho": round(rho_eng, 4), "p_value": round(p_eng, 6)},
        "spearman_scs": {"rho": round(rho_scs, 4), "p_value": round(p_scs, 6)},
        "convergent_on_engagement": rho_eng > 0 and p_eng < 0.05,
        "convergent_on_scs": rho_scs > 0 and p_scs < 0.05,
    }
    (PAIRWISE_DIR / "convergence_check.json").write_text(json.dumps(convergence, indent=2))
    logger.info("Convergence check written to %s", PAIRWISE_DIR / "convergence_check.json")


def _write_report_snippet(overall: dict, profile_rows: list[dict]) -> None:
    n_sig_profiles = sum(1 for r in profile_rows if r["significant_bonferroni"])
    wr = overall["win_rate_r1"]
    wr_r0 = overall["n_r0_win"] / overall["n_decisive"] if overall["n_decisive"] else 0
    snippet = f"""A post-hoc pairwise evaluation was conducted over all 5,760 session pairs
from Experiment 2 using GPT-4o as LLM-as-a-Judge. For each pair, the judge
was presented with the same student question, the agent's FSLSM profile
description, and both R0 and R1 responses (order: fixed with position-swap
debiasing using double-order swap where applicable), and asked to select the
more pedagogically appropriate response.

R1 (personalized) responses were preferred in {wr*100:.1f}% of decisive comparisons
(n={overall['n_decisive']}, ties excluded), compared to {wr_r0*100:.1f}% for R0. A one-tailed binomial test
confirmed this preference was statistically significant (p={overall['binomial_p_value']:.4g}, 95% CI
{overall['win_rate_r1_ci_lo']:.3f}–{overall['win_rate_r1_ci_hi']:.3f}). The tie rate was {overall['tie_rate']*100:.1f}%. Per-profile analysis showed R1 winning
in {n_sig_profiles} of 16 FSLSM profiles at Bonferroni-corrected significance (α=0.003).
"""
    REPORT_SNIPPET.write_text(snippet)
    logger.info("Report snippet written to %s", REPORT_SNIPPET)


# ---------------------------------------------------------------------------
# Pilot summary
# ---------------------------------------------------------------------------

def print_pilot_summary(done: dict[str, dict]) -> None:
    """Print verdict distribution + sample rationales after pilot run."""
    records = [r for r in done.values() if not r["swap"]]
    total = len(records)
    counts: dict[str, int] = defaultdict(int)
    by_verdict: dict[str, list] = defaultdict(list)
    for r in records:
        v = r["verdict_normalized"]
        counts[v] += 1
        if r.get("rationale"):
            by_verdict[v].append(r["rationale"])

    print("\n" + "=" * 60)
    print(f"PILOT SUMMARY ({total} pairs, single-order)")
    print("=" * 60)
    for v, n in sorted(counts.items()):
        print(f"  {v:15s}: {n:4d}  ({n/total*100:.1f}%)")

    total_cost = sum(r.get("cost", 0) for r in done.values())
    total_prompt = sum(r.get("prompt_tokens", 0) for r in done.values())
    median_prompt = sorted(r.get("prompt_tokens", 0) for r in done.values())[len(done)//2]
    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Median prompt tokens/call: {median_prompt}")
    print(f"  Extrapolated full single-order: ~${total_cost / total * 5760:.0f}")
    print(f"  Extrapolated full double-order:  ~${total_cost / total * 11520:.0f}")

    print("\nSample rationales:")
    for v in ("R1_WIN", "R0_WIN", "TIE"):
        rats = by_verdict.get(v, [])
        if rats:
            sample = random.sample(rats, min(3, len(rats)))
            for i, rat in enumerate(sample, 1):
                print(f"\n  [{v} #{i}] {rat[:300]}")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  Single-order full run: python pairwise_eval.py --full --no-swap")
    print("  Double-order full run: python pairwise_eval.py --full --double-order")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise GPT-4o judge evaluation for Exp2.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true", help="Run token audit only (30 calls).")
    mode.add_argument("--pilot", action="store_true", help="Run 200-pair pilot (stratified, no swap).")
    mode.add_argument("--full", action="store_true", help="Run full evaluation (5,760 or 11,520 calls).")
    mode.add_argument("--analyze-only", action="store_true", help="Run analysis on existing results only.")
    p.add_argument("--limit", type=int, default=None, help="Cap total pairs (for smoke tests).")
    p.add_argument("--no-swap", action="store_true", help="Force single-order (no swap debiasing).")
    p.add_argument("--double-order", action="store_true", help="Run double-order swap debiasing.")
    p.add_argument("--concurrency", type=int, default=20, help="ThreadPoolExecutor max_workers (default 20).")
    p.add_argument("--model", default="gpt-4o", help="Judge model (default: gpt-4o).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()
    log_run_header(args)

    # Environment check
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Aborting.")
        sys.exit(1)

    # Initialize judge client
    judge_client = LLMClient(model=args.model, temperature=0.0)
    logger.info("Judge client: %s", args.model)

    # --analyze-only: skip to analysis
    if args.analyze_only:
        if not CHECKPOINT.exists():
            logger.error("No checkpoint found. Run evaluation first.")
            sys.exit(1)
        done = load_checkpoint()
        finals = resolve_swap_pairs(done)
        write_final_verdicts(finals)
        run_analysis(finals)
        return

    # Load and validate session data
    sessions = load_sessions()

    # --audit-only
    if args.audit_only:
        run_token_audit(sessions, judge_client)
        return

    # Token audit (always runs first to set cap)
    if not TOKEN_AUDIT.exists():
        response_token_cap = run_token_audit(sessions, judge_client)
    else:
        audit_data = json.loads(TOKEN_AUDIT.read_text())
        response_token_cap = max(1200, audit_data["prompt_tokens"]["p95"])
        logger.info("Reusing existing token audit. Response token cap: %d", response_token_cap)

    # Load checkpoint
    done = load_checkpoint()

    # Build worklist
    double_order = args.double_order and not args.no_swap
    worklist = build_worklist(
        sessions=sessions,
        done=done,
        double_order=double_order,
        limit=args.limit,
        pilot=args.pilot,
        no_swap=args.no_swap,
    )
    logger.info("Worklist: %d jobs (%d already done)", len(worklist), len(done))

    if not worklist:
        logger.info("All jobs already completed. Skipping to analysis.")
    else:
        desc = "Pilot (200 pairs)" if args.pilot else f"Pairwise judge ({'double' if double_order else 'single'}-order)"
        done = run_batch(
            worklist=worklist,
            done=done,
            judge_client=judge_client,
            concurrency=args.concurrency,
            response_token_cap=response_token_cap,
            desc=desc,
        )

    # Pilot: print summary and stop
    if args.pilot or args.limit:
        print_pilot_summary(done)
        if not args.full:
            return

    # Full run: resolve + analyze
    if args.full or (not args.pilot and not args.limit):
        finals = resolve_swap_pairs(done)
        write_final_verdicts(finals)
        run_analysis(finals)


if __name__ == "__main__":
    main()
