"""
Mini RR Validation — 50 matched pairs on 5 worst-RR profiles.

Validates that RR improvement fixes (A: token budget, B: additive directives,
C: factual anchor) narrow the R1 < R0 inversion before a full re-run.

Target: 5 profiles × 5 agents × 2 questions = 50 pairs × 2 modes = 100 sessions.
Success: RR gap (R1 − R0) improves for ≥ 3 of 5 profiles vs run2_backup baseline.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.tutor.profile_agent import ProfileAgent
from src.tutor.retrieval_agent import RetrievalAgent
from src.tutor.tutor_agent import TutorAgent
from src.utils.llm_client import LLMClient
from src.evaluation.metrics import compute_rr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AGENTS_PATH = PROJECT_ROOT / "data" / "agents" / "validated_agents.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "exp2" / "filtered_questions.json"
CORPUS_PATH = PROJECT_ROOT / "d2l" / "output" / "d2l_corpus_chunks.json"
MINI_DIR = Path(__file__).resolve().parent / "results" / "mini_rr_test"
R0_OUTPUT = MINI_DIR / "raw_sessions_r0.jsonl"
R1_OUTPUT = MINI_DIR / "raw_sessions_r1.jsonl"
RR_CHECKPOINT = MINI_DIR / "rr_checkpoint.json"
RUN2_METRICS = Path(__file__).resolve().parent / "results" / "run2_backup" / "exp2_session_metrics.csv"

# ---------------------------------------------------------------------------
# Target profiles (worst RR gap from run2_backup Fig 8)
# ---------------------------------------------------------------------------
TARGET_PROFILE_CODES = [
    "P01_ActSenVisSeq",  # Active-Sensing-Visual-Sequential  {-1,-1,-1,-1}
    "P02_ActSenVisGlo",  # Active-Sensing-Visual-Global       {-1,-1,-1,+1}
    "P03_ActSenVerSeq",  # Active-Sensing-Verbal-Sequential   {-1,-1,+1,-1}
    "P06_ActIntVisGlo",  # Active-Intuitive-Visual-Global     {-1,+1,-1,+1}
    "P09_RefSenVisSeq",  # Reflective-Sensing-Visual-Sequential {+1,-1,-1,-1}
]
N_QUESTIONS_PER_PROFILE = 2  # 5 agents × 2 questions = 10 pairs per profile


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_agents_for_profiles(codes: list[str]) -> dict[str, list[dict]]:
    """Return {profile_code: [agent, ...]} for each target profile code."""
    with open(AGENTS_PATH) as f:
        all_agents = json.load(f)
    result: dict[str, list[dict]] = {c: [] for c in codes}
    for agent in all_agents:
        uid = agent["agent_uid"]
        for code in codes:
            if uid.startswith(code + "_"):
                result[code].append(agent)
    return result


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def load_corpus_index() -> dict[str, dict]:
    with open(CORPUS_PATH) as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


# ---------------------------------------------------------------------------
# Session building
# ---------------------------------------------------------------------------

def build_sessions(
    agents_by_profile: dict[str, list[dict]],
    questions: list[dict],
    seed: int = 42,
) -> list[dict]:
    """
    Build 100 sessions: 5 profiles × 5 agents × 2 questions × 2 modes.
    Each profile gets the same 2 randomly sampled questions.
    """
    rng = random.Random(seed)
    sampled_qs = rng.sample(questions, N_QUESTIONS_PER_PROFILE)
    logger.info(
        "Sampled %d questions: %s",
        len(sampled_qs),
        [q["question_id"] for q in sampled_qs],
    )

    sessions = []
    for code in TARGET_PROFILE_CODES:
        profile_agents = agents_by_profile.get(code, [])
        if not profile_agents:
            logger.warning("No agents found for profile %s — skipping", code)
            continue
        for agent in profile_agents:
            for question in sampled_qs:
                for mode in ("R0", "R1"):
                    sessions.append({
                        "agent_id": agent["agent_uid"],
                        "profile_label": agent.get("profile_label", agent["agent_uid"]),
                        "profile_code": code,
                        "fslsm_vector": agent["fslsm_vector"],
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "gold_chunk_ids": question.get("gold_chunk_ids", []),
                        "essential_chunk_ids": question.get("essential_chunk_ids", []),
                        "gold_answer": question.get("gold_answer", ""),
                        "mode": mode,
                    })
    logger.info("Built %d total sessions", len(sessions))
    return sessions


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def load_completed_keys(output_path: Path) -> set[tuple]:
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add((rec["agent_id"], rec["question_id"], rec["mode"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_single_session(session: dict, tutor_agent: TutorAgent) -> dict:
    result = tutor_agent.run_session(session)
    result["profile_label"] = session.get("profile_label", "")
    result["profile_code"] = session.get("profile_code", "")
    result["fslsm_vector"] = session["fslsm_vector"]
    result["question"] = session["question"]
    result["gold_chunk_ids"] = session.get("gold_chunk_ids", [])
    result["essential_chunk_ids"] = session.get("essential_chunk_ids", [])
    result["gold_answer"] = session.get("gold_answer", "")
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def append_result(result: dict, path: Path) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Run sessions
# ---------------------------------------------------------------------------

def run_sessions(sessions: list[dict]):
    MINI_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    completed = load_completed_keys(R0_OUTPUT) | load_completed_keys(R1_OUTPUT)
    remaining = [
        s for s in sessions
        if (s["agent_id"], s["question_id"], s["mode"]) not in completed
    ]

    if not remaining:
        print("All sessions already completed.")
        return

    print(f"Sessions: {len(remaining)} remaining ({len(completed)} already done)")

    profile_agent = ProfileAgent(
        profiles_path=PROJECT_ROOT / "data" / "fslsm" / "profiles.json"
    )
    decompose_client = LLMClient("gpt-4.1-mini", temperature=0.0)
    retrieval_agent = RetrievalAgent(decompose_client=decompose_client)
    tutor_client = LLMClient("gpt-4.1-mini", temperature=0.3)
    student_client = LLMClient("gpt-4.1-mini", temperature=0.0)
    tutor_agent = TutorAgent(
        tutor_client=tutor_client,
        student_client=student_client,
        profile_agent=profile_agent,
        retrieval_agent=retrieval_agent,
    )

    success, errors = 0, 0
    with tqdm(total=len(remaining), desc="Mini sessions") as pbar:
        for session in remaining:
            try:
                result = run_single_session(session, tutor_agent)
                out = R0_OUTPUT if result["mode"] == "R0" else R1_OUTPUT
                append_result(result, out)
                success += 1
            except Exception as e:
                errors += 1
                logger.error(
                    "Session failed [%s / %s / %s]: %s\n%s",
                    session["agent_id"], session["question_id"],
                    session["mode"], e, traceback.format_exc(),
                )
            pbar.update(1)
            time.sleep(0.05)  # light throttle

    print(f"\nRun complete: {success} success, {errors} errors")


# ---------------------------------------------------------------------------
# Inline RR evaluation
# ---------------------------------------------------------------------------

def load_sessions_from_jsonl(path: Path) -> list[dict]:
    sessions = []
    if not path.exists():
        return sessions
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def compute_rr_for_sessions(sessions: list[dict], corpus_index: dict) -> dict[str, int]:
    """
    Compute RR for sessions not already in checkpoint.
    Returns {key: score} dict (key = agent_id_question_id_mode).
    """
    # Load checkpoint
    existing: dict[str, int] = {}
    if RR_CHECKPOINT.exists():
        existing = json.loads(RR_CHECKPOINT.read_text())
        logger.info("Loaded %d cached RR scores", len(existing))

    judge_client = LLMClient("gpt-4o", temperature=0.0)
    to_compute = []
    for s in sessions:
        key = f"{s['agent_id']}_{s['question_id']}_{s['mode']}"
        if key not in existing:
            to_compute.append((key, s))

    if not to_compute:
        print("All RR scores already cached.")
        return existing

    print(f"Computing RR for {len(to_compute)} sessions (single-threaded)...")
    for key, s in tqdm(to_compute, desc="RR judge"):
        chunk_ids = s.get("retrieved_chunk_ids", [])
        source_chunks = [corpus_index[cid] for cid in chunk_ids if cid in corpus_index]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                score = compute_rr(
                    s["response"],
                    s["gold_answer"],
                    judge_client,
                    student_query=s.get("question", ""),
                    source_chunks=source_chunks,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + 1
                    logger.warning("RR attempt %d failed, retrying in %ds: %s", attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    logger.warning("RR failed after %d attempts — defaulting to 3", max_retries)
                    score = 3
        existing[key] = score
        # Checkpoint after each score
        RR_CHECKPOINT.write_text(json.dumps(existing))

    return existing


def profile_code_from_result(result: dict) -> str:
    """Extract profile code from result dict (stored directly or derived from agent_id)."""
    if "profile_code" in result:
        return result["profile_code"]
    parts = result["agent_id"].split("_")
    return f"{parts[0]}_{parts[1]}"


def evaluate_and_print(rr_scores: dict[str, int]):
    """Print per-profile RR comparison table with run2_backup baseline."""
    sessions_r0 = load_sessions_from_jsonl(R0_OUTPUT)
    sessions_r1 = load_sessions_from_jsonl(R1_OUTPUT)

    def get_rr(session: dict) -> int:
        key = f"{session['agent_id']}_{session['question_id']}_{session['mode']}"
        return rr_scores.get(key, 3)

    # Group by profile
    profile_rr: dict[str, dict[str, list[int]]] = {
        code: {"R0": [], "R1": []} for code in TARGET_PROFILE_CODES
    }
    for s in sessions_r0:
        code = profile_code_from_result(s)
        if code in profile_rr:
            profile_rr[code]["R0"].append(get_rr(s))
    for s in sessions_r1:
        code = profile_code_from_result(s)
        if code in profile_rr:
            profile_rr[code]["R1"].append(get_rr(s))

    # Load run2_backup baseline if available
    baseline: dict[str, dict[str, float]] = {}
    if RUN2_METRICS.exists():
        try:
            import csv
            with open(RUN2_METRICS) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for code in TARGET_PROFILE_CODES:
                r0_rows = [r for r in rows if r["mode"] == "R0" and
                           r["agent_id"].startswith(code + "_")]
                r1_rows = [r for r in rows if r["mode"] == "R1" and
                           r["agent_id"].startswith(code + "_")]
                if r0_rows and r1_rows:
                    baseline[code] = {
                        "R0": sum(float(r["rr"]) for r in r0_rows) / len(r0_rows),
                        "R1": sum(float(r["rr"]) for r in r1_rows) / len(r1_rows),
                    }
        except Exception as e:
            logger.warning("Could not load run2_backup baseline: %s", e)

    # Print table
    print("\n" + "=" * 78)
    print("MINI RR TEST — Results vs run2_backup Baseline")
    print("=" * 78)
    header = (
        f"{'Profile':<22} {'Base R0':>7} {'Base R1':>7} {'Base Δ':>7}"
        f" | {'Mini R0':>7} {'Mini R1':>7} {'Mini Δ':>7}"
    )
    print(header)
    print("-" * 78)

    all_r0_mini, all_r1_mini = [], []
    improvements = 0
    for code in TARGET_PROFILE_CODES:
        r0_vals = profile_rr[code]["R0"]
        r1_vals = profile_rr[code]["R1"]
        if not r0_vals or not r1_vals:
            print(f"  {code:<20} — no data")
            continue
        r0_m = sum(r0_vals) / len(r0_vals)
        r1_m = sum(r1_vals) / len(r1_vals)
        delta = r1_m - r0_m
        all_r0_mini.extend(r0_vals)
        all_r1_mini.extend(r1_vals)

        base = baseline.get(code, {})
        if base:
            b_r0 = base["R0"]
            b_r1 = base["R1"]
            b_delta = b_r1 - b_r0
            improved = delta > b_delta
            if improved:
                improvements += 1
            marker = " ↑" if improved else "  "
            base_str = f"{b_r0:>7.3f} {b_r1:>7.3f} {b_delta:>+7.3f}"
        else:
            base_str = f"{'N/A':>7} {'N/A':>7} {'N/A':>7}"
            marker = "  "

        print(
            f"  {code:<20} {base_str}"
            f" | {r0_m:>7.3f} {r1_m:>7.3f} {delta:>+7.3f}{marker}"
        )

    print("-" * 78)
    if all_r0_mini and all_r1_mini:
        ov_r0 = sum(all_r0_mini) / len(all_r0_mini)
        ov_r1 = sum(all_r1_mini) / len(all_r1_mini)
        ov_delta = ov_r1 - ov_r0
        print(
            f"  {'Overall':<20} {'':>7} {'':>7} {'':>7}"
            f" | {ov_r0:>7.3f} {ov_r1:>7.3f} {ov_delta:>+7.3f}"
        )

    print("=" * 78)
    print(f"\nVerdict: {improvements}/5 profiles improved vs baseline.")
    if improvements >= 3:
        print("✓ SUCCESS — RR gap narrowed. Proceed to full re-run.")
    else:
        print("✗ INSUFFICIENT — Check that plan['fslsm_vector'] is present in _get_max_tokens.")

    # Also print engagement summary
    all_sessions = sessions_r0 + sessions_r1
    eng_r0 = [s["engagement_score"] for s in sessions_r0]
    eng_r1 = [s["engagement_score"] for s in sessions_r1]
    if eng_r0 and eng_r1:
        print(f"\nEngagement: R0={sum(eng_r0)/len(eng_r0):.2f}  R1={sum(eng_r1)/len(eng_r1):.2f}")

    # Cost summary
    cost = sum(s.get("tutor_cost", 0) for s in all_sessions)
    print(f"Total tutor cost: ${cost:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Mini RR Test ===")
    print(f"Target profiles: {TARGET_PROFILE_CODES}")
    print(f"Sessions per profile: {len([None]) * 5 * N_QUESTIONS_PER_PROFILE * 2} "
          f"(5 agents × {N_QUESTIONS_PER_PROFILE} questions × 2 modes)")

    # Phase 1 — Run sessions
    agents_by_profile = load_agents_for_profiles(TARGET_PROFILE_CODES)
    questions = load_questions()
    sessions = build_sessions(agents_by_profile, questions)

    run_sessions(sessions)

    # Phase 2 — Evaluate RR
    print("\n=== Evaluating RR ===")
    all_sessions = load_sessions_from_jsonl(R0_OUTPUT) + load_sessions_from_jsonl(R1_OUTPUT)
    if not all_sessions:
        print("No sessions found — nothing to evaluate.")
        return

    print(f"Loaded {len(all_sessions)} sessions for RR evaluation")
    corpus_index = load_corpus_index()
    rr_scores = compute_rr_for_sessions(all_sessions, corpus_index)
    evaluate_and_print(rr_scores)


if __name__ == "__main__":
    main()
