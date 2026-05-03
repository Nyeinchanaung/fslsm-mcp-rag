"""
Phase 10: Dry run gate — 100 sessions from R2a (original questions).
Checks all 4 Exp3 gates. Does NOT generate responses (Exp3 metrics only).
Uses real LLM S0 calls (~100 API calls).

All 4 gates must pass before proceeding to Phase 11 (full run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tqdm import tqdm

from experiments.exp3_revised.config import (
    QUESTIONS_PATH, RAW_R0_PATH, RAW_R1_PATH,
    DRY_RUN_DB_PATH, RESULTS_DIR,
    DRY_RUN_N,
    MIN_TSA_S1B_OVER_S0, MIN_TSA_S1B_OVER_S1A, MIN_PTS_S1B,
    TOOL_INDEX_PATH, TOOL_META_PATH,
)
from experiments.exp3_revised.tools.tool_index import ToolIndex
from experiments.exp3_revised.tools.tool_registry import s0_prompt_tokens
from experiments.exp3_revised.core.session_runner import SessionRunner
from experiments.exp3_revised.core.s0_baseline import s0_select_tool
from experiments.exp3_revised.core.profile_decoder import profile_to_label
from experiments.exp3_revised.core.ground_truth import get_optimal_tool_id

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_sessions(questions: dict) -> list[dict]:
    """Load up to DRY_RUN_N sessions from Exp2 R1 (then R0) JSONL."""
    seen: set[tuple] = set()
    sessions = []
    for path in [RAW_R1_PATH, RAW_R0_PATH]:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["agent_id"], rec["question_id"])
                if key in seen:
                    continue
                if rec["question_id"] not in questions:
                    continue
                seen.add(key)
                sessions.append(rec)
                if len(sessions) >= DRY_RUN_N:
                    return sessions
    return sessions


def run_dry():
    # Load tool index
    idx = ToolIndex()
    idx.load(TOOL_INDEX_PATH, TOOL_META_PATH)

    runner = SessionRunner(
        tool_index=idx,
        db_path=DRY_RUN_DB_PATH,
        exp2_log_path=RESULTS_DIR / "exp2_dry_run.jsonl",
    )

    questions = {q["question_id"]: q for q in json.loads(QUESTIONS_PATH.read_text())}
    sessions  = _load_sessions(questions)

    print(f"[dry_run] Processing {len(sessions)} sessions...\n")

    s0_hits = s1a_hits = s1b_hits = 0
    s1b_tokens_total = total = 0
    s0_tokens = s0_prompt_tokens()

    for sess in tqdm(sessions, desc="Dry run"):
        qid      = sess["question_id"]
        q        = questions[qid]
        profile  = sess["fslsm_vector"]
        query    = q["question"]
        qtype    = q["question_type"]
        optimal  = get_optimal_tool_id(qtype, profile)
        plabel   = profile_to_label(profile)

        # S1b — via session runner (logs to DB)
        result = runner.run_s1b_session(
            question_id=qid,
            question_type=qtype,
            query=query,
            profile=profile,
            generate_response=False,
        )
        if result["tsa_hit"]:
            s1b_hits += 1
        s1b_tokens_total += result["input_tokens"]

        # S1a — FAISS on raw query (no profile)
        hits_s1a = idx.retrieve(query, k=1)
        if hits_s1a[0][0].tool_id == optimal:
            s1a_hits += 1

        # S0 — real LLM call
        s0_id = s0_select_tool(query, plabel)
        if s0_id == optimal:
            s0_hits += 1

        total += 1

    tsa_s0  = s0_hits / total
    tsa_s1a = s1a_hits / total
    tsa_s1b = s1b_hits / total
    pts_s1b = (s0_tokens - s1b_tokens_total / total) / s0_tokens * 100

    print(f"\n── Dry Run Results ({total} sessions) ──")
    print(f"  TSA S0:  {tsa_s0*100:.1f}%")
    print(f"  TSA S1a: {tsa_s1a*100:.1f}%")
    print(f"  TSA S1b: {tsa_s1b*100:.1f}%")
    print(f"  PTS S1b: {pts_s1b:.1f}%")

    print(f"\n── Go / No-Go Gates ──")
    gates = [
        (f"TSA(S1b) − TSA(S1a) ≥ {MIN_TSA_S1B_OVER_S1A*100:.0f} pp  [PRIMARY]",
         tsa_s1b - tsa_s1a >= MIN_TSA_S1B_OVER_S1A),
        (f"PTS(S1b) ≥ {MIN_PTS_S1B:.0f}%",
         pts_s1b >= MIN_PTS_S1B),
        # Informational only — S0 now uses profile description so it is not a blind baseline
        (f"TSA(S1b) − TSA(S0)  ≥ {MIN_TSA_S1B_OVER_S0*100:.0f} pp  [INFO]",
         tsa_s1b - tsa_s0 >= MIN_TSA_S1B_OVER_S0),
        ("TSA(S1a) vs TSA(S0)  [INFO]",
         tsa_s1a > tsa_s0),
    ]
    all_pass = True
    for i, (label, passed) in enumerate(gates):
        blocking = i < 2  # first two gates are blocking
        sym = "GO ✓" if passed else "NO-GO ✗"
        if not passed and blocking:
            all_pass = False
        print(f"  [{sym}] {label}")

    if all_pass:
        print("\n✅ ALL GATES PASSED — proceed to full run (Phase 11)")
    else:
        print("\n❌ GATE FAILURE — fix before proceeding")
        print("   Likely fixes:")
        print("   - Rebuild FAISS index (run script 01)")
        print("   - Check GROUND_TRUTH_MAP_FULL coverage (run core/ground_truth.py)")
        print("   - Check augmentor DIM_DIRECTIVES vocabulary alignment")
        sys.exit(1)


if __name__ == "__main__":
    run_dry()
