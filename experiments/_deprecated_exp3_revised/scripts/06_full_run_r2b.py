"""
Phase 11 Step 2: Full run R2b — 16 coverage questions × 80 agents = 1,280 sessions.
generate_response=True — for within-R2b quality analysis only.

NOTE: R2b responses are logged to exp2_r2b_passive.jsonl.
      Do NOT compare R2b to R0/R1 — different question set.
      Within-R2b analysis: does tsa_hit=True → higher SCS/Engagement?

Run with nohup:
  nohup python scripts/06_full_run_r2b.py > results/full_run_r2b.log 2>&1 &
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tqdm import tqdm

from experiments.exp3_revised.config import (
    AGENTS_PATH, COVERAGE_Q_PATH,
    RESULTS_DB_PATH, EXP2_R2B_LOG, RESULTS_DIR,
    TOOL_INDEX_PATH, TOOL_META_PATH,
)
from experiments.exp3_revised.tools.tool_index import ToolIndex
from experiments.exp3_revised.core.session_runner import SessionRunner

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_completed(db_path) -> set:
    if not db_path.exists():
        return set()
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT question_id, student_profile FROM exp3_session_results WHERE condition='S1b'"
        ).fetchall()
        conn.close()
        return {(r[0], r[1]) for r in rows}
    except sqlite3.OperationalError:
        return set()


def run_r2b():
    if not COVERAGE_Q_PATH.exists():
        print(f"ERROR: {COVERAGE_Q_PATH} not found.")
        print("Run scripts/02_generate_coverage_questions.py → review → merge_questions.py first.")
        sys.exit(1)

    idx = ToolIndex()
    idx.load(TOOL_INDEX_PATH, TOOL_META_PATH)

    runner = SessionRunner(
        tool_index=idx,
        db_path=RESULTS_DB_PATH,
        exp2_log_path=EXP2_R2B_LOG,
    )

    agents   = {a["agent_uid"]: a for a in json.loads(AGENTS_PATH.read_text())}
    coverage = json.loads(COVERAGE_Q_PATH.read_text())

    completed = _load_completed(RESULTS_DB_PATH)
    print(f"[R2b] {len(coverage)} coverage questions × {len(agents)} agents = "
          f"{len(coverage) * len(agents)} sessions ({len(completed)} already done)\n")

    total = skipped = 0
    for q in tqdm(coverage, desc="Questions"):
        for agent in agents.values():
            profile_json = json.dumps(agent["fslsm_vector"])
            if (q["question_id"], profile_json) in completed:
                skipped += 1
                continue
            runner.run_s1b_session(
                question_id=q["question_id"],
                question_type=q["question_type"],
                query=q["question"],
                profile=agent["fslsm_vector"],
                rag_chunks="",
                generate_response=True,
            )
            total += 1

    print(f"\n[R2b] Complete. {total} new sessions, {skipped} skipped.")
    print(f"  Exp3 results (appended) → {RESULTS_DB_PATH}")
    print(f"  R2b passive log         → {EXP2_R2B_LOG}")
    print(f"  NOTE: exp2_r2b_passive.jsonl is for within-R2b analysis ONLY.")
    print(f"        Do NOT compare R2b responses to R0/R1 — different question set.")


if __name__ == "__main__":
    run_r2b()
