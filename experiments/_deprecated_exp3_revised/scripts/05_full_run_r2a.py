"""
Phase 11 Step 1: Full run R2a — 72 original questions × 80 agents = 5,760 sessions.
S1b tool selection + response generation (for optional Exp2 R2 extension).

Run with nohup:
  nohup python scripts/05_full_run_r2a.py > results/full_run_r2a.log 2>&1 &
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tqdm import tqdm

from experiments.exp3_revised.config import (
    AGENTS_PATH, QUESTIONS_PATH, RAW_R0_PATH, RAW_R1_PATH,
    RESULTS_DB_PATH, EXP2_PASSIVE_LOG, RESULTS_DIR,
    TOOL_INDEX_PATH, TOOL_META_PATH,
)
from experiments.exp3_revised.tools.tool_index import ToolIndex
from experiments.exp3_revised.core.session_runner import SessionRunner

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_completed(db_path) -> set:
    """Return set of (question_id, student_profile_json) already logged with condition='S1b'."""
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


def _iter_r2a_sessions(questions: dict) -> list[dict]:
    """Yield deduplicated (agent, question) pairs from R1 + R0 sessions."""
    seen: set[tuple] = set()
    sessions = []
    for path in [RAW_R1_PATH, RAW_R0_PATH]:
        if not path.exists():
            print(f"  [warn] {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["agent_id"], rec["question_id"])
                if key in seen or rec["question_id"] not in questions:
                    continue
                seen.add(key)
                sessions.append(rec)
    return sessions


def run_r2a():
    idx = ToolIndex()
    idx.load(TOOL_INDEX_PATH, TOOL_META_PATH)

    runner = SessionRunner(
        tool_index=idx,
        db_path=RESULTS_DB_PATH,
        exp2_log_path=EXP2_PASSIVE_LOG,
    )

    agents    = {a["agent_uid"]: a for a in json.loads(AGENTS_PATH.read_text())}
    questions = {q["question_id"]: q for q in json.loads(QUESTIONS_PATH.read_text())}
    sessions  = _iter_r2a_sessions(questions)

    completed = _load_completed(RESULTS_DB_PATH)
    print(f"[R2a] {len(sessions)} sessions across {len(questions)} questions ({len(completed)} already done)\n")

    skipped = 0
    for sess in tqdm(sessions, desc="R2a full run"):
        qid     = sess["question_id"]
        q       = questions[qid]
        agent   = agents.get(sess["agent_id"])
        if agent is None:
            continue

        profile_json = json.dumps(agent["fslsm_vector"])
        if (qid, profile_json) in completed:
            skipped += 1
            continue

        runner.run_s1b_session(
            question_id=qid,
            question_type=q["question_type"],
            query=q["question"],
            profile=agent["fslsm_vector"],
            rag_chunks="",
            generate_response=True,
        )

    print(f"\n[R2a] Complete. (skipped {skipped} already-done sessions)")
    print(f"  Exp3 results  → {RESULTS_DB_PATH}")
    print(f"  Exp2 passive  → {EXP2_PASSIVE_LOG}")


if __name__ == "__main__":
    run_r2a()
