"""
Phase 12: Post-hoc S0 + S1a ablation on all sessions already in the DB.

For each unique (session_id, question_id, question_type, profile, query) from
S1b results, also run:
  - S0:  real LLM call with all 15 schemas → log S0 result
  - S1a: FAISS on raw query (no profile) → log S1a result

S1b results must already be in RESULTS_DB_PATH (from Phase 11).

Estimated S0 cost: 7,040 calls × ~$0.0001 ≈ $0.70

Run with nohup:
  nohup python scripts/07_run_s0_s1a_ablation.py > results/ablation.log 2>&1 &
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tqdm import tqdm

from experiments.exp3_revised.config import (
    RESULTS_DB_PATH, TOOL_INDEX_PATH, TOOL_META_PATH,
)
from experiments.exp3_revised.tools.tool_index import ToolIndex
from experiments.exp3_revised.tools.tool_registry import s0_prompt_tokens, get_tool_by_id
from experiments.exp3_revised.core.s0_baseline import s0_select_tool
from experiments.exp3_revised.core.profile_decoder import profile_to_label
from experiments.exp3_revised.core.ground_truth import get_optimal_tool_id


def s1_prompt_tokens(tool_id: int) -> int:
    """Token cost for a single tool schema in an S1a/S1b prompt."""
    return get_tool_by_id(tool_id).token_cost


def _log_result(conn: sqlite3.Connection, condition: str, session_id: str,
                question_id: str, question_type: str, profile_json: str,
                query: str, selected: int, optimal: int,
                s0_tokens: int) -> None:
    tsa_hit   = int(selected == optimal)
    s_tokens  = s0_tokens if condition == "S0" else s1_prompt_tokens(selected)
    pts_delta = 0.0 if condition == "S0" else 100.0 * (1 - s_tokens / s0_tokens)
    conn.execute(
        """INSERT INTO exp3_session_results
           (condition, session_id, question_id, question_type,
            student_profile, query, selected_tool, optimal_tool,
            tsa_hit, input_tokens, pts_delta)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (condition, session_id, question_id, question_type,
         profile_json, query, selected, optimal,
         tsa_hit, s_tokens, pts_delta),
    )


def run_ablation():
    conn = sqlite3.connect(str(RESULTS_DB_PATH))

    # Check how many S1b sessions exist
    n_s1b = conn.execute(
        "SELECT COUNT(*) FROM exp3_session_results WHERE condition='S1b'"
    ).fetchone()[0]
    print(f"[ablation] Found {n_s1b} S1b sessions in DB.")

    if n_s1b == 0:
        print("ERROR: No S1b sessions found. Run Phases 11 first.")
        sys.exit(1)

    # Check which sessions already have S0/S1a (resume support)
    done_s0 = {r[0] for r in conn.execute(
        "SELECT session_id FROM exp3_session_results WHERE condition='S0'"
    ).fetchall()}
    done_s1a = {r[0] for r in conn.execute(
        "SELECT session_id FROM exp3_session_results WHERE condition='S1a'"
    ).fetchall()}

    # Load all S1b sessions not yet ablated
    rows = conn.execute(
        """SELECT session_id, question_id, question_type, student_profile, query
           FROM exp3_session_results WHERE condition='S1b'"""
    ).fetchall()

    s0_tokens = s0_prompt_tokens()

    idx = ToolIndex()
    idx.load(TOOL_INDEX_PATH, TOOL_META_PATH)

    pending = [(sid, qid, qtype, prof, q)
               for sid, qid, qtype, prof, q in rows
               if sid not in done_s0 or sid not in done_s1a]

    print(f"[ablation] {len(pending)} sessions need S0/S1a (done: {len(done_s0)} S0, {len(done_s1a)} S1a)")

    for sid, qid, qtype, prof_json, query in tqdm(pending, desc="S0+S1a ablation"):
        profile = json.loads(prof_json)
        optimal = get_optimal_tool_id(qtype, profile)
        plabel  = profile_to_label(profile)

        if sid not in done_s0:
            s0_id = s0_select_tool(query, plabel)
            _log_result(conn, "S0", sid, qid, qtype, prof_json, query,
                        s0_id, optimal, s0_tokens)
            done_s0.add(sid)

        if sid not in done_s1a:
            s1a_hits = idx.retrieve(query, k=1)
            s1a_id   = s1a_hits[0][0].tool_id
            _log_result(conn, "S1a", sid, qid, qtype, prof_json, query,
                        s1a_id, optimal, s0_tokens)
            done_s1a.add(sid)

        conn.commit()

    conn.close()
    print(f"\n[ablation] Complete. S0+S1a results logged to {RESULTS_DB_PATH}")


if __name__ == "__main__":
    run_ablation()
