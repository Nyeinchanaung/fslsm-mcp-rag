from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.config import METRICS_JSON_PATH, RESULTS_DB_PATH
from experiments.exp3_mcp_runtime.core.run_artifacts import get_run_artifacts


CONDITION_ORDER = {"S0": 0, "S1a": 1, "S1b": 2}


def _mean(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(exp3_runtime_sessions)").fetchall()
    }
    new_cols = {
        "benchmark": "TEXT NOT NULL DEFAULT 'legacy'",
        "profile_label": "TEXT NOT NULL DEFAULT ''",
        "task_optimal_tool_id": "INTEGER NOT NULL DEFAULT 0",
        "task_tsa_hit": "INTEGER NOT NULL DEFAULT 0",
        "profile_optimal_tool_id": "INTEGER NOT NULL DEFAULT 0",
        "profile_tsa_hit": "INTEGER NOT NULL DEFAULT 0",
        "profile_eval_eligible": "INTEGER NOT NULL DEFAULT 0",
    }
    for col_name, col_type in new_cols.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE exp3_runtime_sessions ADD COLUMN {col_name} {col_type}")
    conn.commit()


def _load_rows(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    _ensure_columns(conn)
    raw_rows = conn.execute(
        """SELECT benchmark, session_id, condition,
                  CASE WHEN task_optimal_tool_id = 0 THEN optimal_tool_id ELSE task_optimal_tool_id END,
                  CASE WHEN task_optimal_tool_id = 0 THEN tsa_hit ELSE task_tsa_hit END,
                  CASE WHEN profile_optimal_tool_id = 0 THEN NULL ELSE profile_optimal_tool_id END,
                  CASE WHEN profile_optimal_tool_id = 0 THEN NULL ELSE profile_tsa_hit END,
                  profile_eval_eligible, pts_delta, execution_success, latency_ms,
                  question_type, profile_label, retrieved_evidence_json
           FROM exp3_runtime_sessions"""
    ).fetchall()
    conn.close()

    rows = []
    for (
        benchmark,
        session_id,
        condition,
        task_optimal_tool_id,
        task_tsa_hit,
        profile_optimal_tool_id,
        profile_tsa_hit,
        profile_eval_eligible,
        pts_delta,
        execution_success,
        latency_ms,
        question_type,
        profile_label,
        evidence_json,
    ) in raw_rows:
        try:
            evidence = json.loads(evidence_json)
        except json.JSONDecodeError:
            evidence = []
        rows.append(
            {
                "benchmark": benchmark,
                "session_id": session_id,
                "condition": condition,
                "task_optimal_tool_id": int(task_optimal_tool_id),
                "task_tsa_hit": int(task_tsa_hit),
                "profile_optimal_tool_id": (
                    int(profile_optimal_tool_id) if profile_optimal_tool_id is not None else None
                ),
                "profile_tsa_hit": int(profile_tsa_hit) if profile_tsa_hit is not None else None,
                "profile_eval_eligible": bool(profile_eval_eligible),
                "pts_delta": float(pts_delta),
                "execution_success": int(execution_success),
                "latency_ms": float(latency_ms),
                "question_type": question_type,
                "profile_label": profile_label,
                "has_evidence": bool(evidence),
            }
        )
    return rows


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profile_rows = [row for row in rows if row["profile_tsa_hit"] is not None]
    eligible_rows = [
        row
        for row in profile_rows
        if row["profile_eval_eligible"]
    ]
    task_tsa = _mean([row["task_tsa_hit"] for row in rows])
    return {
        "n": len(rows),
        "task_tsa": task_tsa,
        "tsa": task_tsa,
        "pts": _mean([row["pts_delta"] for row in rows]),
        "execution_success_rate": _mean([row["execution_success"] for row in rows]),
        "latency_ms": _mean([row["latency_ms"] for row in rows]),
        "grounded_tool_output_rate": _mean([1 if row["has_evidence"] else 0 for row in rows]),
        "profile_tsa_all": _mean([row["profile_tsa_hit"] for row in profile_rows]),
        "profile_tsa_eligible": _mean([row["profile_tsa_hit"] for row in eligible_rows]),
        "profile_eligible_n": len(eligible_rows),
    }


def _paired_delta(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    key: str,
    *,
    eligible_only: bool = False,
) -> tuple[float | None, int]:
    left = {row["session_id"]: row for row in left_rows}
    right = {row["session_id"]: row for row in right_rows}
    deltas = []
    for session_id in sorted(set(left) & set(right)):
        lrow = left[session_id]
        rrow = right[session_id]
        if eligible_only and not (lrow["profile_eval_eligible"] and rrow["profile_eval_eligible"]):
            continue
        if lrow[key] is None or rrow[key] is None:
            continue
        deltas.append(lrow[key] - rrow[key])
    return _mean(deltas), len(deltas)


def _paired_metrics(by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    paired = {}
    for lhs, rhs in [("S1b", "S1a"), ("S1b", "S0"), ("S1a", "S0")]:
        left_rows = by_condition.get(lhs, [])
        right_rows = by_condition.get(rhs, [])
        task_delta, task_n = _paired_delta(left_rows, right_rows, "task_tsa_hit")
        profile_all_delta, profile_all_n = _paired_delta(left_rows, right_rows, "profile_tsa_hit")
        profile_eligible_delta, profile_eligible_n = _paired_delta(
            left_rows,
            right_rows,
            "profile_tsa_hit",
            eligible_only=True,
        )
        pts_delta, pts_n = _paired_delta(left_rows, right_rows, "pts_delta")
        latency_delta, latency_n = _paired_delta(left_rows, right_rows, "latency_ms")
        if task_n:
            paired[f"{lhs}_minus_{rhs}"] = {
                "task_tsa_delta": task_delta,
                "tsa_delta": task_delta,
                "profile_tsa_all_delta": profile_all_delta,
                "profile_tsa_eligible_delta": profile_eligible_delta,
                "pts_delta": pts_delta,
                "latency_ms_delta": latency_delta,
                "n_pairs": task_n,
                "profile_all_n_pairs": profile_all_n,
                "profile_eligible_n_pairs": profile_eligible_n,
                "pts_n_pairs": pts_n,
                "latency_n_pairs": latency_n,
            }
    return paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Exp3 metrics for a run DB.")
    parser.add_argument("--run-id", default=None, help="Run folder id under results/runs/.")
    parser.add_argument("--db-path", default=None, help="Explicit DB path. Overrides --run-id.")
    parser.add_argument("--metrics-path", default=None, help="Explicit metrics JSON output path.")
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path)
        metrics_path = Path(args.metrics_path) if args.metrics_path else METRICS_JSON_PATH
    elif args.run_id:
        artifacts = get_run_artifacts(args.run_id)
        db_path = artifacts.db_path
        metrics_path = Path(args.metrics_path) if args.metrics_path else artifacts.metrics_path
    else:
        db_path = RESULTS_DB_PATH
        metrics_path = Path(args.metrics_path) if args.metrics_path else METRICS_JSON_PATH

    rows = _load_rows(db_path)
    by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_benchmark[row["benchmark"]][row["condition"]].append(row)

    benchmarks = {}
    for benchmark, by_condition in sorted(by_benchmark.items()):
        metrics = {}
        for condition in sorted(by_condition, key=lambda item: CONDITION_ORDER.get(item, 99)):
            metrics[condition] = _condition_metrics(by_condition[condition])
        benchmarks[benchmark] = {
            "conditions": metrics,
            "paired": _paired_metrics(by_condition),
        }

    out = {
        "source_db": str(db_path),
        "benchmarks": benchmarks,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
