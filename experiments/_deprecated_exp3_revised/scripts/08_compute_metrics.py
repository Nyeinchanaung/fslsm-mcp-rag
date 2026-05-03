"""
Phase 13: Compute final Exp3 metrics for thesis Table 3.4.

Outputs:
  - exp3_metrics.json (machine-readable)
  - prints summary table

Computes:
  - Per-condition TSA ± SE (S0, S1a, S1b)
  - Per-condition PTS ± SE
  - Per-FSLSM-pole TSA breakdown for S1b
  - Per-question-type TSA breakdown for S1b
  - Per-tool TSA breakdown for S1b
  - Statistical tests: t-test + Cohen's h (S1b vs S0, S1b vs S1a)
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy import stats

from experiments.exp3_revised.config import RESULTS_DB_PATH, METRICS_JSON_PATH, RESULTS_DIR

FSLSM_POLES = ("Active", "Reflective", "Sensing", "Intuitive",
               "Visual", "Verbal", "Sequential", "Global")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def main() -> None:
    if not RESULTS_DB_PATH.exists():
        sys.exit(f"Results DB missing: {RESULTS_DB_PATH}")

    conn = sqlite3.connect(str(RESULTS_DB_PATH))

    def q(sql: str, params: tuple = ()) -> list:
        return conn.execute(sql, params).fetchall()

    metrics: dict = {}

    # Per-condition TSA + PTS
    for cond in ("S0", "S1a", "S1b"):
        rows = q("SELECT tsa_hit, pts_delta FROM exp3_session_results WHERE condition=?", (cond,))
        if not rows:
            print(f"  [warn] No rows for condition {cond}")
            continue
        tsa_hits = [r[0] for r in rows]
        pts_vals = [r[1] for r in rows]
        n = len(tsa_hits)
        metrics[cond] = {
            "n":        n,
            "tsa_mean": float(np.mean(tsa_hits)),
            "tsa_se":   float(np.std(tsa_hits, ddof=1) / np.sqrt(n)),
            "pts_mean": float(np.mean(pts_vals)),
            "pts_se":   float(np.std(pts_vals, ddof=1) / np.sqrt(n)),
        }

    # Per-FSLSM-pole TSA breakdown for S1b
    rows_s1b = q(
        "SELECT student_profile, tsa_hit FROM exp3_session_results WHERE condition='S1b'"
    )
    pole_hits: dict[str, list[int]] = {p: [] for p in FSLSM_POLES}
    for prof_json, hit in rows_s1b:
        profile = json.loads(prof_json)
        poles = set()
        poles.add("Active"     if profile["act_ref"] < 0 else "Reflective")
        poles.add("Sensing"    if profile["sen_int"] < 0 else "Intuitive")
        poles.add("Visual"     if profile["vis_ver"] < 0 else "Verbal")
        poles.add("Sequential" if profile["seq_glo"] < 0 else "Global")
        for pole in poles:
            pole_hits[pole].append(hit)
    metrics["dim_tsa_S1b"] = {
        pole: float(np.mean(hits)) if hits else 0.0
        for pole, hits in pole_hits.items()
    }

    # Per-question-type TSA breakdown for S1b
    qtypes = [r[0] for r in q(
        "SELECT DISTINCT question_type FROM exp3_session_results WHERE condition='S1b'"
    )]
    qtype_tsa = {}
    for qt in qtypes:
        rows = q("SELECT tsa_hit FROM exp3_session_results WHERE condition='S1b' AND question_type=?", (qt,))
        hits = [r[0] for r in rows]
        qtype_tsa[qt] = float(np.mean(hits)) if hits else 0.0
    metrics["qtype_tsa_S1b"] = qtype_tsa

    # Per-tool TSA breakdown for S1b (what TSA does each tool achieve when optimal?)
    tool_ids = [r[0] for r in q(
        "SELECT DISTINCT optimal_tool FROM exp3_session_results WHERE condition='S1b'"
    )]
    tool_tsa = {}
    for tid in sorted(tool_ids):
        rows = q(
            "SELECT tsa_hit FROM exp3_session_results WHERE condition='S1b' AND optimal_tool=?",
            (tid,),
        )
        hits = [r[0] for r in rows]
        tool_tsa[tid] = float(np.mean(hits)) if hits else 0.0
    metrics["tool_tsa_S1b"] = tool_tsa

    # Statistical tests
    def get_hits(cond: str) -> list[int]:
        return [r[0] for r in q(
            "SELECT tsa_hit FROM exp3_session_results WHERE condition=?", (cond,)
        )]

    def paired_test(hits_a: list[int], hits_b: list[int]) -> dict:
        t, p = stats.ttest_ind(hits_a, hits_b)
        h    = cohens_h(np.mean(hits_a), np.mean(hits_b))
        return {"t": float(t), "p": float(p), "cohens_h": float(h)}

    s0_hits  = get_hits("S0")
    s1a_hits = get_hits("S1a")
    s1b_hits = get_hits("S1b")
    if s0_hits and s1a_hits and s1b_hits:
        metrics["stats"] = {
            "S1b_vs_S0":  paired_test(s1b_hits, s0_hits),
            "S1b_vs_S1a": paired_test(s1b_hits, s1a_hits),
        }

    conn.close()

    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_JSON_PATH}")

    # Print summary table
    print(f"\n{'─'*65}")
    print(f"{'Condition':<12} {'n':>6} {'TSA':>8} {'±SE':>8} {'PTS':>8} {'±SE':>8}")
    print(f"{'─'*65}")
    for cond in ("S0", "S1a", "S1b"):
        if cond not in metrics:
            continue
        m = metrics[cond]
        print(f"{cond:<12} {m['n']:>6} {m['tsa_mean']*100:>7.1f}% "
              f"{m['tsa_se']*100:>6.2f}  {m['pts_mean']:>7.1f}% "
              f"{m['pts_se']:>6.2f}")
    print(f"{'─'*65}")

    if "stats" in metrics:
        for comp, s in metrics["stats"].items():
            print(f"\n{comp}: t={s['t']:.3f}, p={s['p']:.2e}, h={s['cohens_h']:.3f}")

    print(f"\n── Per-Dim TSA (S1b) ──")
    for pole, tsa in sorted(metrics.get("dim_tsa_S1b", {}).items()):
        bar = "█" * int(tsa * 20)
        print(f"  {pole:<12} {tsa*100:>5.1f}%  {bar}")

    print(f"\n── Per-Question-Type TSA (S1b) ──")
    for qt, tsa in sorted(metrics.get("qtype_tsa_S1b", {}).items()):
        print(f"  {qt:<25} {tsa*100:>5.1f}%")


if __name__ == "__main__":
    main()
