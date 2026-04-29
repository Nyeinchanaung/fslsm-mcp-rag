"""Compute final metrics for thesis Table 3.4.

Outputs:
  - exp3_metrics.json (machine-readable)
  - prints summary table
  - per-condition TSA, SE
  - per-condition PTS, SE
  - per-FSLSM-dimension TSA breakdown for S1b
  - t-test S1b vs S0, S1b vs S1a
  - Cohen's h effect size
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.exp3_mcp_tool_selection.config import (  # noqa: E402
    METRICS_JSON_PATH,
    RESULTS_DB_PATH,
)
from experiments.exp3_mcp_tool_selection.tool_registry import TOOL_REGISTRY  # noqa: E402


FSLSM_POLES = (
    "Active", "Reflective", "Sensing", "Intuitive",
    "Visual", "Verbal", "Sequential", "Global",
)


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def main() -> None:
    if not RESULTS_DB_PATH.exists():
        sys.exit(f"results DB missing: {RESULTS_DB_PATH}")

    conn = sqlite3.connect(str(RESULTS_DB_PATH))

    def query(sql: str, params: tuple = ()) -> list[tuple]:
        return conn.execute(sql, params).fetchall()

    metrics: dict = {}

    # Per-condition TSA + PTS
    for cond in ("S0", "S1a", "S1b"):
        rows = query(
            "SELECT tsa_hit, pts_delta FROM exp3_session_results WHERE condition = ?",
            (cond,),
        )
        if not rows:
            continue
        hits = np.array([r[0] for r in rows], dtype=float)
        pts  = np.array([r[1] for r in rows], dtype=float)
        metrics[cond] = {
            "n":         int(len(hits)),
            "tsa_mean":  float(hits.mean()),
            "tsa_se":    float(hits.std(ddof=1) / np.sqrt(len(hits))) if len(hits) > 1 else 0.0,
            "pts_mean":  float(pts.mean()),
            "pts_se":    float(pts.std(ddof=1) / np.sqrt(len(pts)))  if len(pts)  > 1 else 0.0,
        }

    # Per-FSLSM-dim TSA for S1b
    # We attribute each session to the dim(s) of its OPTIMAL tool.
    tool_dim_map = {t.tool_id: t.fslsm_dims for t in TOOL_REGISTRY}
    rows = query(
        "SELECT optimal_tool_id, tsa_hit FROM exp3_session_results WHERE condition='S1b'"
    )
    by_dim: dict[str, list[int]] = {p: [] for p in FSLSM_POLES}
    for opt_id, hit in rows:
        for dim in tool_dim_map.get(opt_id, ()):
            if dim in by_dim:
                by_dim[dim].append(hit)
    metrics["dim_tsa_S1b"] = {
        d: float(np.mean(v)) for d, v in by_dim.items() if v
    }

    # Statistical tests S1b vs S0, S1b vs S1a
    s0  = [r[0] for r in query("SELECT tsa_hit FROM exp3_session_results WHERE condition='S0'")]
    s1a = [r[0] for r in query("SELECT tsa_hit FROM exp3_session_results WHERE condition='S1a'")]
    s1b = [r[0] for r in query("SELECT tsa_hit FROM exp3_session_results WHERE condition='S1b'")]

    metrics["stats"] = {}
    if s0 and s1b:
        t, p = stats.ttest_ind(s1b, s0, equal_var=False)
        metrics["stats"]["S1b_vs_S0"] = {
            "t": float(t), "p": float(p),
            "cohens_h": cohens_h(np.mean(s1b), np.mean(s0)),
        }
    if s1a and s1b:
        t, p = stats.ttest_ind(s1b, s1a, equal_var=False)
        metrics["stats"]["S1b_vs_S1a"] = {
            "t": float(t), "p": float(p),
            "cohens_h": cohens_h(np.mean(s1b), np.mean(s1a)),
        }

    conn.close()

    # Console output
    print("\n── Experiment 3 Results (Table 3.4) ──\n")
    print(f"{'Condition':<10} {'TSA Mean':>10} {'TSA SE':>8} {'PTS Mean':>10} {'N':>6}")
    print("─" * 50)
    for cond in ("S0", "S1a", "S1b"):
        if cond not in metrics:
            continue
        m = metrics[cond]
        print(f"{cond:<10} {m['tsa_mean']*100:>9.1f}% {m['tsa_se']*100:>7.1f}% "
              f"{m['pts_mean']:>9.1f}% {m['n']:>6}")

    print("\n── Statistical Tests ──")
    for key, val in metrics["stats"].items():
        sig = ("***" if val["p"] < 0.001 else
               "**"  if val["p"] < 0.01 else
               "*"   if val["p"] < 0.05 else "ns")
        print(f"  {key:<15} t={val['t']:+.3f}  p={val['p']:.4f} {sig}  "
              f"Cohen's h={val['cohens_h']:.3f}")

    print("\n── Per-Dimension TSA (S1b) ──")
    for dim, tsa in sorted(metrics["dim_tsa_S1b"].items(), key=lambda x: -x[1]):
        print(f"  {dim:<12} {tsa*100:.1f}%")

    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
