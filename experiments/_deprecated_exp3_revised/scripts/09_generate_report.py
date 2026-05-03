"""
Phase 13: Generate thesis-ready Table 3.4 (markdown) + 3 figures.

Figures:
  fig_tsa_by_condition.png  — TSA ± SE for S0, S1a, S1b
  fig_dim_tsa_s1b.png       — Per-FSLSM-pole TSA for S1b
  fig_pts_by_condition.png  — PTS ± SE for S1a, S1b

Requires exp3_metrics.json to exist (run 08_compute_metrics.py first).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp3_revised.config import (
    METRICS_JSON_PATH, TABLE_MD_PATH, FIGURES_DIR
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics() -> dict:
    if not METRICS_JSON_PATH.exists():
        sys.exit(f"Metrics file missing: {METRICS_JSON_PATH}\nRun 08_compute_metrics.py first.")
    with open(METRICS_JSON_PATH) as f:
        return json.load(f)


def plot_tsa_by_condition(metrics: dict) -> None:
    conds = [c for c in ("S0", "S1a", "S1b") if c in metrics]
    means = [metrics[c]["tsa_mean"] * 100 for c in conds]
    ses   = [metrics[c]["tsa_se"]   * 100 for c in conds]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(conds, means, yerr=ses, capsize=5,
                  color=["#d9534f", "#5bc0de", "#5cb85c"])
    ax.set_ylabel("Tool Selection Accuracy (%)")
    ax.set_title("TSA by Condition (Exp3 Revised)")
    ax.set_ylim(0, max(means) * 1.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_tsa_by_condition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_dim_tsa_s1b(metrics: dict) -> None:
    dim_tsa = metrics.get("dim_tsa_S1b", {})
    if not dim_tsa:
        print("  [skip] No dim_tsa_S1b data")
        return

    poles  = list(dim_tsa.keys())
    values = [dim_tsa[p] * 100 for p in poles]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(poles, values, color="#5cb85c")
    ax.set_xlabel("TSA (%)")
    ax.set_title("Per-Dimension TSA — S1b (Exp3 Revised)")
    ax.set_xlim(0, max(values) * 1.25 if values else 100)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_dim_tsa_s1b.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_pts_by_condition(metrics: dict) -> None:
    conds = [c for c in ("S1a", "S1b") if c in metrics]
    means = [metrics[c]["pts_mean"] for c in conds]
    ses   = [metrics[c]["pts_se"]   for c in conds]

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(conds, means, yerr=ses, capsize=5,
                  color=["#5bc0de", "#5cb85c"])
    ax.set_ylabel("Prompt Token Savings (%)")
    ax.set_title("PTS by Condition (vs S0 baseline)")
    ax.set_ylim(0, 110)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_pts_by_condition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def write_table_34(metrics: dict) -> None:
    lines = [
        "# Table 3.4 — Exp3 Revised: Tool Selection Accuracy & Prompt Token Savings\n",
        "| Condition | n | TSA (%) | ±SE | PTS (%) | ±SE |",
        "|-----------|---|---------|-----|---------|-----|",
    ]
    for cond in ("S0", "S1a", "S1b"):
        if cond not in metrics:
            continue
        m = metrics[cond]
        lines.append(
            f"| {cond} | {m['n']} "
            f"| {m['tsa_mean']*100:.1f} | {m['tsa_se']*100:.2f} "
            f"| {m['pts_mean']:.1f} | {m['pts_se']:.2f} |"
        )

    if "stats" in metrics:
        lines.append("\n## Statistical Tests\n")
        for comp, s in metrics["stats"].items():
            sig = "✓ significant" if s["p"] < 0.05 else "✗ not significant"
            lines.append(f"**{comp}**: t={s['t']:.3f}, p={s['p']:.2e}, "
                         f"Cohen's h={s['cohens_h']:.3f} — {sig}")

    if "dim_tsa_S1b" in metrics:
        lines.append("\n## Per-Dimension TSA (S1b)\n")
        lines.append("| Dimension | TSA (%) |")
        lines.append("|-----------|---------|")
        for pole, tsa in sorted(metrics["dim_tsa_S1b"].items()):
            lines.append(f"| {pole} | {tsa*100:.1f} |")

    TABLE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Saved {TABLE_MD_PATH}")


def main() -> None:
    metrics = load_metrics()
    print("Generating report...\n")
    write_table_34(metrics)
    plot_tsa_by_condition(metrics)
    plot_dim_tsa_s1b(metrics)
    plot_pts_by_condition(metrics)
    print("\n✅ Report generation complete.")


if __name__ == "__main__":
    main()
