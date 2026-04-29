"""Generate thesis-ready figures and Table 3.4 markdown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.exp3_mcp_tool_selection.config import (  # noqa: E402
    FIGURES_DIR,
    METRICS_JSON_PATH,
    TABLE_MD_PATH,
)


CONDITIONS = ("S0", "S1a", "S1b")
COLOURS = ("#d9534f", "#f0ad4e", "#5cb85c")


def main() -> None:
    if not METRICS_JSON_PATH.exists():
        sys.exit(f"metrics JSON missing: {METRICS_JSON_PATH} — run 05_compute_metrics.py first")

    m = json.loads(METRICS_JSON_PATH.read_text())
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1 — TSA by condition
    means = [m[c]["tsa_mean"] * 100 for c in CONDITIONS]
    ses   = [m[c]["tsa_se"]   * 100 for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CONDITIONS, means, yerr=ses, capsize=5,
                  color=COLOURS, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Tool Selection Accuracy (%)")
    ax.set_title("Experiment 3: TSA by Condition")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_tsa_by_condition.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR / 'fig_tsa_by_condition.png'}")

    # Figure 2 — Per-dim TSA for S1b
    dim_data = m.get("dim_tsa_S1b", {})
    if dim_data:
        dims = sorted(dim_data, key=lambda x: -dim_data[x])
        values = [dim_data[d] * 100 for d in dims]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(dims, values, color="#5cb85c", edgecolor="black", linewidth=0.7)
        ax.set_xlabel("TSA (%)")
        ax.set_title("S1b TSA per FSLSM Dimension")
        ax.set_xlim(0, 100)
        for i, v in enumerate(values):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_dim_tsa_s1b.png", dpi=150)
        plt.close()
        print(f"Saved {FIGURES_DIR / 'fig_dim_tsa_s1b.png'}")

    # Figure 3 — PTS by condition
    pts_means = [m[c]["pts_mean"] for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CONDITIONS, pts_means, color=COLOURS, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Prompt Token Savings (%)")
    ax.set_title("Experiment 3: PTS by Condition")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, pts_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_pts_by_condition.png", dpi=150)
    plt.close()
    print(f"Saved {FIGURES_DIR / 'fig_pts_by_condition.png'}")

    # Table 3.4 markdown
    lines = [
        "## Table 3.4 — Experiment 3 Results: Efficiency and Precision Improvements",
        "",
        "| Condition | TSA Mean (%) | TSA SE (%) | PTS Mean (%) | N |",
        "|---|---|---|---|---|",
    ]
    for c in CONDITIONS:
        lines.append(
            f"| {c} | {m[c]['tsa_mean']*100:.1f} | "
            f"{m[c]['tsa_se']*100:.1f} | {m[c]['pts_mean']:.1f} | {m[c]['n']} |"
        )

    if m.get("stats"):
        lines += ["", "### Statistical Comparisons", "",
                  "| Comparison | t-statistic | p-value | Cohen's h |",
                  "|---|---|---|---|"]
        for key, val in m["stats"].items():
            sig = ("***" if val["p"] < 0.001 else
                   "**"  if val["p"] < 0.01 else
                   "*"   if val["p"] < 0.05 else "ns")
            lines.append(
                f"| {key} | {val['t']:+.3f} | {val['p']:.4f} {sig} | {val['cohens_h']:.3f} |"
            )

    if m.get("dim_tsa_S1b"):
        lines += ["", "### Per-Dimension TSA (S1b)", "",
                  "| Dimension | TSA (%) |", "|---|---|"]
        for d, v in sorted(m["dim_tsa_S1b"].items(), key=lambda x: -x[1]):
            lines.append(f"| {d} | {v*100:.1f} |")

    TABLE_MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved {TABLE_MD_PATH}")


if __name__ == "__main__":
    main()
