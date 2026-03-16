"""Generate all Experiment 1 visualizations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from src.evaluation.visualizer import (
    baseline_bias_radar,
    cost_per_model_bar,
    das_comparison_bar,
    das_fslsm_vs_baseline_bar,
    fslsm_vs_baseline_bar,
    heatmap_baseline_bias,
    heatmap_profiles,
    knowledge_level_comparison,
    model_comparison_bar,
)

MODELS = ["gpt-4.1-mini", "claude-sonnet-4-20250514", "llama3.1:8b"]


def main():
    metrics_dir = Path("results/exp1/metrics")

    # Load CSVs
    df_pra = pd.read_csv(metrics_dir / "pra_das_summary.csv")
    df_baseline = pd.read_csv(metrics_dir / "baseline_analysis.csv")
    df_cost = pd.read_csv(metrics_dir / "cost_summary.csv")

    # 1. Model comparison bar (FSLSM per-dimension PRA)
    p = model_comparison_bar(df_pra)
    print(f"1. Model comparison bar → {p}")

    # 2. Knowledge-level PRA comparison
    p = knowledge_level_comparison(df_pra)
    print(f"2. Knowledge-level PRA → {p}")

    # 3. Cost per model
    p = cost_per_model_bar(df_cost)
    print(f"3. Cost per model → {p}")

    # 4. FSLSM vs Baseline PRA
    p = fslsm_vs_baseline_bar(df_pra)
    print(f"4. FSLSM vs Baseline → {p}")

    # 5. Baseline bias radar
    p = baseline_bias_radar(df_baseline)
    print(f"5. Baseline bias radar → {p}")

    # 6. DAS comparison bar
    das_path = metrics_dir / "das_summary.csv"
    if das_path.exists():
        df_das = pd.read_csv(das_path)
        p = das_comparison_bar(df_das)
        print(f"6. DAS comparison bar → {p}")
    else:
        print("6. DAS comparison bar — skipped (das_summary.csv not found)")

    # 7. Heatmaps per model (from FSLSM results JSON)
    for model in MODELS:
        safe = model.replace("/", "_").replace(":", "_")
        results_file = metrics_dir / f"{safe}_results.json"
        if results_file.exists():
            results = json.loads(results_file.read_text())
            p = heatmap_profiles(results, model)
            print(f"7. Heatmap {model} → {p}")

    # 8. Baseline heatmaps per model
    for model in MODELS:
        safe = model.replace("/", "_").replace(":", "_")
        bl_file = metrics_dir / f"{safe}_baseline_results.json"
        if bl_file.exists():
            bl_results = json.loads(bl_file.read_text())
            p = heatmap_baseline_bias(bl_results, model)
            print(f"8. Baseline heatmap {model} → {p}")

    # 9. DAS FSLSM vs Baseline
    das_path = metrics_dir / "das_summary.csv"
    if das_path.exists():
        df_das = pd.read_csv(das_path)
        p = das_fslsm_vs_baseline_bar(df_das)
        print(f"9. DAS FSLSM vs Baseline → {p}")

    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
