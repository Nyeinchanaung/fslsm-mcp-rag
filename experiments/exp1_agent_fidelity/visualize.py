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
    cost_per_model_bar,
    das_comparison_bar,
    heatmap_profiles,
    knowledge_level_comparison,
    model_comparison_bar,
    per_question_alignment_heatmap,
)

MODELS = [
    "gpt-4.1-mini",
    "claude-sonnet-4-20250514",
    "llama3.1:8b",
    "qwen2.5:7b",
    "gemma2:9b",
    "gemma3:12b",
    "qwen2.5:3b",
    "gemma3:4b",
    "llama3.2:3b",
    "phi4-mini",
    "llama3.2:1b",
    "mistral:7b",
    "qwen2.5:1.5b",
]


def main():
    metrics_dir = Path("results/exp1/metrics")

    # Load CSVs
    df_pra = pd.read_csv(metrics_dir / "pra_das_summary.csv")
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

    # 4. DAS comparison bar
    das_path = metrics_dir / "das_summary.csv"
    if das_path.exists():
        df_das = pd.read_csv(das_path)
        p = das_comparison_bar(df_das)
        print(f"4. DAS comparison bar → {p}")
    else:
        print("4. DAS comparison bar — skipped (das_summary.csv not found)")

    # 5. Heatmaps per model (from FSLSM results JSON)
    for model in MODELS:
        safe = model.replace("/", "_").replace(":", "_")
        results_file = metrics_dir / f"{safe}_results.json"
        if results_file.exists():
            results = json.loads(results_file.read_text())
            p = heatmap_profiles(results, model)
            print(f"5. Heatmap {model} → {p}")

    # 6. Per-question alignment heatmap
    pq_path = metrics_dir / "per_question_alignment.csv"
    if pq_path.exists():
        df_pq = pd.read_csv(pq_path)
        p = per_question_alignment_heatmap(df_pq)
        print(f"6. Per-question alignment heatmap → {p}")
    else:
        print("6. Per-question alignment heatmap — skipped (per_question_alignment.csv not found)")

    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
