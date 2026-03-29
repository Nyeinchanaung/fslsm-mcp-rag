"""Analysis script for Experiment 1 — computes PRA/DAS and exports CSV."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from config.constants import FSLSM_DIMENSIONS
from src.evaluation.metrics import (
    baseline_pra_vs_all_profiles,
    compute_baseline_das,
    compute_das_for_results,
    cost_summary,
    pra_by_knowledge_level,
    profile_recovery_accuracy,
)

# Models to analyze (must match safe filenames in results/exp1/metrics/)
MODELS = [
    "gpt-4.1-mini",
    "claude-sonnet-4-20250514",
    "llama3.1:8b",
    "qwen2.5:7b",
    "gemma2:9b",
    "gemma3:12b",
]


def run_analysis():
    """Load per-model results, compute PRA + DAS + cost, export CSV."""
    all_rows: list[dict] = []
    cost_rows: list[dict] = []
    baseline_rows: list[dict] = []
    das_rows: list[dict] = []

    profiles = json.loads(Path("data/fslsm/profiles.json").read_text())

    for model in MODELS:
        safe = model.replace("/", "_").replace(":", "_")

        # ── FSLSM analysis ──
        results_file = Path(f"results/exp1/metrics/{safe}_results.json")
        if not results_file.exists():
            print(f"Skipping FSLSM {model} — {results_file} not found")
        else:
            results = json.loads(results_file.read_text())

            # PRA overall
            pra = profile_recovery_accuracy(results)
            for dim in FSLSM_DIMENSIONS:
                all_rows.append({
                    "model": model,
                    "condition": "FSLSM",
                    "knowledge_level": "ALL",
                    "dimension": dim,
                    "pra": pra["per_dimension"][dim],
                    "ties": pra["ties_per_dimension"][dim],
                })
            all_rows.append({
                "model": model,
                "condition": "FSLSM",
                "knowledge_level": "ALL",
                "dimension": "overall_4d",
                "pra": pra["overall_4d"],
                "ties": sum(pra["ties_per_dimension"].values()),
            })

            # PRA by knowledge_level
            for level, level_pra in pra_by_knowledge_level(results).items():
                all_rows.append({
                    "model": model,
                    "condition": "FSLSM",
                    "knowledge_level": level,
                    "dimension": "overall_4d",
                    "pra": level_pra["overall_4d"],
                    "ties": sum(level_pra["ties_per_dimension"].values()),
                })

            # Cost
            cost_rows.append({"model": model, "condition": "FSLSM",
                              **cost_summary(results)})

            # DAS (formula-based)
            print(f"  Computing DAS for {model}…")
            for row in compute_das_for_results(results):
                das_rows.append({"model": model, "condition": "FSLSM", **row})

        # ── Baseline analysis ──
        baseline_file = Path(f"results/exp1/metrics/{safe}_baseline_results.json")
        if not baseline_file.exists():
            print(f"Skipping Baseline {model} — {baseline_file} not found")
        else:
            bl_results = json.loads(baseline_file.read_text())
            bl_pra = baseline_pra_vs_all_profiles(bl_results, profiles)

            # Overall baseline PRA
            all_rows.append({
                "model": model,
                "condition": "Baseline",
                "knowledge_level": "ALL",
                "dimension": "overall_4d",
                "pra": bl_pra["avg_pra_vs_all"],
                "ties": 0,
            })

            # Baseline by knowledge level
            for level in ["beginner", "intermediate", "advanced", "general"]:
                level_results = [
                    r for r in bl_results
                    if (r.get("knowledge_level") or "general") == level
                ]
                if level_results:
                    level_bl_pra = baseline_pra_vs_all_profiles(
                        level_results, profiles
                    )
                    all_rows.append({
                        "model": model,
                        "condition": "Baseline",
                        "knowledge_level": level,
                        "dimension": "overall_4d",
                        "pra": level_bl_pra["avg_pra_vs_all"],
                        "ties": 0,
                    })

            # Detailed baseline report
            baseline_rows.append({
                "model": model,
                "avg_pra_vs_all": bl_pra["avg_pra_vs_all"],
                "std_pra_vs_all": bl_pra["std_pra_vs_all"],
                "best_match_profile": bl_pra["best_match_profile"],
                "best_match_pra": bl_pra["best_match_pra"],
                **{f"bias_{d}": bl_pra["dimension_bias"][d]["pole_label"]
                   for d in FSLSM_DIMENSIONS},
                **{f"bias_{d}_score": bl_pra["dimension_bias"][d]["mean_score"]
                   for d in FSLSM_DIMENSIONS},
            })

            cost_rows.append({"model": model, "condition": "Baseline",
                              **cost_summary(bl_results)})

            # Baseline DAS
            print(f"  Computing baseline DAS for {model}…")
            for row in compute_baseline_das(bl_results, profiles):
                das_rows.append({"model": model, "condition": "Baseline", **row})

    # ── Export everything ──
    out_dir = Path("results/exp1/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pra = pd.DataFrame(all_rows)
    pra_path = out_dir / "pra_das_summary.csv"
    df_pra.to_csv(pra_path, index=False)
    print(f"PRA → {pra_path}")

    df_baseline = pd.DataFrame(baseline_rows)
    if not df_baseline.empty:
        baseline_path = out_dir / "baseline_analysis.csv"
        df_baseline.to_csv(baseline_path, index=False)
        print(f"Baseline → {baseline_path}")

    df_cost = pd.DataFrame(cost_rows)
    cost_path = out_dir / "cost_summary.csv"
    df_cost.to_csv(cost_path, index=False)
    print(f"Cost → {cost_path}")

    # Print summaries
    print("\n=== FSLSM vs Baseline PRA ===")
    overview = df_pra[df_pra["dimension"] == "overall_4d"]
    if not overview.empty:
        print(overview.to_string(index=False))

    if not df_baseline.empty:
        print("\n=== Baseline Natural Bias ===")
        print(df_baseline.to_string(index=False))

    if not df_cost.empty:
        print(f"\n=== Cost Summary ===")
        print(df_cost.to_string(index=False))

    # ── DAS export ──
    df_das_raw = pd.DataFrame(das_rows)
    if not df_das_raw.empty:
        # Melt to long format: model, condition, knowledge_level, dimension, das
        das_long_rows: list[dict] = []
        for _, r in df_das_raw.iterrows():
            for d in FSLSM_DIMENSIONS:
                das_long_rows.append({
                    "model": r["model"],
                    "condition": r["condition"],
                    "knowledge_level": r["knowledge_level"],
                    "dimension": d,
                    "das": r[f"das_{d}"],
                })
            das_long_rows.append({
                "model": r["model"],
                "condition": r["condition"],
                "knowledge_level": r["knowledge_level"],
                "dimension": "overall_4d",
                "das": r["das_overall"],
            })
        df_das = pd.DataFrame(das_long_rows)
        # Aggregate: mean DAS per model / condition / knowledge_level / dimension
        df_das_agg = (
            df_das.groupby(
                ["model", "condition", "knowledge_level", "dimension"], sort=False
            )["das"]
            .mean()
            .reset_index()
        )
        # Add ALL-level rows
        all_das_rows = []
        for model in MODELS:
            for cond in ["FSLSM", "Baseline"]:
                m_data = df_das[
                    (df_das["model"] == model) & (df_das["condition"] == cond)
                ]
                if m_data.empty:
                    continue
                for d in list(FSLSM_DIMENSIONS) + ["overall_4d"]:
                    mean_val = m_data[m_data["dimension"] == d]["das"].mean()
                    all_das_rows.append({
                        "model": model,
                        "condition": cond,
                        "knowledge_level": "ALL",
                        "dimension": d,
                        "das": mean_val,
                    })
        df_das_agg = pd.concat(
            [pd.DataFrame(all_das_rows), df_das_agg], ignore_index=True
        )
        das_path = out_dir / "das_summary.csv"
        df_das_agg.to_csv(das_path, index=False)
        print(f"DAS → {das_path}")

        print("\n=== DAS (overall_4d, ALL knowledge levels) ===")
        das_overview = df_das_agg[
            (df_das_agg["dimension"] == "overall_4d")
            & (df_das_agg["knowledge_level"] == "ALL")
        ][["model", "condition", "das"]]
        print(das_overview.to_string(index=False))
    else:
        df_das_agg = pd.DataFrame()

    return df_pra, df_baseline, df_cost, df_das_agg


if __name__ == "__main__":
    run_analysis()
