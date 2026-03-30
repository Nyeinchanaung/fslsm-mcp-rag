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
    compute_baseline_natural_style,
    compute_das_for_results,
    compute_per_question_alignment,
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
    das_rows: list[dict] = []
    baseline_style_rows: list[dict] = []
    pq_rows: list[dict] = []

    questionnaire = json.loads(Path("data/fslsm/ils_questionnaire.json").read_text())
    raw_dir = Path("results/exp1/raw_responses")

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
                    "knowledge_level": "ALL",
                    "dimension": dim,
                    "pra": pra["per_dimension"][dim],
                    "ties": pra["ties_per_dimension"][dim],
                })
            all_rows.append({
                "model": model,
                "knowledge_level": "ALL",
                "dimension": "overall_4d",
                "pra": pra["overall_4d"],
                "ties": sum(pra["ties_per_dimension"].values()),
            })

            # PRA by knowledge_level
            for level, level_pra in pra_by_knowledge_level(results).items():
                all_rows.append({
                    "model": model,
                    "knowledge_level": level,
                    "dimension": "overall_4d",
                    "pra": level_pra["overall_4d"],
                    "ties": sum(level_pra["ties_per_dimension"].values()),
                })

            # Cost
            cost_rows.append({"model": model, **cost_summary(results)})

            # DAS (formula-based)
            print(f"  Computing DAS for {model}…")
            for row in compute_das_for_results(results):
                das_rows.append({"model": model, **row})

            # Per-question alignment
            print(f"  Computing per-question alignment for {model}…")
            pq_rows.extend(
                compute_per_question_alignment(model, results, raw_dir, questionnaire)
            )

        # ── Baseline natural style ──
        baseline_file = Path(f"results/exp1/metrics/{safe}_baseline_results.json")
        if not baseline_file.exists():
            print(f"Skipping Baseline {model} — {baseline_file} not found")
        else:
            bl_results = json.loads(baseline_file.read_text())
            baseline_style_rows.append(
                compute_baseline_natural_style(model, bl_results)
            )
            cost_rows.append({"model": f"{model} (baseline)", **cost_summary(bl_results)})

    # ── Export everything ──
    out_dir = Path("results/exp1/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pra = pd.DataFrame(all_rows)
    pra_path = out_dir / "pra_das_summary.csv"
    df_pra.to_csv(pra_path, index=False)
    print(f"PRA → {pra_path}")

    df_cost = pd.DataFrame(cost_rows)
    cost_path = out_dir / "cost_summary.csv"
    df_cost.to_csv(cost_path, index=False)
    print(f"Cost → {cost_path}")

    df_baseline_style = pd.DataFrame(baseline_style_rows)
    if not df_baseline_style.empty:
        style_path = out_dir / "baseline_natural_style.csv"
        df_baseline_style.to_csv(style_path, index=False)
        print(f"Baseline natural style → {style_path}")

    df_pq = pd.DataFrame(pq_rows)
    if not df_pq.empty:
        pq_path = out_dir / "per_question_alignment.csv"
        df_pq.to_csv(pq_path, index=False)
        print(f"Per-question alignment → {pq_path}")

    # Print summaries
    print("\n=== FSLSM PRA (overall_4d, ALL knowledge levels) ===")
    overview = df_pra[
        (df_pra["dimension"] == "overall_4d") & (df_pra["knowledge_level"] == "ALL")
    ]
    if not overview.empty:
        print(overview.to_string(index=False))

    if not df_baseline_style.empty:
        print("\n=== Baseline Natural Learning Style ===")
        print(df_baseline_style[["model", "detected_style",
                                  "act_ref_mean_score", "sen_int_mean_score",
                                  "vis_ver_mean_score", "seq_glo_mean_score"]
              ].to_string(index=False))

    if not df_cost.empty:
        print(f"\n=== Cost Summary ===")
        print(df_cost.to_string(index=False))

    # ── DAS export ──
    df_das_raw = pd.DataFrame(das_rows)
    if not df_das_raw.empty:
        das_long_rows: list[dict] = []
        for _, r in df_das_raw.iterrows():
            for d in FSLSM_DIMENSIONS:
                das_long_rows.append({
                    "model": r["model"],
                    "knowledge_level": r["knowledge_level"],
                    "dimension": d,
                    "das": r[f"das_{d}"],
                })
            das_long_rows.append({
                "model": r["model"],
                "knowledge_level": r["knowledge_level"],
                "dimension": "overall_4d",
                "das": r["das_overall"],
            })
        df_das = pd.DataFrame(das_long_rows)
        df_das_agg = (
            df_das.groupby(
                ["model", "knowledge_level", "dimension"], sort=False
            )["das"]
            .mean()
            .reset_index()
        )
        # Add ALL-level rows
        all_das_rows = []
        for model in MODELS:
            m_data = df_das[df_das["model"] == model]
            if m_data.empty:
                continue
            for d in list(FSLSM_DIMENSIONS) + ["overall_4d"]:
                mean_val = m_data[m_data["dimension"] == d]["das"].mean()
                all_das_rows.append({
                    "model": model,
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
        ][["model", "das"]]
        print(das_overview.to_string(index=False))
    else:
        df_das_agg = pd.DataFrame()

    return df_pra, df_baseline_style, df_cost, df_das_agg, df_pq


if __name__ == "__main__":
    run_analysis()
