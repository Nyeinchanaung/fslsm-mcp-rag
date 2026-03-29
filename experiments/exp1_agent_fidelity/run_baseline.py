"""Experiment 1 baseline runner — Non-Personalized Baseline (Task 2.8)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

import yaml

from src.agents.agent_factory import create_baseline_agents
from src.agents.ils_evaluator import run_baseline_experiment
from src.evaluation.metrics import baseline_pra_vs_all_profiles

CONFIG = yaml.safe_load(
    Path("experiments/exp1_agent_fidelity/config.yaml").read_text()
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Run only this model (short name from config.yaml)")
    args = parser.parse_args()

    profiles = json.loads(Path("data/fslsm/profiles.json").read_text())

    for model_cfg in CONFIG["models"]:
        if args.model and model_cfg["name"] != args.model:
            continue
        model_name = model_cfg["name"]
        temperature = model_cfg.get("temperature", 0.3)

        print(f"\n{'=' * 60}")
        print(f"  Baseline — Model: {model_name}")
        print(f"{'=' * 60}")

        # Step 1: Create 5 baseline agents (idempotent)
        print("\n--- Creating baseline agents ---")
        create_baseline_agents(model_name)

        # Step 2: Run ILS questionnaire
        num_trials = CONFIG["ils_questionnaire"]["num_trials"]
        print(f"\n--- Running baseline ILS questionnaire ({num_trials} trials) ---")
        results = run_baseline_experiment(
            llm_model=model_name,
            num_trials=num_trials,
            temperature=temperature,
        )

        # Step 3: Save results
        out_dir = Path("results/exp1/metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace("/", "_").replace(":", "_")
        out_file = out_dir / f"{safe_name}_baseline_results.json"
        out_file.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {len(results)} baseline records → {out_file}")

        # Step 4: Quick baseline PRA summary
        bl_pra = baseline_pra_vs_all_profiles(results, profiles)
        print(f"Baseline avg PRA vs all 16 profiles: {bl_pra['avg_pra_vs_all']:.3f} "
              f"± {bl_pra['std_pra_vs_all']:.3f}")
        print(f"Best-matching profile: {bl_pra['best_match_profile']} "
              f"(PRA={bl_pra['best_match_pra']:.3f})")
        print("Dimension bias:")
        for d, info in bl_pra["dimension_bias"].items():
            print(f"  {d}: {info['pole_label']} (mean_score={info['mean_score']:.1f})")

        # Step 5: Cost
        total_cost = sum(r.get("cost_usd", 0) for r in results)
        print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
