"""Experiment 1 runner — Virtual Student Agent Fidelity (RQ2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml

from src.agents.agent_factory import create_agents
from src.agents.ils_evaluator import run_experiment1
from src.evaluation.metrics import profile_recovery_accuracy

CONFIG = yaml.safe_load(
    Path("experiments/exp1_agent_fidelity/config.yaml").read_text()
)


def main():
    for model_cfg in CONFIG["models"]:
        model_name = model_cfg["name"]
        temperature = model_cfg.get("temperature", 0.3)

        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"  (LiteLLM handles provider routing automatically)")
        print(f"{'=' * 60}")

        # Step 1: Create 80 agents (48 leveled + 32 general)
        print("\n--- Creating agents ---")
        create_agents(model_name)

        # Step 2: Run ILS questionnaire
        num_trials = CONFIG["ils_questionnaire"]["num_trials"]
        print(f"\n--- Running ILS questionnaire ({num_trials} trials) ---")
        results = run_experiment1(
            llm_model=model_name,
            num_trials=num_trials,
            temperature=temperature,
        )

        # Step 3: Save results
        out_dir = Path("results/exp1/metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace("/", "_").replace(":", "_")
        out_file = out_dir / f"{safe_name}_results.json"
        out_file.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {len(results)} records → {out_file}")

        # Step 4: Quick PRA summary
        pra = profile_recovery_accuracy(results)
        print(f"PRA (overall 4D): {pra['overall_4d']:.3f}")
        for dim, score in pra["per_dimension"].items():
            print(f"  {dim}: {score:.3f}")

        # Step 5: Cost summary
        total_cost = sum(r.get("cost_usd", 0) for r in results)
        print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
