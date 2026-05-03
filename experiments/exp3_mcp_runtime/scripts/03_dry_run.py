from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import CORE_BENCHMARK_NAME, CORE_QUESTIONS_PATH, DRY_RUN_N, S1B_MIN_LIFT_OVER_S1A, MIN_PTS
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles
from experiments.exp3_mcp_runtime.core.run_artifacts import get_run_artifacts
from experiments.exp3_mcp_runtime.core.session_runner import Exp3SessionRunner
from experiments.exp3_mcp_runtime.server.app import create_mcp_server
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.runtime_types import Condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an isolated Exp3-Core dry run.")
    parser.add_argument("--run-id", default=None, help="Run folder id under results/runs/.")
    parser.add_argument("--limit", type=int, default=DRY_RUN_N, help="Number of core questions to dry run.")
    args = parser.parse_args()

    artifacts = get_run_artifacts(args.run_id, prefix="dry_exp3_core")
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)

    idx = ToolIndex()
    try:
        idx.load()
    except Exception:
        idx.build()
        idx.save()

    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    runner = Exp3SessionRunner(
        idx,
        client,
        db_path=artifacts.db_path,
        passive_log_path=artifacts.passive_log_path,
    )

    questions = json.loads(CORE_QUESTIONS_PATH.read_text())[: args.limit]
    profiles = load_canonical_profiles()

    rows = []
    for i, question in enumerate(questions):
        profile = profiles[i % len(profiles)]["fslsm_vector"]
        for condition in (Condition.S0, Condition.S1A, Condition.S1B):
            rows.append(
                runner.run_session(
                    question_id=question["question_id"],
                    question_type=question["question_family"],
                    query=question["question"],
                    profile=profile,
                    condition=condition,
                    benchmark=CORE_BENCHMARK_NAME,
                    question_record=question,
                    corpus_backed=question["grounding_mode"] == "d2l",
                    log_passive=False,
                )
            )

    def mean(condition: Condition, attr: str) -> float:
        vals = [getattr(row, attr) for row in rows if row.condition == condition.value]
        return sum(vals) / len(vals)

    s0_tsa = mean(Condition.S0, "tsa_hit")
    s1a_tsa = mean(Condition.S1A, "tsa_hit")
    s1b_tsa = mean(Condition.S1B, "tsa_hit")
    s1a_pts = mean(Condition.S1A, "pts_delta")
    s1b_pts = mean(Condition.S1B, "pts_delta")
    expected_pairs = len(questions)
    observed_pairs = len({row.session_id for row in rows})
    print(
        {
            "run_id": artifacts.run_id,
            "db_path": str(artifacts.db_path),
            "S0_task_tsa": s0_tsa,
            "S1a_task_tsa": s1a_tsa,
            "S1b_task_tsa": s1b_tsa,
            "S1a_pts": s1a_pts,
            "S1b_pts": s1b_pts,
            "matched_pairs": observed_pairs,
            "expected_pairs": expected_pairs,
        }
    )
    if observed_pairs != expected_pairs:
        raise SystemExit("Dry run failed: matched-pair session count is inconsistent.")
    draft_dataset = any(question.get("manual_review_status") != "reviewed" for question in questions)
    if (s1b_tsa - s1a_tsa) < S1B_MIN_LIFT_OVER_S1A:
        if draft_dataset:
            print("Dry run warning: S1b lift over S1a is below threshold on the draft Exp3-Core dataset.")
        else:
            raise SystemExit("Dry run failed: S1b lift over S1a is below threshold.")
    if s1a_pts < MIN_PTS or s1b_pts < MIN_PTS:
        if draft_dataset:
            print("Dry run warning: PTS is below threshold on the draft Exp3-Core dataset.")
        else:
            raise SystemExit("Dry run failed: PTS below threshold.")
    print("Dry run passed.")
