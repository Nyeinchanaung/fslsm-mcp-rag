from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import TRANSFER_BENCHMARK_NAME, TRANSFER_QUESTIONS_PATH
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles
from experiments.exp3_mcp_runtime.core.run_artifacts import get_run_artifacts
from experiments.exp3_mcp_runtime.core.session_runner import Exp3SessionRunner
from experiments.exp3_mcp_runtime.server.app import create_mcp_server
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.runtime_types import Condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the isolated R2a transfer benchmark.")
    parser.add_argument("--run-id", default=None, help="Run folder id under results/runs/.")
    parser.add_argument("--no-passive", action="store_true", help="Disable passive replay JSONL logging.")
    args = parser.parse_args()

    artifacts = get_run_artifacts(args.run_id, prefix="r2a_transfer")
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)

    idx = ToolIndex()
    idx.load()
    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    runner = Exp3SessionRunner(
        idx,
        client,
        db_path=artifacts.db_path,
        passive_log_path=artifacts.passive_log_path,
    )

    questions = json.loads(TRANSFER_QUESTIONS_PATH.read_text())
    profiles = load_canonical_profiles()
    for question in questions:
        for profile in profiles:
            for condition in (Condition.S0, Condition.S1A, Condition.S1B):
                runner.run_session(
                    question_id=question["question_id"],
                    question_type=question["question_type"],
                    query=question["question"],
                    profile=profile["fslsm_vector"],
                    condition=condition,
                    benchmark=TRANSFER_BENCHMARK_NAME,
                    question_record=question,
                    corpus_backed=True,
                    log_passive=not args.no_passive,
                )
    print(f"R2a transfer full run complete: {artifacts.run_dir}")
