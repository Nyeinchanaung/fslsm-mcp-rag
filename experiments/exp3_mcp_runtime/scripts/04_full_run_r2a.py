from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import AGENTS_PATH, PASSIVE_R2_LOG_PATH, R2A_QUESTIONS_PATH
from experiments.exp3_mcp_runtime.core.session_runner import Exp3SessionRunner
from experiments.exp3_mcp_runtime.server.app import create_mcp_server
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.runtime_types import Condition


if __name__ == "__main__":
    idx = ToolIndex()
    idx.load()
    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    runner = Exp3SessionRunner(idx, client, passive_log_path=PASSIVE_R2_LOG_PATH)

    questions = json.loads(R2A_QUESTIONS_PATH.read_text())
    agents = json.loads(AGENTS_PATH.read_text())
    for question in questions:
        for agent in agents:
            for condition in (Condition.S0, Condition.S1A, Condition.S1B):
                runner.run_session(
                    question_id=question["question_id"],
                    question_type=question["question_type"],
                    query=question["question"],
                    profile=agent["fslsm_vector"],
                    condition=condition,
                    corpus_backed=True,
                    log_passive=True,
                )
    print("R2a full run complete.")
