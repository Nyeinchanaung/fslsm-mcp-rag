from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import (
    CORE_BENCHMARK_NAME,
    PASSIVE_R2_LOG_PATH,
    RESULTS_DB_PATH,
    TOP_K_CHUNKS,
)
from experiments.exp3_mcp_runtime.core.ground_truth import (
    get_core_profile_eval_eligible,
    get_core_profile_tool_id,
    get_core_task_tool_id,
    get_optimal_tool_id,
)
from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_label
from experiments.exp3_mcp_runtime.core.retriever import D2LRetriever
from experiments.exp3_mcp_runtime.core.selector import select_retrieved_tool, select_s0_tool
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.tools.tool_registry import get_tool_by_id, s0_prompt_tokens
from experiments.exp3_mcp_runtime.runtime_types import Condition, SessionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exp3_runtime_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    benchmark TEXT NOT NULL DEFAULT 'legacy',
    condition TEXT NOT NULL,
    question_id TEXT NOT NULL,
    question_type TEXT NOT NULL,
    query TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    profile_label TEXT NOT NULL DEFAULT '',
    selected_tool_id INTEGER NOT NULL,
    selected_tool_name TEXT NOT NULL,
    task_optimal_tool_id INTEGER NOT NULL DEFAULT 0,
    task_tsa_hit INTEGER NOT NULL DEFAULT 0,
    profile_optimal_tool_id INTEGER NOT NULL DEFAULT 0,
    profile_tsa_hit INTEGER NOT NULL DEFAULT 0,
    profile_eval_eligible INTEGER NOT NULL DEFAULT 0,
    optimal_tool_id INTEGER NOT NULL,
    tsa_hit INTEGER NOT NULL,
    pts_delta REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    execution_success INTEGER NOT NULL,
    retrieved_evidence_json TEXT NOT NULL,
    final_response TEXT NOT NULL,
    tool_result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runtime_condition ON exp3_runtime_sessions(condition);
CREATE INDEX IF NOT EXISTS idx_runtime_session_id ON exp3_runtime_sessions(session_id);
"""


class Exp3SessionRunner:
    def __init__(
        self,
        tool_index: ToolIndex,
        client: MCPRuntimeClient,
        db_path: str | Path = RESULTS_DB_PATH,
        passive_log_path: str | Path = PASSIVE_R2_LOG_PATH,
    ) -> None:
        self.tool_index = tool_index
        self.client = client
        self.retriever = D2LRetriever()
        self.db_path = Path(db_path)
        self.passive_log_path = Path(passive_log_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.passive_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(_SCHEMA)
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(exp3_runtime_sessions)").fetchall()
        }
        if "benchmark" not in existing_cols:
            conn.execute(
                "ALTER TABLE exp3_runtime_sessions ADD COLUMN benchmark TEXT NOT NULL DEFAULT 'legacy'"
            )
        new_cols = {
            "profile_label": "TEXT NOT NULL DEFAULT ''",
            "task_optimal_tool_id": "INTEGER NOT NULL DEFAULT 0",
            "task_tsa_hit": "INTEGER NOT NULL DEFAULT 0",
            "profile_optimal_tool_id": "INTEGER NOT NULL DEFAULT 0",
            "profile_tsa_hit": "INTEGER NOT NULL DEFAULT 0",
            "profile_eval_eligible": "INTEGER NOT NULL DEFAULT 0",
        }
        for col_name, col_type in new_cols.items():
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE exp3_runtime_sessions ADD COLUMN {col_name} {col_type}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runtime_benchmark ON exp3_runtime_sessions(benchmark)"
        )
        conn.commit()
        conn.close()

    def run_session(
        self,
        *,
        question_id: str,
        question_type: str,
        query: str,
        profile: dict[str, Any],
        condition: Condition,
        benchmark: str = CORE_BENCHMARK_NAME,
        question_record: dict[str, Any] | None = None,
        corpus_backed: bool = True,
        log_passive: bool = False,
        session_id: str | None = None,
    ) -> SessionRecord:
        label = profile_to_label(profile)
        sid = session_id or f"{benchmark}:{question_id}:{label}"
        question_record = question_record or {}
        source_text = question_record.get("source_text", "")
        retrieval = self.retriever.retrieve(query, k=TOP_K_CHUNKS) if corpus_backed else {
            "chunk_ids": [],
            "evidence": [],
            "combined_text": "",
        }

        selected_tool_id = self._select_tool(condition, query, profile)
        core_task_tool_id = get_core_task_tool_id(question_id)
        task_optimal_tool_id = (
            core_task_tool_id if core_task_tool_id is not None else get_optimal_tool_id(question_type, profile)
        )
        core_profile_tool_id = get_core_profile_tool_id(question_id, profile)
        profile_optimal_tool_id = core_profile_tool_id if core_profile_tool_id is not None else task_optimal_tool_id
        profile_eval_eligible = get_core_profile_eval_eligible(question_id)
        selected_tool = get_tool_by_id(selected_tool_id)

        arguments: dict[str, Any] = {
            "question": query,
            "fslsm_profile": profile if condition != Condition.S0 or selected_tool_id not in {14, 15} else None,
            "question_type": question_type,
        }
        if selected_tool_id == 14:
            arguments["k"] = TOP_K_CHUNKS
        elif selected_tool_id == 15:
            arguments["max_results"] = 3
        else:
            arguments["evidence"] = retrieval["evidence"]
            arguments["source_text"] = source_text or retrieval["combined_text"]

        tool_result = self.client.execute_tool(selected_tool.mcp_name, arguments)

        input_tokens = s0_prompt_tokens() if condition == Condition.S0 else selected_tool.token_cost
        pts_delta = 0.0 if condition == Condition.S0 else (1 - (input_tokens / s0_prompt_tokens())) * 100

        record = SessionRecord(
            session_id=sid,
            benchmark=benchmark,
            condition=condition.value,
            question_id=question_id,
            question_type=question_type,
            query=query,
            profile=profile,
            profile_label=label,
            selected_tool_id=selected_tool_id,
            selected_tool_name=selected_tool.name,
            task_optimal_tool_id=task_optimal_tool_id,
            task_tsa_hit=selected_tool_id == task_optimal_tool_id,
            profile_optimal_tool_id=profile_optimal_tool_id,
            profile_tsa_hit=selected_tool_id == profile_optimal_tool_id,
            profile_eval_eligible=profile_eval_eligible,
            optimal_tool_id=task_optimal_tool_id,
            tsa_hit=selected_tool_id == task_optimal_tool_id,
            pts_delta=pts_delta,
            input_tokens=input_tokens,
            latency_ms=float(tool_result["latency_ms"]),
            execution_success=bool(tool_result["execution_success"]),
            retrieved_evidence=retrieval["evidence"],
            final_response=tool_result["tool_output"],
            tool_result=tool_result,
        )

        self._log_record(record)
        if log_passive:
            self._log_passive(record)
        return record

    def _select_tool(
        self,
        condition: Condition,
        query: str,
        profile: dict[str, Any],
    ) -> int:
        if condition == Condition.S0:
            return select_s0_tool(query)
        if condition == Condition.S1A:
            return select_retrieved_tool(index=self.tool_index, query=query, k=5)
        return select_retrieved_tool(
            index=self.tool_index,
            query=query,
            profile=profile,
            use_profile=True,
            k=5,
        )

    def _log_record(self, record: SessionRecord) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO exp3_runtime_sessions
            (session_id, benchmark, condition, question_id, question_type, query, profile_json,
             profile_label, selected_tool_id, selected_tool_name, task_optimal_tool_id,
             task_tsa_hit, profile_optimal_tool_id, profile_tsa_hit, profile_eval_eligible,
             optimal_tool_id, tsa_hit, pts_delta,
             input_tokens, latency_ms, execution_success, retrieved_evidence_json,
             final_response, tool_result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.session_id,
                record.benchmark,
                record.condition,
                record.question_id,
                record.question_type,
                record.query,
                json.dumps(record.profile),
                record.profile_label,
                record.selected_tool_id,
                record.selected_tool_name,
                record.task_optimal_tool_id,
                int(record.task_tsa_hit),
                record.profile_optimal_tool_id,
                int(record.profile_tsa_hit),
                int(record.profile_eval_eligible),
                record.optimal_tool_id,
                int(record.tsa_hit),
                record.pts_delta,
                record.input_tokens,
                record.latency_ms,
                int(record.execution_success),
                json.dumps(record.retrieved_evidence),
                record.final_response,
                json.dumps(record.tool_result),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def _log_passive(self, record: SessionRecord) -> None:
        with open(self.passive_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
