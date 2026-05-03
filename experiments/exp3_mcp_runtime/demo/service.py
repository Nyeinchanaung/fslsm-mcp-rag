from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import (
    CHUNKS_PATH,
    CORE_ANSWER_KEY_PATH,
    CORE_BENCHMARK_NAME,
    CORE_QUESTIONS_PATH,
    DEMO_REPLAY_PATH,
    LEGACY_REPLAY_PATH,
    METRICS_JSON_PATH,
    TOOL_INDEX_PATH,
    TOOL_META_PATH,
)
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles
from experiments.exp3_mcp_runtime.core.session_runner import Exp3SessionRunner
from experiments.exp3_mcp_runtime.server.app import create_mcp_server
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.runtime_types import Condition


@lru_cache(maxsize=1)
def get_runtime() -> Exp3SessionRunner:
    idx = ToolIndex()
    try:
        idx.load()
    except Exception:
        idx.build()
        idx.save()
    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    return Exp3SessionRunner(idx, client)


def load_profiles() -> list[dict[str, Any]]:
    return load_canonical_profiles()


@lru_cache(maxsize=1)
def load_core_questions() -> list[dict[str, Any]]:
    if CORE_QUESTIONS_PATH.exists():
        return json.loads(CORE_QUESTIONS_PATH.read_text())
    return []


@lru_cache(maxsize=1)
def load_core_answer_key() -> dict[str, dict[str, Any]]:
    if CORE_ANSWER_KEY_PATH.exists():
        return json.loads(CORE_ANSWER_KEY_PATH.read_text())
    return {}


def get_core_question(question_id: str) -> dict[str, Any] | None:
    normalized = question_id.strip().upper()
    for question in load_core_questions():
        if question["question_id"].upper() == normalized:
            return question
    return None


def infer_question_type(question: str) -> str:
    q = question.lower()
    if "compare" in q:
        return "compare"
    if "summar" in q:
        return "summarize"
    if "latest" in q or "recent" in q:
        return "search"
    if "workflow" in q or "process" in q:
        return "synthesize_workflow"
    return "explain_relationship"


def run_demo_session(
    question: str,
    profile: dict[str, Any],
    condition: str,
    question_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner = get_runtime()
    if question_record:
        question_id = question_record["question_id"]
        question_type = question_record["question_family"]
        query = question_record["question"]
        corpus_backed = question_record["grounding_mode"] == "d2l"
        benchmark = CORE_BENCHMARK_NAME
    else:
        question_id = "demo_live"
        question_type = infer_question_type(question)
        query = question
        corpus_backed = True
        benchmark = "demo_live"

    record = runner.run_session(
        question_id=question_id,
        question_type=question_type,
        query=query,
        profile=profile,
        condition=Condition(condition),
        benchmark=benchmark,
        question_record=question_record,
        corpus_backed=corpus_backed,
        log_passive=False,
    )
    return record.to_dict()


def load_replays(limit: int = 25) -> list[dict[str, Any]]:
    for path in (DEMO_REPLAY_PATH, LEGACY_REPLAY_PATH):
        path = Path(path)
        if path.exists():
            rows = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
                    if len(rows) >= limit:
                        break
            return rows
    return []


def load_metrics() -> dict[str, Any]:
    if METRICS_JSON_PATH.exists():
        return json.loads(METRICS_JSON_PATH.read_text())
    return {}


def _count_jsonl_rows(path: Path, limit: int = 1000) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
            if count >= limit:
                break
    return count


def get_demo_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "profiles_loaded": False,
        "profile_count": 0,
        "core_dataset_available": CORE_QUESTIONS_PATH.exists(),
        "chunks_available": CHUNKS_PATH.exists(),
        "tool_index_available": TOOL_INDEX_PATH.exists() and TOOL_META_PATH.exists(),
        "openai_key_loaded": bool(os.environ.get("OPENAI_API_KEY")),
        "tavily_key_loaded": bool(os.environ.get("TAVILY_API_KEY")),
        "metrics_available": METRICS_JSON_PATH.exists(),
        "replay_count": 0,
        "legacy_replay_count": 0,
        "runtime_ready": False,
        "fastmcp_active": False,
        "registered_tool_count": 0,
        "status_errors": [],
    }

    try:
        profiles = load_profiles()
        status["profiles_loaded"] = True
        status["profile_count"] = len(profiles)
    except Exception as exc:
        status["status_errors"].append(f"profiles: {exc}")

    status["replay_count"] = _count_jsonl_rows(DEMO_REPLAY_PATH, limit=5000)
    status["legacy_replay_count"] = _count_jsonl_rows(LEGACY_REPLAY_PATH, limit=5000)

    try:
        runtime = get_runtime()
        tools = runtime.client.list_tools()
        status["runtime_ready"] = True
        status["registered_tool_count"] = len(tools)
        status["fastmcp_active"] = getattr(runtime.client.server, "fastmcp_app", None) is not None
    except Exception as exc:
        status["status_errors"].append(f"runtime: {exc}")

    return status
