from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import (
    CHUNKS_PATH,
    CORE_ANSWER_KEY_PATH,
    CORE_BENCHMARK_NAME,
    CORE_QUESTIONS_PATH,
    DEMO_REPLAY_PATH,
    LEGACY_REPLAY_PATH,
    METRICS_JSON_PATH,
    METRICS_JSON_FILENAME,
    REPO_ROOT,
    RUNS_DIR,
    TOOL_INDEX_PATH,
    TOOL_META_PATH,
)
from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_label
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles
from experiments.exp3_mcp_runtime.core.session_runner import Exp3SessionRunner
from experiments.exp3_mcp_runtime.server.app import create_mcp_server
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.runtime_types import Condition


FINAL_EXP3_RUN_ID = "exp3_core_real_20260503_1"
FINAL_EXP3_RUN_DIR = RUNS_DIR / FINAL_EXP3_RUN_ID
FINAL_EXP3_METRICS_PATH = FINAL_EXP3_RUN_DIR / METRICS_JSON_FILENAME
FINAL_EXP3_REPLAY_PATH = FINAL_EXP3_RUN_DIR / "exp2_r2_passive_log.jsonl"

EXP1_METRICS_DIR = REPO_ROOT / "results" / "exp1" / "metrics"
EXP1_FIGURES_DIR = REPO_ROOT / "experiments" / "exp1_agent_fidelity" / "results" / "exp1" / "final_defense_figures"
EXP2_RESULTS_DIR = REPO_ROOT / "experiments" / "exp2_tutor_personalization" / "results"
EXP2_FIGURES_DIR = EXP2_RESULTS_DIR / "final_defense_figures"
EXP3_FIGURES_DIR = FINAL_EXP3_RUN_DIR / "final_defense_figures"
REPLAY_PATHS = (FINAL_EXP3_REPLAY_PATH, DEMO_REPLAY_PATH, LEGACY_REPLAY_PATH)

CHITCHAT_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "who are you",
}

COURSE_SCOPE_KEYWORDS = {
    "activation",
    "adam",
    "attention",
    "backprop",
    "batch normalization",
    "classification",
    "cnn",
    "convolution",
    "d2l",
    "deep learning",
    "dropout",
    "embedding",
    "gradient",
    "gru",
    "learning rate",
    "linear regression",
    "logistic regression",
    "loss",
    "machine learning",
    "ml",
    "model",
    "mxnet",
    "neural",
    "optimizer",
    "overfitting",
    "pytorch",
    "regularization",
    "resnet",
    "rnn",
    "self-attention",
    "softmax",
    "tensorflow",
    "transformer",
    "vgg",
}


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


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def is_chitchat_question(question: str) -> bool:
    normalized = _normalize_question(question).strip("?!.,")
    return normalized in CHITCHAT_PATTERNS or len(normalized.split()) <= 2 and normalized in CHITCHAT_PATTERNS


def should_use_corpus_for_custom_question(question: str) -> bool:
    normalized = _normalize_question(question)
    if is_chitchat_question(question):
        return False
    if "latest" in normalized or "recent" in normalized or "current" in normalized:
        return False
    return any(keyword in normalized for keyword in COURSE_SCOPE_KEYWORDS)


def build_out_of_scope_demo_response(
    question: str,
    profile: dict[str, Any],
    condition: str,
    reason: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile_label = profile_to_label(profile)
    message = (
        "This demo is scoped to D2L machine-learning tutoring and Exp3 tool selection. "
        "I skipped corpus retrieval because the custom question does not appear to need "
        "D2L evidence. Ask a machine-learning or D2L-related question to run the MCP "
        "tool pipeline."
    )
    tool_result = {
        "tool_id": 0,
        "tool_name": "Demo Scope Guard",
        "tool_output": message,
        "evidence": [],
        "sources": [],
        "latency_ms": (time.perf_counter() - started_at) * 1000,
        "token_cost_estimate": 0,
        "execution_success": True,
        "metadata": {"reason": reason, "profile_used_post_selection": False},
    }
    return {
        "session_id": f"demo_live:out_of_scope:{profile_label}",
        "benchmark": "demo_live",
        "condition": condition,
        "question_id": "demo_live",
        "question_type": "out_of_scope",
        "query": question,
        "profile": profile,
        "profile_label": profile_label,
        "selected_tool_id": 0,
        "selected_tool_name": "Demo Scope Guard",
        "task_optimal_tool_id": 0,
        "task_tsa_hit": False,
        "profile_optimal_tool_id": 0,
        "profile_tsa_hit": False,
        "profile_eval_eligible": False,
        "optimal_tool_id": 0,
        "tsa_hit": False,
        "pts_delta": 0.0,
        "input_tokens": 0,
        "latency_ms": tool_result["latency_ms"],
        "execution_success": True,
        "retrieved_evidence": [],
        "final_response": message,
        "tool_result": tool_result,
    }


def run_demo_session(
    question: str,
    profile: dict[str, Any],
    condition: str,
    question_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        corpus_backed = should_use_corpus_for_custom_question(question)
        benchmark = "demo_live"
        if not corpus_backed and question_type != "search":
            return build_out_of_scope_demo_response(
                question=query,
                profile=profile,
                condition=condition,
                reason="custom_question_outside_course_scope",
            )

    runner = get_runtime()
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
    for path in REPLAY_PATHS:
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
    if FINAL_EXP3_METRICS_PATH.exists():
        return json.loads(FINAL_EXP3_METRICS_PATH.read_text())
    if METRICS_JSON_PATH.exists():
        return json.loads(METRICS_JSON_PATH.read_text())
    return {}


def _existing_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.png"))


@lru_cache(maxsize=1)
def load_exp1_summary() -> dict[str, Any]:
    pra_path = EXP1_METRICS_DIR / "pra_das_summary.csv"
    das_path = EXP1_METRICS_DIR / "das_summary.csv"
    if not pra_path.exists() or not das_path.exists():
        return {}

    pra = pd.read_csv(pra_path)
    das = pd.read_csv(das_path)
    pra_overall = pra[(pra["dimension"] == "overall_4d") & (pra["knowledge_level"] == "ALL")][["model", "pra"]]
    das_overall = das[(das["dimension"] == "overall_4d") & (das["knowledge_level"] == "ALL")][["model", "das"]]
    summary = pra_overall.merge(das_overall, on="model", how="inner").sort_values(["pra", "das"], ascending=False)
    summary["h2_pra_pass"] = summary["pra"] >= 0.82
    summary["h2_das_pass"] = summary["das"] >= 0.75
    summary["h2_both_pass"] = summary["h2_pra_pass"] & summary["h2_das_pass"]

    return {
        "n_models": int(summary["model"].nunique()),
        "mean_pra": float(summary["pra"].mean()),
        "mean_das": float(summary["das"].mean()),
        "pra_pass_n": int(summary["h2_pra_pass"].sum()),
        "das_pass_n": int(summary["h2_das_pass"].sum()),
        "both_pass_n": int(summary["h2_both_pass"].sum()),
        "top_models": summary.head(5).to_dict(orient="records"),
        "table": summary.to_dict(orient="records"),
        "figures": [str(p) for p in _existing_images(EXP1_FIGURES_DIR)],
    }


@lru_cache(maxsize=1)
def load_exp2_summary() -> dict[str, Any]:
    summary_path = EXP2_RESULTS_DIR / "exp2_results_summary.json"
    pairwise_path = EXP2_RESULTS_DIR / "pairwise" / "summary_overall.json"
    metrics_path = EXP2_RESULTS_DIR / "exp2_session_metrics.csv"
    if not summary_path.exists():
        return {}

    summary = json.loads(summary_path.read_text())
    pairwise = json.loads(pairwise_path.read_text()) if pairwise_path.exists() else {}
    session_count = 0
    profile_count = 0
    question_count = 0
    if metrics_path.exists():
        df = pd.read_csv(metrics_path, usecols=["profile_label", "question_id"])
        session_count = int(len(df))
        profile_count = int(df["profile_label"].nunique())
        question_count = int(df["question_id"].nunique())

    rows = []
    for metric in ["SCS", "Eng", "RR", "CR@5", "CR@10", "ER"]:
        sig = summary.get("significance", {}).get(metric, {})
        rows.append({
            "metric": metric,
            "r0_mean": sig.get("r0_mean", summary.get("metrics", {}).get(metric, {}).get("R0", {}).get("mean")),
            "r1_mean": sig.get("r1_mean", summary.get("metrics", {}).get(metric, {}).get("R1", {}).get("mean")),
            "delta": sig.get("mean_diff"),
            "cohens_d": sig.get("cohens_d"),
            "p_value": sig.get("p_value"),
            "significant": sig.get("significant"),
        })

    return {
        "n_sessions": session_count,
        "n_pairs": int(summary.get("n_matched_pairs", 0)),
        "n_profiles": profile_count,
        "n_questions": question_count,
        "pairwise": pairwise,
        "metrics_table": rows,
        "figures": [str(p) for p in _existing_images(EXP2_FIGURES_DIR)],
    }


@lru_cache(maxsize=1)
def load_exp3_summary() -> dict[str, Any]:
    metrics = load_metrics()
    benchmark = metrics.get("benchmarks", {}).get("exp3_core", {})
    conditions = benchmark.get("conditions", {})
    paired = benchmark.get("paired", {})
    rows = [{"condition": name, **payload} for name, payload in conditions.items()]
    order = {"S0": 0, "S1a": 1, "S1b": 2}
    rows.sort(key=lambda row: order.get(row["condition"], 99))
    return {
        "run_id": FINAL_EXP3_RUN_ID if FINAL_EXP3_METRICS_PATH.exists() else "shared",
        "conditions": rows,
        "paired": [{"comparison": name, **payload} for name, payload in paired.items()],
        "figures": [str(p) for p in _existing_images(EXP3_FIGURES_DIR)],
    }


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
        "metrics_available": FINAL_EXP3_METRICS_PATH.exists() or METRICS_JSON_PATH.exists(),
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

    status["replay_count"] = sum(_count_jsonl_rows(path, limit=5000) for path in REPLAY_PATHS[:-1])
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
