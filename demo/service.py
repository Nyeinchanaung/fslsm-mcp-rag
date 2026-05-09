from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

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
from src.agents.prompts.ils_answering import build_ils_question_prompt
from src.agents.prompts.student_system import build_student_system_prompt
from src.utils.helpers import extract_ab_choice


FINAL_EXP3_RUN_ID = "exp3_core_real_20260503_1"
FINAL_EXP3_RUN_DIR = RUNS_DIR / FINAL_EXP3_RUN_ID
FINAL_EXP3_METRICS_PATH = FINAL_EXP3_RUN_DIR / METRICS_JSON_FILENAME
FINAL_EXP3_REPLAY_PATH = FINAL_EXP3_RUN_DIR / "exp2_r2_passive_log.jsonl"

EXP1_METRICS_DIR = REPO_ROOT / "results" / "exp1" / "metrics"
EXP1_CONFIG_PATH = REPO_ROOT / "experiments" / "exp1_agent_fidelity" / "config.yaml"
EXP1_RAW_DIR = REPO_ROOT / "results" / "exp1" / "raw_responses"
EXP1_FIGURES_DIR = REPO_ROOT / "experiments" / "exp1_agent_fidelity" / "results" / "exp1" / "final_defense_figures"
EXP2_QUESTIONS_PATH = REPO_ROOT / "data" / "exp2" / "filtered_questions.json"
EXP2_RESULTS_DIR = REPO_ROOT / "experiments" / "exp2_tutor_personalization" / "results"
EXP2_FIGURES_DIR = EXP2_RESULTS_DIR / "final_defense_figures"
EXP2_PAIRWISE_DIR = EXP2_RESULTS_DIR / "pairwise"
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
    "broadcasting",
    "calculus",
    "classification",
    "cnn",
    "convolution",
    "d2l",
    "derivative",
    "deep learning",
    "dropout",
    "eigenvalue",
    "factorization",
    "embedding",
    "gradient",
    "gru",
    "jax",
    "learning rate",
    "linear algebra",
    "linear regression",
    "logistic regression",
    "loss",
    "machine learning",
    "matrix",
    "matrix multiplication",
    "multilayer perceptron",
    "ml",
    "mlp",
    "model",
    "mxnet",
    "neural",
    "normalization",
    "optimizer",
    "overfitting",
    "parameter",
    "probability",
    "pytorch",
    "regularization",
    "resnet",
    "rnn",
    "self-attention",
    "softmax",
    "tensor",
    "tokenization",
    "tensorflow",
    "transformer",
    "vector",
    "vectorization",
    "vgg",
}

API_MODEL_PREFIXES = ("gpt-", "claude-")
EXP1_DISABLED_LIVE_MODELS = {
    "gemma3:12b": "Disabled for live demo on this Mac because it can hang the local runtime.",
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


@lru_cache(maxsize=1)
def load_exp1_config() -> dict[str, Any]:
    if not EXP1_CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(EXP1_CONFIG_PATH.read_text()) or {}


def load_exp1_model_options() -> list[dict[str, Any]]:
    models = load_exp1_config().get("models", [])
    options = []
    for row in models:
        name = row["name"]
        is_api = name.startswith(API_MODEL_PREFIXES)
        options.append({
            "name": name,
            "temperature": row.get("temperature", 0.3),
            "litellm_model": name,
            "source": "API" if is_api else "Local",
            "disabled": name in EXP1_DISABLED_LIVE_MODELS,
            "disabled_reason": EXP1_DISABLED_LIVE_MODELS.get(name, ""),
        })
    return options


@lru_cache(maxsize=1)
def load_ils_questions() -> list[dict[str, Any]]:
    path = REPO_ROOT / "data" / "fslsm" / "ils_questionnaire.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def get_exp1_mini_questions(size: int = 4) -> list[dict[str, Any]]:
    questions = load_ils_questions()
    if size == 44:
        return questions
    per_dim_by_size = {4: 1, 8: 2, 16: 4, 32: 8}
    if size not in per_dim_by_size:
        size = 4
    per_dim = per_dim_by_size[size]
    selected = []
    counts = {"act_ref": 0, "sen_int": 0, "vis_ver": 0, "seq_glo": 0}
    for question in questions:
        dim = question["dimension"]
        if counts[dim] < per_dim:
            selected.append(question)
            counts[dim] += 1
        if len(selected) == size:
            break
    return selected


@lru_cache(maxsize=1)
def load_fslsm_profiles_by_label() -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "data" / "fslsm" / "profiles.json"
    if not path.exists():
        return {}
    profiles = json.loads(path.read_text())
    return {
        profile["label"]: profile
        for profile in profiles
        if profile.get("dimensions", {}).get("act_ref") != 0
    }


def _profile_by_label(profile_label: str) -> dict[str, Any]:
    profile = load_fslsm_profiles_by_label().get(profile_label)
    if profile:
        return profile
    raise ValueError(f"Unknown profile label: {profile_label}")


def _detected_from_scores(scores: dict[str, int]) -> dict[str, int]:
    return {
        dim: 1 if score > 0 else (-1 if score < 0 else 0)
        for dim, score in scores.items()
    }


def _mini_das_scores(
    raw_scores: dict[str, int],
    assigned: dict[str, int],
    question_counts: dict[str, int],
) -> tuple[dict[str, float], float]:
    dim_scores = {}
    for dim, count in question_counts.items():
        if count <= 0:
            continue
        dim_scores[dim] = (raw_scores[dim] * assigned[dim] + count) / (2 * count)
    overall = sum(dim_scores.values()) / len(dim_scores) if dim_scores else 0.0
    return dim_scores, overall


def _pole_label(dim: str, value: int) -> str:
    labels = {
        "act_ref": {-1: "Active", 1: "Reflective", 0: "Tie"},
        "sen_int": {-1: "Sensing", 1: "Intuitive", 0: "Tie"},
        "vis_ver": {-1: "Visual", 1: "Verbal", 0: "Tie"},
        "seq_glo": {-1: "Sequential", 1: "Global", 0: "Tie"},
    }
    return labels.get(dim, {}).get(value, str(value))


def _expected_answer_for_profile(question: dict[str, Any], assigned_pole: int) -> str:
    if question["option_a"]["pole"] == assigned_pole:
        return "a"
    if question["option_b"]["pole"] == assigned_pole:
        return "b"
    return ""


def format_exp1_questions_for_profile(
    profile_label: str,
    question_count: int = 4,
) -> list[dict[str, Any]]:
    profile = _profile_by_label(profile_label)
    assigned = profile["dimensions"]
    rows = []
    for question in get_exp1_mini_questions(question_count):
        dim = question["dimension"]
        expected_pole = assigned[dim]
        expected_answer = _expected_answer_for_profile(question, expected_pole)
        rows.append({
            "q_num": question["q_num"],
            "dimension": dim,
            "question": question["text"],
            "option_a": question["option_a"]["text"],
            "option_b": question["option_b"]["text"],
            "expected_answer": expected_answer,
            "expected_pole": expected_pole,
            "expected_label": _pole_label(dim, expected_pole),
        })
    return rows


def run_exp1_mini_demo(
    model_name: str,
    profile_label: str,
    knowledge_level: str | None,
    question_count: int = 4,
    client: Any | None = None,
) -> dict[str, Any]:
    if client is None:
        from src.utils.llm_client import LLMClient
        client = LLMClient(model_name, temperature=0.3)

    profile = _profile_by_label(profile_label)
    questions = get_exp1_mini_questions(question_count)
    system_prompt = build_student_system_prompt(profile, knowledge_level=knowledge_level)
    raw_scores = {dim: 0 for dim in ("act_ref", "sen_int", "vis_ver", "seq_glo")}
    question_counts = {dim: 0 for dim in ("act_ref", "sen_int", "vis_ver", "seq_glo")}
    rows = []
    started_at = time.perf_counter()
    total_cost = 0.0
    total_tokens = 0

    for question in questions:
        dim = question["dimension"]
        question_counts[dim] += 1
        expected_pole = profile["dimensions"][dim]
        expected_answer = _expected_answer_for_profile(question, expected_pole)
        prompt = build_ils_question_prompt(question)
        response = client.chat(system=system_prompt, user=prompt, max_tokens=10)
        total_cost += response.cost
        total_tokens += response.total_tokens
        answer = extract_ab_choice(response.content)
        pole = None
        if answer in ("a", "b"):
            pole = question[f"option_{answer}"]["pole"]
            raw_scores[dim] += pole
        rows.append({
            "q_num": question["q_num"],
            "dimension": dim,
            "question": question["text"],
            "option_a": question["option_a"]["text"],
            "option_b": question["option_b"]["text"],
            "expected_answer": expected_answer,
            "expected_pole": expected_pole,
            "expected_label": _pole_label(dim, expected_pole),
            "answer": answer or "unparsed",
            "detected_answer": answer or "unparsed",
            "detected_pole": pole,
            "detected_label": _pole_label(dim, pole or 0),
            "match": pole == expected_pole,
            "raw_text": response.content.strip(),
            "selected_pole": pole,
            "selected_label": _pole_label(dim, pole or 0),
        })

    assigned = dict(profile["dimensions"])
    detected = _detected_from_scores(raw_scores)
    represented_dims = sorted({question["dimension"] for question in questions})
    matches = [
        dim for dim in represented_dims
        if detected[dim] != 0 and detected[dim] == assigned[dim]
    ]
    mini_pra = len(matches) / len(represented_dims) if represented_dims else 0.0
    question_matches = sum(1 for row in rows if row["match"])
    question_accuracy = question_matches / len(rows) if rows else 0.0
    dimension_das, mini_das = _mini_das_scores(raw_scores, assigned, question_counts)

    return {
        "model": model_name,
        "litellm_model": client.litellm_model,
        "source": "Local" if client.litellm_model.startswith("ollama/") else "API",
        "profile_label": profile_label,
        "knowledge_level": knowledge_level or "general",
        "question_count": len(questions),
        "assigned": assigned,
        "detected": detected,
        "raw_scores": raw_scores,
        "dimension_das": dimension_das,
        "mini_das": mini_das,
        "mini_pra": mini_pra,
        "dimension_matches": len(matches),
        "dimension_count": len(represented_dims),
        "question_accuracy": question_accuracy,
        "question_matches": question_matches,
        "rows": rows,
        "latency_ms": (time.perf_counter() - started_at) * 1000,
        "token_count": total_tokens,
        "cost_usd": total_cost,
        "note": "Mini-ILS is a dashboard demonstration, not the full 44-item Exp1 protocol.",
    }


def list_exp1_raw_artifacts(limit: int = 5000) -> list[dict[str, Any]]:
    if not EXP1_RAW_DIR.exists():
        return []
    rows = []
    for path in sorted(EXP1_RAW_DIR.glob("*.json"))[:limit]:
        stem = path.stem
        if "_trial" not in stem:
            continue
        agent_uid, trial = stem.rsplit("_trial", 1)
        rows.append({
            "label": f"{agent_uid} | trial {trial}",
            "agent_uid": agent_uid,
            "trial": trial,
            "path": str(path),
        })
    return rows


def load_exp1_raw_artifact(path: str) -> dict[str, Any]:
    raw_path = Path(path)
    if not raw_path.exists() or raw_path.parent != EXP1_RAW_DIR:
        raise FileNotFoundError(f"Exp1 raw artifact not found: {path}")
    payload = json.loads(raw_path.read_text())
    scores = payload.get("dim_scores", {})
    detected = _detected_from_scores(scores)
    return {
        "path": str(raw_path),
        "agent_uid": payload.get("agent_uid", raw_path.stem),
        "model": payload.get("model", ""),
        "trial": payload.get("trial", ""),
        "knowledge_level": payload.get("knowledge_level") or "general",
        "raw_scores": scores,
        "detected": detected,
        "raw": payload.get("raw", []),
        "total_cost_usd": payload.get("total_cost_usd", 0.0),
    }


@lru_cache(maxsize=1)
def load_exp2_questions() -> list[dict[str, Any]]:
    if EXP2_QUESTIONS_PATH.exists():
        return json.loads(EXP2_QUESTIONS_PATH.read_text())
    return []


@lru_cache(maxsize=1)
def get_exp2_demo_tutor() -> TutorAgent:
    from src.tutor.profile_agent import ProfileAgent
    from src.tutor.retrieval_agent import RetrievalAgent
    from src.tutor.tutor_agent import TutorAgent
    from src.utils.llm_client import LLMClient

    profile_agent = ProfileAgent(profiles_path=REPO_ROOT / "data" / "fslsm" / "profiles.json")
    decompose_client = LLMClient("gpt-4.1-mini", temperature=0.0)
    retrieval_agent = RetrievalAgent(decompose_client=decompose_client)
    tutor_client = LLMClient("gpt-4.1-mini", temperature=0.3)
    student_client = LLMClient("gpt-4.1-mini", temperature=0.0)
    return TutorAgent(
        tutor_client=tutor_client,
        student_client=student_client,
        profile_agent=profile_agent,
        retrieval_agent=retrieval_agent,
    )


def _normalize_exp2_result(result: dict[str, Any]) -> dict[str, Any]:
    chunks = result.get("retrieved_chunks", [])
    return {
        "mode": result.get("mode"),
        "response": result.get("response", ""),
        "system_prompt_used": result.get("system_prompt_used", ""),
        "retrieved_chunk_ids": result.get("retrieved_chunk_ids", []),
        "retrieved_chunks": chunks,
        "reformulated_query": result.get("reformulated_query", ""),
        "engagement_score": result.get("engagement_score"),
        "latency_ms": result.get("latency_ms", 0),
        "token_count": result.get("token_count", 0),
        "tutor_cost": result.get("tutor_cost", 0.0),
    }


def build_exp2_out_of_scope_pair(question: str, profile: dict[str, Any], reason: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile_label = profile_to_label(profile)
    message = (
        "This Exp2 demo is scoped to D2L machine-learning tutoring. I skipped RAG "
        "retrieval because the custom question does not appear to be a D2L or "
        "machine-learning course question. Try a question about neural networks, "
        "optimization, deep learning, PyTorch, TensorFlow, or D2L textbook topics."
    )
    latency_ms = (time.perf_counter() - started_at) * 1000
    base = {
        "response": message,
        "system_prompt_used": "Demo Scope Guard",
        "retrieved_chunk_ids": [],
        "retrieved_chunks": [],
        "reformulated_query": "",
        "engagement_score": None,
        "latency_ms": latency_ms,
        "token_count": 0,
        "tutor_cost": 0.0,
    }
    return {
        "question": question,
        "question_id": "custom_out_of_scope",
        "profile_label": profile_label,
        "fslsm_vector": profile,
        "reasoning_plan": {
            "profile_code": "scope_guard",
            "style_label": profile_label,
            "retrieval_directive": "No retrieval: custom question is outside the D2L machine-learning scope.",
            "generation_directive": message,
            "reranking_bias": [],
            "deprioritize": [],
            "metadata": {"reason": reason},
        },
        "r0": {"mode": "R0", **base},
        "r1": {"mode": "R1", **base},
        "retrieval_overlap": 0,
        "retrieval_union": 0,
        "gold_chunk_ids": [],
        "essential_chunk_ids": [],
        "out_of_scope": True,
        "scope_guard_reason": reason,
    }


def run_exp2_pair_demo(
    question: str,
    profile: dict[str, Any],
    question_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if question_record is None and not should_use_corpus_for_custom_question(question):
        return build_exp2_out_of_scope_pair(
            question=question,
            profile=profile,
            reason="custom_question_outside_course_scope",
        )

    tutor = get_exp2_demo_tutor()
    profile_agent = tutor.profile_agent
    plan = profile_agent.generate_reasoning_plan(profile)
    base_session = {
        "agent_id": "demo_exp2",
        "profile_label": profile_to_label(profile),
        "fslsm_vector": profile,
        "question_id": question_record.get("question_id", "custom") if question_record else "custom",
        "question": question_record.get("question", question) if question_record else question,
        "gold_chunk_ids": question_record.get("gold_chunk_ids", []) if question_record else [],
        "essential_chunk_ids": question_record.get("essential_chunk_ids", []) if question_record else [],
        "gold_answer": question_record.get("gold_answer", "") if question_record else "",
    }
    r0 = tutor.run_session({**base_session, "mode": "R0"})
    r1 = tutor.run_session({**base_session, "mode": "R1"})
    r0_ids = set(r0.get("retrieved_chunk_ids", []))
    r1_ids = set(r1.get("retrieved_chunk_ids", []))
    return {
        "question": base_session["question"],
        "question_id": base_session["question_id"],
        "profile_label": base_session["profile_label"],
        "fslsm_vector": profile,
        "reasoning_plan": plan,
        "r0": _normalize_exp2_result(r0),
        "r1": _normalize_exp2_result(r1),
        "retrieval_overlap": len(r0_ids & r1_ids),
        "retrieval_union": len(r0_ids | r1_ids),
        "gold_chunk_ids": base_session["gold_chunk_ids"],
        "essential_chunk_ids": base_session["essential_chunk_ids"],
    }


def judge_exp2_pair_demo(
    pair_result: dict[str, Any],
    judge_client: Any | None = None,
) -> dict[str, Any]:
    from src.evaluation.metrics import judge_pairwise

    if judge_client is None:
        from src.utils.llm_client import LLMClient
        judge_client = LLMClient("gpt-4o", temperature=0.0)

    session = {
        "session_id": f"demo_exp2__{pair_result.get('question_id', 'custom')}",
        "agent_id": "demo_exp2",
        "profile_label": pair_result.get("profile_label", ""),
        "fslsm_vector": pair_result.get("fslsm_vector", {}),
        "question_id": pair_result.get("question_id", ""),
        "question_text": pair_result.get("question", ""),
        "question_type": "",
        "r0_response": pair_result.get("r0", {}).get("response", ""),
        "r1_response": pair_result.get("r1", {}).get("response", ""),
    }
    return judge_pairwise(
        session=session,
        swap=False,
        judge_client=judge_client,
        max_tokens=200,
        response_token_cap=1200,
    )


@lru_cache(maxsize=1)
def load_exp2_pairwise_track() -> dict[str, Any]:
    summary_path = EXP2_PAIRWISE_DIR / "summary_overall.json"
    profile_path = EXP2_PAIRWISE_DIR / "summary_by_profile.csv"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text())
    profiles = []
    if profile_path.exists():
        profiles = pd.read_csv(profile_path).to_dict(orient="records")
    return {
        "summary": summary,
        "profiles": profiles,
        "figures": [str(p) for p in _existing_images(EXP2_PAIRWISE_DIR / "figures")],
    }


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
        "top_das_models": summary.sort_values(["das", "pra"], ascending=False).head(5).to_dict(orient="records"),
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
        "exp1_config_available": EXP1_CONFIG_PATH.exists(),
        "exp1_raw_artifacts": len(list(EXP1_RAW_DIR.glob("*.json"))) if EXP1_RAW_DIR.exists() else 0,
        "exp2_questions_available": EXP2_QUESTIONS_PATH.exists(),
        "exp2_question_count": len(load_exp2_questions()),
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
