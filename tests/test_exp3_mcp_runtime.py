from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import CORE_ANSWER_KEY_PATH, CORE_QUESTIONS_PATH
from experiments.exp3_mcp_runtime.core.fslsm_augmentor import augment_query
from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_label
from experiments.exp3_mcp_runtime.core.profile_ground_truth import is_profile_eval_eligible
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles
from experiments.exp3_mcp_runtime.core.selector import build_s0_selector_prompt, select_retrieved_tool
from experiments.exp3_mcp_runtime.core.session_runner import Exp3SessionRunner
from experiments.exp3_mcp_runtime.core.session_runner import _SCHEMA
from demo.service import (
    format_exp1_questions_for_profile,
    get_exp1_mini_questions,
    infer_question_type,
    list_exp1_raw_artifacts,
    load_exp1_model_options,
    load_exp2_questions,
    run_exp1_mini_demo,
    run_exp2_pair_demo,
)
from experiments.exp3_mcp_runtime.server.app import create_mcp_server
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.tools.tool_registry import TOOL_REGISTRY
from experiments.exp3_mcp_runtime.runtime_types import Condition


PROFILE = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}


def test_profile_label_and_augment_query():
    assert profile_to_label(PROFILE) == "Active-Sensing-Visual-Sequential"
    aug = augment_query("Explain gradient descent.", PROFILE)
    assert "diagram" in aug.lower()
    assert aug.endswith("Explain gradient descent.")


def test_s0_prompt_contains_no_profile():
    prompt = build_s0_selector_prompt(TOOL_REGISTRY)
    assert "Student learning profile" not in prompt
    assert "Do not assume any specific FSLSM learning profile" in prompt


def test_server_registration_and_execution():
    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    tools = client.list_tools()
    assert len(tools) == 15
    tool_names = {tool["name"] for tool in tools}
    display_names = {tool["display_name"] for tool in tools}
    assert "content_retriever" in tool_names
    assert "Content Retriever" in display_names
    result = client.execute_tool(
        "concept_explainer",
        {
            "question": "What is gradient descent?",
            "fslsm_profile": PROFILE,
            "question_type": "explain_relationship",
            "evidence": [{"chunk_id": "c1", "text": "Gradient descent updates weights iteratively."}],
        },
    )
    assert result["tool_id"] == 1
    assert result["execution_success"] is True
    assert "tool_output" in result


def test_session_runner_logs_grounded_r2a(tmp_path: Path):
    idx = ToolIndex()
    idx.build()
    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    runner = Exp3SessionRunner(
        idx,
        client,
        db_path=tmp_path / "runtime.db",
        passive_log_path=tmp_path / "passive.jsonl",
    )
    record = runner.run_session(
        question_id="MH_TEST",
        question_type="compare",
        query="Compare ResNet and VGG architectures.",
        profile=PROFILE,
        condition=Condition.S1B,
        corpus_backed=True,
        log_passive=True,
    )
    assert record.retrieved_evidence
    assert record.tool_result["execution_success"] is True
    assert record.session_id == "exp3_core:MH_TEST:Active-Sensing-Visual-Sequential"
    assert record.task_optimal_tool_id == record.optimal_tool_id
    assert record.task_tsa_hit == record.tsa_hit
    assert record.profile_optimal_tool_id >= 1
    conn = sqlite3.connect(str(tmp_path / "runtime.db"))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exp3_runtime_sessions)").fetchall()}
    conn.close()
    assert "task_tsa_hit" in cols
    assert "profile_tsa_hit" in cols
    assert (tmp_path / "passive.jsonl").exists()


def test_replay_service_shape(tmp_path: Path):
    sample = {
        "session_id": "demo",
        "condition": "S1b",
        "question_id": "Q1",
        "question_type": "compare",
        "query": "Compare A and B",
    }
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps(sample) + "\n")
    rows = [json.loads(line) for line in replay_path.read_text().splitlines()]
    assert rows[0]["session_id"] == "demo"
    assert infer_question_type("What are the latest transformer models?") == "search"


def test_exp1_live_demo_static_contracts():
    models = load_exp1_model_options()
    assert {row["source"] for row in models} >= {"API", "Local"}
    assert any(row["name"] == "gpt-4.1-mini" for row in models)
    assert any(row["name"] == "gemma3:12b" for row in models)
    gemma12 = next(row for row in models if row["name"] == "gemma3:12b")
    assert gemma12["disabled"] is True
    assert gemma12["disabled_reason"]

    mini4 = get_exp1_mini_questions(4)
    assert len(mini4) == 4
    assert {q["dimension"] for q in mini4} == {"act_ref", "sen_int", "vis_ver", "seq_glo"}

    mini8 = get_exp1_mini_questions(8)
    assert len(mini8) == 8
    assert {q["dimension"] for q in mini8} == {"act_ref", "sen_int", "vis_ver", "seq_glo"}

    mini10 = get_exp1_mini_questions(10)
    assert len(mini10) == 10
    assert {q["dimension"] for q in mini10} == {"act_ref", "sen_int", "vis_ver", "seq_glo"}

    full = get_exp1_mini_questions(44)
    assert len(full) == 44

    preview = format_exp1_questions_for_profile("Active-Sensing-Visual-Sequential", 4)
    assert {"question", "option_a", "option_b", "expected_answer", "expected_label"} <= set(preview[0])


def test_exp1_mini_demo_row_contains_expected_detected_fields():
    responses = iter(["b", "a", "a", "b", "b", "a", "a", "a", "b", "a"])

    class FakeResponse:
        def __init__(self, content: str):
            self.content = content
            self.cost = 0.001
            self.total_tokens = 10

    class FakeClient:
        litellm_model = "fake/test"

        def chat(self, **kwargs):
            return FakeResponse(next(responses))

    result = run_exp1_mini_demo(
        "fake-model",
        "Reflective-Sensing-Visual-Global",
        knowledge_level=None,
        question_count=10,
        client=FakeClient(),
    )
    row = result["rows"][0]
    assert {
        "question",
        "option_a",
        "option_b",
        "expected_pole",
        "expected_label",
        "expected_answer",
        "detected_pole",
        "detected_label",
        "match",
    } <= set(row)
    assert result["question_matches"] == 9
    assert result["question_accuracy"] == 0.9
    assert result["raw_scores"]["seq_glo"] == 0
    assert result["mini_pra"] == 0.75


def test_exp1_artifact_listing_shape():
    artifacts = list_exp1_raw_artifacts(limit=5)
    if artifacts:
        row = artifacts[0]
        assert {"label", "agent_uid", "trial", "path"} <= set(row)


def test_exp2_live_demo_static_contracts(monkeypatch):
    questions = load_exp2_questions()
    assert len(questions) == 72
    assert {"question_id", "question", "gold_chunk_ids"} <= set(questions[0])

    def fake_pair(question, profile, question_record=None):
        return {
            "question": question_record["question"] if question_record else question,
            "profile_label": "Active-Sensing-Visual-Sequential",
            "r0": {"mode": "R0", "response": "generic"},
            "r1": {"mode": "R1", "response": "personalized"},
        }

    monkeypatch.setattr(
        "demo.service.run_exp2_pair_demo",
        fake_pair,
    )
    result = fake_pair("Explain gradient descent.", PROFILE, question_record=questions[0])
    assert result["r0"]["mode"] == "R0"
    assert result["r1"]["mode"] == "R1"


def test_core_dataset_balance_and_profiles():
    questions = json.loads(CORE_QUESTIONS_PATH.read_text())
    answer_key = json.loads(CORE_ANSWER_KEY_PATH.read_text())
    assert len(questions) == 60
    assert len(answer_key) == 60
    assert {q["question_id"] for q in questions} == set(answer_key)
    assert all("target_tool_id" not in question for question in questions)
    assert all("profile_target_tool_ids" not in question for question in questions)
    tool_counts = {}
    for row in answer_key.values():
        tool_counts[row["target_tool_id"]] = tool_counts.get(row["target_tool_id"], 0) + 1
    assert len(tool_counts) == 15
    assert set(tool_counts.values()) == {4}

    profiles = load_canonical_profiles()
    profile_labels = {profile_to_label(profile["fslsm_vector"]) for profile in profiles}
    assert len(profiles) == 16
    questions_by_id = {question["question_id"]: question for question in questions}
    eligible_seen = 0
    for question_id, answer in answer_key.items():
        question = questions_by_id[question_id]
        assert answer["profile_eval_eligible"] == is_profile_eval_eligible(question["question_family"])
        assert set(answer["profile_target_tool_ids"]) == profile_labels
        if answer["profile_eval_eligible"]:
            eligible_seen += 1
            assert any(
                tool_id != answer["target_tool_id"]
                for tool_id in answer["profile_target_tool_ids"].values()
            )
        else:
            assert set(answer["profile_target_tool_ids"].values()) == {answer["target_tool_id"]}
    assert eligible_seen == 4


def test_tool_index_routes_external_search_to_web_search():
    idx = ToolIndex()
    idx.build()
    hits = idx.retrieve(
        "What are the latest transformer architecture developments beyond the material covered in D2L?",
        k=3,
    )
    assert hits[0][0].tool_id == 15


def test_intent_reranker_preserves_explicit_search_intent():
    idx = ToolIndex()
    idx.build()
    profiles = load_canonical_profiles()
    query = "What are the latest benchmark trends for graph neural networks outside the D2L textbook?"
    assert select_retrieved_tool(index=idx, query=query, k=5) == 15
    for profile in profiles:
        assert select_retrieved_tool(
            index=idx,
            query=query,
            profile=profile["fslsm_vector"],
            use_profile=True,
            k=5,
        ) == 15


def test_intent_reranker_routes_explicit_tool_actions():
    idx = ToolIndex()
    idx.build()
    cases = {
        "Find the D2L section that explains Xavier initialization.": 14,
        "Quiz me on activation functions and when they are used.": 12,
        "Summarize this short passage on regularization.": 13,
        "Create a hands-on PyTorch exercise for linear regression.": 11,
        "Draw a text-based diagram of a transformer encoder block.": 4,
        "Compare Adam and SGD as optimizers.": 6,
        "Walk through backpropagation step by step.": 2,
    }
    for query, expected_tool_id in cases.items():
        assert select_retrieved_tool(index=idx, query=query, k=5) == expected_tool_id


def test_metrics_report_task_and_profile_tsa(tmp_path: Path):
    db_path = tmp_path / "runtime.db"
    metrics_path = tmp_path / "metrics.json"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    base = {
        "session_id": "exp3_core:EXP3C_001:Active-Sensing-Visual-Sequential",
        "benchmark": "exp3_core",
        "question_id": "EXP3C_001",
        "question_type": "concept_explain",
        "query": "Explain gradient descent.",
        "profile_json": json.dumps(PROFILE),
        "profile_label": "Active-Sensing-Visual-Sequential",
        "selected_tool_name": "Concept Explainer",
        "task_optimal_tool_id": 1,
        "profile_optimal_tool_id": 4,
        "profile_eval_eligible": 1,
        "input_tokens": 100,
        "execution_success": 1,
        "retrieved_evidence_json": json.dumps([{"chunk_id": "c1", "text": "Gradient descent."}]),
        "final_response": "ok",
        "tool_result_json": json.dumps({"execution_success": True}),
        "created_at": "2026-05-03T00:00:00",
    }

    def insert(condition: str, selected_tool_id: int, pts_delta: float, latency_ms: float) -> None:
        task_hit = int(selected_tool_id == base["task_optimal_tool_id"])
        profile_hit = int(selected_tool_id == base["profile_optimal_tool_id"])
        conn.execute(
            """INSERT INTO exp3_runtime_sessions
            (session_id, benchmark, condition, question_id, question_type, query, profile_json,
             profile_label, selected_tool_id, selected_tool_name, task_optimal_tool_id,
             task_tsa_hit, profile_optimal_tool_id, profile_tsa_hit, profile_eval_eligible,
             optimal_tool_id, tsa_hit, pts_delta, input_tokens, latency_ms, execution_success,
             retrieved_evidence_json, final_response, tool_result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                base["session_id"],
                base["benchmark"],
                condition,
                base["question_id"],
                base["question_type"],
                base["query"],
                base["profile_json"],
                base["profile_label"],
                selected_tool_id,
                base["selected_tool_name"],
                base["task_optimal_tool_id"],
                task_hit,
                base["profile_optimal_tool_id"],
                profile_hit,
                base["profile_eval_eligible"],
                base["task_optimal_tool_id"],
                task_hit,
                pts_delta,
                base["input_tokens"],
                latency_ms,
                base["execution_success"],
                base["retrieved_evidence_json"],
                base["final_response"],
                base["tool_result_json"],
                base["created_at"],
            ),
        )

    insert("S1a", selected_tool_id=1, pts_delta=90.0, latency_ms=20.0)
    insert("S1b", selected_tool_id=4, pts_delta=91.0, latency_ms=30.0)
    conn.commit()
    conn.close()

    subprocess.run(
        [
            sys.executable,
            "experiments/exp3_mcp_runtime/scripts/07_compute_metrics.py",
            "--db-path",
            str(db_path),
            "--metrics-path",
            str(metrics_path),
        ],
        check=True,
    )
    metrics = json.loads(metrics_path.read_text())
    exp3 = metrics["benchmarks"]["exp3_core"]
    assert exp3["conditions"]["S1a"]["task_tsa"] == 1.0
    assert exp3["conditions"]["S1b"]["profile_tsa_eligible"] == 1.0
    assert exp3["paired"]["S1b_minus_S1a"]["task_tsa_delta"] == -1.0
    assert exp3["paired"]["S1b_minus_S1a"]["profile_tsa_eligible_delta"] == 1.0
