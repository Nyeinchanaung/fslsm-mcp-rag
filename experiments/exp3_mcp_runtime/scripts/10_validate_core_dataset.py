from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import CORE_ANSWER_KEY_PATH, CORE_QUESTIONS_PATH
from experiments.exp3_mcp_runtime.core.ground_truth import get_core_profile_tool_id, get_core_task_tool_id
from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_label
from experiments.exp3_mcp_runtime.core.profile_ground_truth import is_profile_eval_eligible
from experiments.exp3_mcp_runtime.core.retriever import D2LRetriever
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles
from experiments.exp3_mcp_runtime.server.app import create_mcp_server
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.tools.tool_registry import get_tool_by_id


if __name__ == "__main__":
    questions = json.loads(CORE_QUESTIONS_PATH.read_text())
    answer_key = json.loads(CORE_ANSWER_KEY_PATH.read_text())
    profiles = load_canonical_profiles()

    assert len(questions) == 60, f"Expected 60 questions, got {len(questions)}"
    assert len(answer_key) == 60, f"Expected 60 answer-key labels, got {len(answer_key)}"
    assert {q["question_id"] for q in questions} == set(answer_key), "Question ids and answer-key ids differ"
    tool_counts = Counter(row["target_tool_id"] for row in answer_key.values())
    assert len(tool_counts) == 15, f"Expected 15 tools, got {len(tool_counts)}"
    assert all(count == 4 for count in tool_counts.values()), f"Expected 4 questions per tool, got {tool_counts}"
    profile_labels = {profile_to_label(profile["fslsm_vector"]) for profile in profiles}

    for question in questions:
        assert "target_tool_id" not in question, f"{question['question_id']} leaks target_tool_id"
        assert "profile_target_overrides" not in question, f"{question['question_id']} leaks profile overrides"
        assert "profile_target_tool_ids" not in question, f"{question['question_id']} leaks profile labels"
        mode = question["grounding_mode"]
        if mode == "d2l":
            assert question["essential_chunk_ids"], f"{question['question_id']} missing essential chunk ids"
        elif mode == "style_fixture":
            assert question["source_text"], f"{question['question_id']} missing source_text"
            assert not question["essential_chunk_ids"], f"{question['question_id']} should not carry D2L gold"
        elif mode == "search":
            assert question["target_evidence_criteria"], f"{question['question_id']} missing search criteria"
            assert not question["essential_chunk_ids"], f"{question['question_id']} should not carry D2L gold"
        else:
            raise AssertionError(f"Unknown grounding mode: {mode}")

        answer = answer_key[question["question_id"]]
        assert answer.get("profile_eval_eligible") == is_profile_eval_eligible(question["question_family"])
        profile_targets = answer.get("profile_target_tool_ids", {})
        assert set(profile_targets) == profile_labels, f"{question['question_id']} missing profile target labels"
        if not answer["profile_eval_eligible"]:
            assert all(
                int(tool_id) == int(answer["target_tool_id"])
                for tool_id in profile_targets.values()
            ), f"{question['question_id']} should preserve explicit task intent"
        else:
            assert any(
                int(tool_id) != int(answer["target_tool_id"])
                for tool_id in profile_targets.values()
            ), f"{question['question_id']} has no profile-divergent labels"

        task_tool_id = get_core_task_tool_id(question["question_id"])
        assert 1 <= task_tool_id <= 15, f"{question['question_id']} invalid task label {task_tool_id}"
        for profile in profiles:
            tool_id = get_core_profile_tool_id(question["question_id"], profile["fslsm_vector"])
            assert 1 <= tool_id <= 15, f"{question['question_id']} invalid tool mapping {tool_id}"

    # Smoke test one question per tool through the runtime.
    idx = ToolIndex()
    try:
        idx.load()
    except Exception:
        idx.build()
        idx.save()
    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    retriever = D2LRetriever()
    first_profile = profiles[0]["fslsm_vector"]
    samples = {}
    for question in questions:
        target_tool_id = answer_key[question["question_id"]]["target_tool_id"]
        samples.setdefault(target_tool_id, question)
    for tool_id, question in sorted(samples.items()):
        tool = get_tool_by_id(tool_id)
        retrieval = retriever.retrieve(question["question"], k=5) if question["grounding_mode"] == "d2l" else {
            "evidence": [],
            "combined_text": "",
        }
        arguments = {
            "question": question["question"],
            "fslsm_profile": first_profile,
            "question_type": question["question_family"],
        }
        if tool_id == 14:
            arguments["k"] = 5
        elif tool_id == 15:
            arguments["max_results"] = 3
        else:
            arguments["evidence"] = retrieval["evidence"]
            arguments["source_text"] = question.get("source_text", "") or retrieval["combined_text"]

        result = client.execute_tool(tool.mcp_name, arguments)
        if tool_id == 15:
            assert "tool_output" in result, "Search tool did not return a structured payload"
        else:
            assert result["execution_success"] is True, f"Runtime failed for tool {tool_id}"

    print("Exp3-Core dataset validation passed.")
