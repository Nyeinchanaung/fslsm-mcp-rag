from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.demo.service import (
    get_demo_status,
    load_core_answer_key,
    load_core_questions,
    load_metrics,
    load_profiles,
    load_replays,
    run_demo_session,
)

st.set_page_config(page_title="Exp3 MCP Runtime Demo", layout="wide")

st.title("Exp3 MCP Runtime Demo")
st.caption("Presentation surface for the rebuilt FastMCP-based Experiment 3 runtime.")

profiles = load_profiles()
profile_labels = {p["profile_label"]: p for p in profiles}
status = get_demo_status()


def render_check(label: str, ok: bool, detail: str = "") -> None:
    prefix = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    st.write(f"{prefix} {label}{suffix}")


with st.sidebar:
    st.subheader("Startup Checklist")
    render_check("Profiles", status["profiles_loaded"], f"{status['profile_count']} loaded")
    render_check("Exp3-Core dataset", status["core_dataset_available"])
    render_check("D2L chunks", status["chunks_available"])
    render_check("Tool index", status["tool_index_available"])
    render_check("FastMCP runtime", status["runtime_ready"], f"{status['registered_tool_count']} tools")
    render_check("FastMCP backend", status["fastmcp_active"])
    render_check("OPENAI_API_KEY", status["openai_key_loaded"])
    render_check("TAVILY_API_KEY", status["tavily_key_loaded"])
    render_check("Metrics artifact", status["metrics_available"])
    render_check(
        "Replay sessions",
        (status["replay_count"] + status["legacy_replay_count"]) > 0,
        f"{status['replay_count']} current / {status['legacy_replay_count']} legacy",
    )
    if status["status_errors"]:
        st.warning("Startup issues detected")
        for err in status["status_errors"]:
            st.code(err)

mode = st.sidebar.selectbox("Mode", ["Live Demo", "Replay Demo", "Results View"])

if mode == "Live Demo":
    input_mode = st.sidebar.selectbox("Question Source", ["Exp3-Core", "Custom"], index=0)
    core_questions = load_core_questions()
    answer_key = load_core_answer_key()
    question_record = None
    if input_mode == "Exp3-Core" and core_questions:
        question_labels = {
            f"{q['question_id']} · T{answer_key.get(q['question_id'], {}).get('target_tool_id', '?')} · {q['question_family']}": q
            for q in core_questions
        }
        selected_question = st.sidebar.selectbox("Core Question", sorted(question_labels))
        question_record = question_labels[selected_question]
        question = st.text_area("Question", value=question_record["question"], disabled=True)
        target_tool_id = answer_key.get(question_record["question_id"], {}).get("target_tool_id", "?")
        st.caption(
            f"{question_record['question_id']} · target tool {target_tool_id} · "
            f"{question_record['grounding_mode']} · {question_record['manual_review_status']}"
        )
    else:
        question = st.text_area("Question", value="Compare ResNet and VGG architectures.")
    profile_label = st.sidebar.selectbox("Profile", sorted(profile_labels))
    condition = st.sidebar.selectbox("Condition", ["S0", "S1a", "S1b"], index=2)
    show_raw = st.sidebar.toggle("Show Raw JSON", value=False)
    if st.button("Run Live Demo"):
        result = run_demo_session(
            question,
            profile_labels[profile_label]["fslsm_vector"],
            condition,
            question_record=question_record,
        )
        st.success("Live session completed.")
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Condition", result["condition"])
        stat2.metric("Selected Tool", f"{result['selected_tool_id']}")
        stat3.metric("Latency (ms)", f"{result['latency_ms']:.1f}")
        stat4.metric("Task-TSA Hit", "Yes" if result["task_tsa_hit"] else "No")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Final Response")
            st.write(result["final_response"])
            st.subheader("Retrieved Evidence")
            if result["retrieved_evidence"]:
                for idx, item in enumerate(result["retrieved_evidence"], 1):
                    with st.expander(f"Evidence {idx}: {item.get('chunk_id', item.get('source', 'source'))}"):
                        st.write(item.get("text", item.get("content", "")))
            else:
                st.info("No retrieved evidence for this run.")
        with col2:
            st.subheader("Selection")
            st.write(f"Tool: **{result['selected_tool_name']}**")
            st.write(f"Task optimal tool id: `{result['task_optimal_tool_id']}`")
            st.write(f"Profile optimal tool id: `{result['profile_optimal_tool_id']}`")
            st.write(f"Profile-TSA hit: `{result['profile_tsa_hit']}`")
            st.write(f"Profile eligible: `{result['profile_eval_eligible']}`")
            if question_record:
                target_tool_id = answer_key.get(question_record["question_id"], {}).get("target_tool_id", "?")
                st.write(f"Dataset target tool id: `{target_tool_id}`")
            st.write(f"PTS delta: `{result['pts_delta']}`")
            st.write(f"Execution success: `{result['execution_success']}`")
            st.subheader("Tool Result")
            st.write(result["tool_result"]["tool_output"])
            if show_raw:
                st.json(result["tool_result"])
                st.subheader("Session JSON")
                st.json(result)

elif mode == "Replay Demo":
    replays = load_replays()
    labels = [f"{row.get('session_id', idx)} · {row.get('condition', row.get('mode', 'R2'))}" for idx, row in enumerate(replays)]
    if not replays:
        st.warning("No replay sessions available yet.")
    else:
        selected = st.selectbox("Replay Session", range(len(replays)), format_func=lambda idx: labels[idx])
        row = replays[selected]
        st.subheader("Replay Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Condition", row.get("condition", row.get("mode", "R2")))
        c2.metric("Selected Tool", str(row.get("selected_tool_id", "n/a")))
        c3.metric("Execution", "Yes" if row.get("execution_success", False) else "No")
        if row.get("final_response"):
            st.subheader("Final Response")
            st.write(row["final_response"])
        st.subheader("Replay JSON")
        st.json(row)

else:
    metrics = load_metrics()
    if not metrics:
        st.info("No runtime metrics found yet.")
    else:
        st.subheader("Metrics Snapshot")
        for benchmark, payload in metrics.get("benchmarks", {}).items():
            st.markdown(f"### {benchmark}")
            core = payload.get("conditions", {}).get("S1b", {})
            top = st.columns(4)
            top[0].metric("Task-TSA", f"{core.get('task_tsa', 0):.3f}" if isinstance(core.get("task_tsa"), (int, float)) else "n/a")
            top[1].metric("PTS", f"{core.get('pts', 0):.1f}" if isinstance(core.get("pts"), (int, float)) else "n/a")
            top[2].metric(
                "Profile-TSA Eligible",
                f"{core.get('profile_tsa_eligible', 0):.3f}" if isinstance(core.get("profile_tsa_eligible"), (int, float)) else "n/a",
            )
            top[3].metric(
                "Execution Success",
                f"{core.get('execution_success_rate', 0):.3f}" if isinstance(core.get("execution_success_rate"), (int, float)) else "n/a",
            )
        st.json(metrics)
