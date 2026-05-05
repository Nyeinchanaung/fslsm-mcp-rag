from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.demo.service import (
    get_demo_status,
    load_core_answer_key,
    load_core_questions,
    load_exp1_summary,
    load_exp2_summary,
    load_exp3_summary,
    load_metrics,
    load_profiles,
    load_replays,
    run_demo_session,
)


st.set_page_config(
    page_title="FSLSM-RAG-MCP Final Defense Dashboard",
    layout="wide",
)


def render_check(label: str, ok: bool, detail: str = "") -> None:
    prefix = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    st.write(f"{prefix} {label}{suffix}")


def render_figure_grid(paths: list[str], captions: dict[str, str], columns: int = 2) -> None:
    if not paths:
        st.info("No generated figures found for this experiment.")
        return
    cols = st.columns(columns)
    for idx, path in enumerate(paths):
        p = Path(path)
        with cols[idx % columns]:
            st.image(str(p), caption=captions.get(p.name, p.stem.replace("_", " ").title()))


def render_exp1_tab() -> None:
    summary = load_exp1_summary()
    if not summary:
        st.warning("Exp1 summary artifacts are not available.")
        return

    st.subheader("Experiment 1: Virtual Student Agent Fidelity")
    st.caption("Validation experiment for FSLSM-conditioned virtual student agents.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Models", summary["n_models"])
    c2.metric("Mean PRA", f"{summary['mean_pra']:.3f}")
    c3.metric("Mean DAS", f"{summary['mean_das']:.3f}")
    c4.metric("PRA >= 0.82", f"{summary['pra_pass_n']}/{summary['n_models']}")
    c5.metric("Both H2 Targets", f"{summary['both_pass_n']}/{summary['n_models']}")

    st.markdown(
        "Exp1 supports the methodological use of FSLSM-conditioned virtual agents. "
        "The correct claim is synthetic profile fidelity under the ILS-based protocol, "
        "not proof of human learning-style validity."
    )

    top_df = pd.DataFrame(summary["top_models"])[["model", "pra", "das", "h2_both_pass"]]
    st.dataframe(
        top_df.style.format({"pra": "{:.3f}", "das": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    captions = {
        "exp1_defense_pra_das_by_model.png": "PRA and DAS by model with H2 thresholds",
        "exp1_defense_h2_target_zone.png": "H2 target-zone scatter plot",
        "exp1_defense_dimension_heatmap.png": "PRA by FSLSM dimension and model",
        "exp1_defense_knowledge_level_pra.png": "PRA robustness across knowledge levels",
        "exp1_defense_baseline_bias_heatmap.png": "No-persona baseline style bias",
        "exp1_defense_hardest_items.png": "Hardest ILS questionnaire items",
        "exp1_defense_cost_vs_pra.png": "Cost versus profile fidelity",
    }
    render_figure_grid(summary["figures"], captions)


def render_exp2_tab() -> None:
    summary = load_exp2_summary()
    if not summary:
        st.warning("Exp2 summary artifacts are not available.")
        return

    st.subheader("Experiment 2: FSLSM-Conditioned Tutor Personalization")
    st.caption("D2L-grounded RAG comparison between generic RAG (R0) and FSLSM-conditioned RAG (R1).")

    pairwise = summary.get("pairwise", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matched Pairs", f"{summary['n_pairs']:,}")
    c2.metric("Profiles", summary["n_profiles"])
    c3.metric("Questions", summary["n_questions"])
    c4.metric("R1 Win Rate", f"{pairwise.get('win_rate_r1', 0):.3f}")
    c5.metric("R1 Wins", f"{pairwise.get('n_r1_win', 0):,}")

    st.markdown(
        "Exp2 shows strong gains in style conformance and engagement, while response relevance is preserved. "
        "Retrieval metrics slightly decline, so the thesis interpretation should frame personalization as mainly "
        "a generation and presentation benefit rather than a retrieval-recall improvement."
    )

    metrics_df = pd.DataFrame(summary["metrics_table"])
    st.dataframe(
        metrics_df.style.format(
            {
                "r0_mean": "{:.3f}",
                "r1_mean": "{:.3f}",
                "delta": "{:+.3f}",
                "cohens_d": "{:+.3f}",
                "p_value": "{:.3g}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    captions = {
        "exp2_defense_track_a_metric_means.png": "Track A metric means: R0 vs R1",
        "exp2_defense_metric_deltas.png": "R1-R0 deltas and effect sizes",
        "exp2_defense_score_distributions.png": "Session-level SCS and engagement distributions",
        "exp2_defense_profile_tradeoff.png": "Profile-level retrieval-style tradeoff",
        "exp2_defense_pairwise_preference.png": "Blind pairwise preference results",
        "exp2_defense_dimension_lift.png": "Personalization lift by FSLSM pole",
    }
    render_figure_grid(summary["figures"], captions)


def render_exp3_tab() -> None:
    summary = load_exp3_summary()
    if not summary.get("conditions"):
        st.warning("Final Exp3 metrics are not available.")
        return

    st.subheader("Experiment 3: FastMCP Tool Selection Runtime")
    st.caption("Final full run of Exp3-Core using the real FastMCP runtime.")

    condition_df = pd.DataFrame(summary["conditions"])
    s0 = condition_df[condition_df["condition"] == "S0"].iloc[0]
    s1a = condition_df[condition_df["condition"] == "S1a"].iloc[0]
    s1b = condition_df[condition_df["condition"] == "S1b"].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Run ID", summary["run_id"])
    c2.metric("Rows", f"{int(condition_df['n'].sum()):,}")
    c3.metric("Best Task-TSA", f"{s1a['task_tsa']:.3f}", "S1a")
    c4.metric("S1b Profile-TSA Eligible", f"{s1b['profile_tsa_eligible']:.3f}")
    c5.metric("S1b PTS", f"{s1b['pts']:.1f}%")

    st.markdown(
        "Exp3 confirms that the final system executes through a real MCP layer with high token savings. "
        "The primary Task-TSA result is strongest for S1a, while S1b shows a limited personalization benefit "
        "on Profile-TSA Eligible. Do not claim that FSLSM conditioning universally improves tool selection."
    )

    st.dataframe(
        condition_df[
            [
                "condition",
                "n",
                "task_tsa",
                "profile_tsa_all",
                "profile_tsa_eligible",
                "pts",
                "execution_success_rate",
                "latency_ms",
                "grounded_tool_output_rate",
            ]
        ].style.format(
            {
                "task_tsa": "{:.3f}",
                "profile_tsa_all": "{:.3f}",
                "profile_tsa_eligible": "{:.3f}",
                "pts": "{:.1f}",
                "execution_success_rate": "{:.3f}",
                "latency_ms": "{:.1f}",
                "grounded_tool_output_rate": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    paired_df = pd.DataFrame(summary["paired"])
    with st.expander("Matched Paired Comparisons", expanded=True):
        st.dataframe(
            paired_df.style.format(
                {
                    "task_tsa_delta": "{:+.3f}",
                    "profile_tsa_all_delta": "{:+.3f}",
                    "profile_tsa_eligible_delta": "{:+.3f}",
                    "pts_delta": "{:+.1f}",
                    "latency_ms_delta": "{:+.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    captions = {
        "exp3_defense_accuracy_pts.png": "Task-TSA, Profile-TSA, and prompt token savings",
        "exp3_defense_paired_deltas.png": "Matched accuracy and latency deltas",
        "exp3_defense_runtime_validity.png": "Execution success, latency, and grounded output",
        "exp3_defense_tool_level_tsa.png": "Task-TSA by target tool",
        "exp3_defense_selected_tool_distribution.png": "Selected tool distribution",
    }
    render_figure_grid(summary["figures"], captions)


def render_live_demo(profile_labels: dict[str, dict]) -> None:
    input_mode = st.sidebar.selectbox("Question Source", ["Exp3-Core", "Custom"], index=0)
    core_questions = load_core_questions()
    answer_key = load_core_answer_key()
    question_record = None

    if input_mode == "Exp3-Core" and core_questions:
        question_labels = {
            f"{q['question_id']} - T{answer_key.get(q['question_id'], {}).get('target_tool_id', '?')} - {q['question_family']}": q
            for q in core_questions
        }
        selected_question = st.sidebar.selectbox("Core Question", sorted(question_labels))
        question_record = question_labels[selected_question]
        question = st.text_area("Question", value=question_record["question"], disabled=True)
        target_tool_id = answer_key.get(question_record["question_id"], {}).get("target_tool_id", "?")
        st.caption(
            f"{question_record['question_id']} - target tool {target_tool_id} - "
            f"{question_record['grounding_mode']} - {question_record['manual_review_status']}"
        )
    else:
        question = st.text_area("Question", value="Compare ResNet and VGG architectures.")

    profile_label = st.sidebar.selectbox("Profile", sorted(profile_labels))
    condition = st.sidebar.selectbox("Condition", ["S0", "S1a", "S1b"], index=2)
    show_raw = st.sidebar.toggle("Show Raw JSON", value=False)

    if st.button("Ask me!"):
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
                    label = item.get("chunk_id", item.get("source", "source"))
                    with st.expander(f"Evidence {idx}: {label}"):
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


def render_replay_demo() -> None:
    replays = load_replays()
    labels = [f"{row.get('session_id', idx)} - {row.get('condition', row.get('mode', 'R2'))}" for idx, row in enumerate(replays)]
    if not replays:
        st.warning("No replay sessions available yet.")
        return
    selected = st.selectbox("Replay Session", range(len(replays)), format_func=lambda idx: labels[idx])
    row = replays[selected]
    st.subheader("Replay Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Condition", row.get("condition", row.get("mode", "R2")))
    c2.metric("Selected Tool", str(row.get("selected_tool_id", "n/a")))
    c3.metric("Task-TSA", "Hit" if row.get("task_tsa_hit", row.get("tsa_hit", False)) else "Miss")
    c4.metric("Execution", "Yes" if row.get("execution_success", False) else "No")

    st.subheader("Evaluation")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Task Optimal Tool", str(row.get("task_optimal_tool_id", row.get("optimal_tool_id", "n/a"))))
    e2.metric("Profile Optimal Tool", str(row.get("profile_optimal_tool_id", "n/a")))
    e3.metric("Profile-TSA", "Hit" if row.get("profile_tsa_hit", False) else "Miss")
    e4.metric("PTS Delta", f"{row.get('pts_delta', 0):.1f}%")

    st.write(f"Selected tool name: **{row.get('selected_tool_name', 'n/a')}**")
    st.write(f"Profile eligible: `{row.get('profile_eval_eligible', False)}`")
    st.write(f"Latency: `{row.get('latency_ms', 0):.1f} ms`")

    if row.get("query"):
        st.subheader("Question")
        st.write(row["query"])

    if row.get("final_response"):
        st.subheader("Final Response")
        st.write(row["final_response"])

    evidence = row.get("retrieved_evidence") or []
    if evidence:
        st.subheader("Retrieved Evidence")
        for idx, item in enumerate(evidence, 1):
            label = item.get("chunk_id", item.get("source", "source"))
            with st.expander(f"Evidence {idx}: {label}"):
                st.write(item.get("text", item.get("content", "")))

    st.subheader("Replay JSON")
    st.json(row)


profiles = load_profiles()
profile_labels = {p["profile_label"]: p for p in profiles}
status = get_demo_status()

st.title("FSLSM-RAG-MCP Final Defense Dashboard")
st.caption("Presentation dashboard for Exp1 agent fidelity, Exp2 tutor personalization, and Exp3 FastMCP runtime results.")

with st.sidebar:
    st.subheader("Presentation Mode")
    page = st.radio("View", ["Overview", "Exp1", "Exp2", "Exp3", "Live Demo", "Replay Demo"], index=0)

    st.divider()
    st.subheader("Runtime Checklist")
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

if page == "Overview":
    exp1 = load_exp1_summary()
    exp2 = load_exp2_summary()
    exp3 = load_exp3_summary()

    st.subheader("Cross-Experiment Thesis Story")
    st.markdown(
        "The three experiments form a staged argument: Exp1 validates FSLSM virtual students, "
        "Exp2 tests whether FSLSM conditioning improves tutor responses, and Exp3 operationalizes "
        "tool selection in a real FastMCP runtime."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Exp1")
        if exp1:
            st.metric("Best Result", f"{exp1['both_pass_n']}/{exp1['n_models']} models")
            st.write("Validated FSLSM agent fidelity for strong API models and several larger open models.")
    with c2:
        st.markdown("### Exp2")
        if exp2:
            st.metric("R1 Pairwise Win Rate", f"{exp2.get('pairwise', {}).get('win_rate_r1', 0):.3f}")
            st.write("Personalization improves style conformance and engagement while preserving relevance.")
    with c3:
        st.markdown("### Exp3")
        if exp3.get("conditions"):
            s1b = next(row for row in exp3["conditions"] if row["condition"] == "S1b")
            st.metric("S1b PTS", f"{s1b['pts']:.1f}%")
            st.write("FastMCP runtime achieves high prompt-token savings with complete execution success.")

    st.info(
        "Defense wording: Exp3 does not show universal S1b superiority on Task-TSA. "
        "The strongest Exp3 claim is runtime validity and efficiency, with a limited Profile-TSA Eligible gain."
    )

elif page == "Exp1":
    render_exp1_tab()
elif page == "Exp2":
    render_exp2_tab()
elif page == "Exp3":
    render_exp3_tab()
elif page == "Live Demo":
    render_live_demo(profile_labels)
elif page == "Replay Demo":
    render_replay_demo()

with st.expander("Raw Exp3 Metrics JSON"):
    st.json(load_metrics())
