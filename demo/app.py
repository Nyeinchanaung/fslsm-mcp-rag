from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.service import (
    get_exp1_mini_questions,
    get_demo_status,
    format_exp1_questions_for_profile,
    list_exp1_raw_artifacts,
    load_core_answer_key,
    load_core_questions,
    load_exp1_model_options,
    load_exp1_raw_artifact,
    load_exp1_summary,
    load_exp2_summary,
    load_exp2_questions,
    load_exp3_summary,
    load_fslsm_profiles_by_label,
    load_metrics,
    load_profiles,
    load_replays,
    run_exp1_mini_demo,
    run_exp2_pair_demo,
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


def render_runtime_checklist(status: dict) -> None:
    st.divider()
    st.subheader("Runtime Checklist")
    render_check("Profiles", status["profiles_loaded"], f"{status['profile_count']} loaded")
    render_check("Exp3-Core dataset", status["core_dataset_available"])
    render_check("D2L chunks", status["chunks_available"])
    render_check("Tool index", status["tool_index_available"])
    render_check("Exp1 config", status["exp1_config_available"])
    render_check("Exp1 raw artifacts", status["exp1_raw_artifacts"] > 0, f"{status['exp1_raw_artifacts']} files")
    render_check("Exp2 questions", status["exp2_questions_available"], f"{status['exp2_question_count']} loaded")
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


def render_exp1_live(profile_labels: dict[str, dict]) -> None:
    st.subheader("Experiment 1 Live: Virtual Student Fidelity")
    st.caption("Hybrid demo: short live mini-ILS check plus cached full-run artifact inspection.")

    live_tab, artifact_tab = st.tabs(["Quick Live Check", "Artifact Explorer"])

    with live_tab:
        model_options = load_exp1_model_options()
        if not model_options:
            st.warning("Exp1 model configuration is not available.")
            return

        model_labels = {
            f"{row['name']} ({row['source']}{', disabled' if row.get('disabled') else ''})": row
            for row in model_options
        }
        c1, c2, c3, c4 = st.columns([1.3, 1.5, 1.0, 1.0])
        with c1:
            selected_model_label = st.selectbox("Model", sorted(model_labels), key="exp1_live_model")
            selected_model = model_labels[selected_model_label]
        with c2:
            profile_label = st.selectbox("Profile", sorted(profile_labels), key="exp1_live_profile")
        with c3:
            level_label = st.selectbox(
                "Knowledge Level",
                ["general", "beginner", "intermediate", "advanced"],
                key="exp1_live_level",
            )
            knowledge_level = None if level_label == "general" else level_label
        with c4:
            question_count = st.selectbox("Mini-ILS Size", [4, 8, 10, 44], index=0, key="exp1_live_qcount")

        if selected_model["source"] == "Local":
            st.info("Local model selected. The model is loaded only when this run starts and requires the local Ollama backend.")
        if selected_model.get("disabled"):
            st.warning(selected_model.get("disabled_reason") or "This model is disabled for live demo runs.")
        if question_count == 44:
            st.warning("Full 44-question ILS is available for validation-style demos, but it will take longer and may cost more.")

        with st.expander("Mini-ILS Questions"):
            preview_rows = format_exp1_questions_for_profile(profile_label, question_count)
            st.dataframe(
                pd.DataFrame(preview_rows)[
                    ["q_num", "dimension", "question", "option_a", "option_b", "expected_answer", "expected_label"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Run Mini-ILS", key="exp1_live_run", disabled=bool(selected_model.get("disabled"))):
            try:
                result = run_exp1_mini_demo(
                    selected_model["name"],
                    profile_label,
                    knowledge_level,
                    question_count,
                )
            except Exception as exc:
                st.error(f"Exp1 live run failed: {exc}")
                return

            st.success("Mini-ILS run completed.")
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("Model Source", result["source"])
            m2.metric(
                "Mini-PRA",
                f"{result['mini_pra']:.3f}",
                f"{result['dimension_matches']}/{result['dimension_count']} dims",
            )
            m3.metric(
                "Question Accuracy",
                f"{result['question_accuracy']:.3f}",
                f"{result['question_matches']}/{result['question_count']} questions",
            )
            m4.metric("Questions", result["question_count"])
            m5.metric("Latency", f"{result['latency_ms']:.0f} ms")
            m6.metric("Cost", f"${result['cost_usd']:.5f}")
            m7.metric("Tokens", f"{result['token_count']:,}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Assigned vs Recovered")
                rows = []
                for dim, assigned in result["assigned"].items():
                    rows.append({
                        "dimension": dim,
                        "assigned": assigned,
                        "detected": result["detected"][dim],
                        "mini_score": result["raw_scores"][dim],
                        "match": result["detected"][dim] == assigned,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with col2:
                st.subheader("Question Answers")
                st.dataframe(
                    pd.DataFrame(result["rows"])[
                        [
                            "q_num",
                            "dimension",
                            "question",
                            "option_a",
                            "option_b",
                            "expected_answer",
                            "expected_label",
                            "detected_answer",
                            "detected_label",
                            "match",
                            "raw_text",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption(result["note"])

    with artifact_tab:
        artifacts = list_exp1_raw_artifacts()
        if not artifacts:
            st.warning("No cached Exp1 raw artifacts found under results/exp1/raw_responses.")
            return
        labels = {row["label"]: row for row in artifacts}
        selected = st.selectbox("Cached Trial", sorted(labels), key="exp1_artifact")
        if st.button("Load Cached Trial", key="exp1_artifact_load"):
            try:
                record = load_exp1_raw_artifact(labels[selected]["path"])
            except Exception as exc:
                st.error(f"Could not load Exp1 artifact: {exc}")
                return

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Model", record["model"])
            a2.metric("Trial", record["trial"])
            a3.metric("Knowledge", record["knowledge_level"])
            a4.metric("Cost", f"${record['total_cost_usd']:.5f}")

            score_rows = [
                {
                    "dimension": dim,
                    "raw_score": score,
                    "detected": record["detected"].get(dim),
                }
                for dim, score in record["raw_scores"].items()
            ]
            st.subheader("Recovered Profile Scores")
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

            st.subheader("Raw ILS Responses")
            raw_rows = pd.DataFrame(record["raw"])
            if not raw_rows.empty:
                st.dataframe(
                    raw_rows[["q_num", "answer", "raw_text", "cost_usd"]],
                    use_container_width=True,
                    hide_index=True,
                )


def render_exp2_live(profile_labels: dict[str, dict]) -> None:
    st.subheader("Experiment 2 Live: R0 vs R1 Tutor Personalization")
    st.caption("Live paired run for generic RAG and FSLSM-conditioned RAG using the same question and profile.")

    questions = load_exp2_questions()
    profiles_by_label = load_fslsm_profiles_by_label()
    source = st.sidebar.selectbox("Exp2 Question Source", ["Exp2 Dataset", "Custom"], key="exp2_source")
    question_record = None
    st.metric("Exp2 Questions Loaded", len(questions))
    if source == "Exp2 Dataset" and questions:
        question_labels = {
            f"{q['question_id']} - {q.get('question_type', 'question')}": q
            for q in questions
        }
        selected_question = st.selectbox("Question", sorted(question_labels), key="exp2_question")
        question_record = question_labels[selected_question]
        question = question_record["question"]
        st.subheader("Selected Question")
        st.write(question)
        st.caption(
            f"{question_record['question_id']} - {question_record.get('quality_tier', 'n/a')} - "
            f"{len(question_record.get('gold_chunk_ids', []))} gold chunks"
        )
    else:
        question = st.text_area(
            "Question",
            value="Compare minibatch stochastic gradient descent and batch normalization.",
            key="exp2_custom_question",
        )

    profile_label = st.sidebar.selectbox("Exp2 Profile", sorted(profile_labels), key="exp2_profile")
    show_internals = st.sidebar.toggle("Show Exp2 Internals", value=True, key="exp2_internals")
    selected_profile = profiles_by_label.get(profile_label, {})
    vector = profile_labels[profile_label]["fslsm_vector"]

    p1, p2 = st.columns([1, 2])
    p1.metric("Learning Style Profile", profile_label)
    p2.write(f"FSLSM vector: `{vector}`")
    with st.expander("Learning Style Profile", expanded=True):
        profile_rows = [
            {"dimension": dim, "pole": value}
            for dim, value in vector.items()
        ]
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)
        descriptor = selected_profile.get("style_descriptor_graf")
        if descriptor:
            st.write(descriptor)

    if st.button("Run R0/R1 Pair", key="exp2_run_pair"):
        try:
            result = run_exp2_pair_demo(
                question,
                vector,
                question_record=question_record,
            )
        except Exception as exc:
            st.error(f"Exp2 live run failed: {exc}")
            return

        st.success("Paired Exp2 run completed.")
        r0 = result["r0"]
        r1 = result["r1"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Profile", result["profile_label"])
        m2.metric("R0 Engagement", r0.get("engagement_score", 0))
        m3.metric("R1 Engagement", r1.get("engagement_score", 0))
        m4.metric("Retrieval Overlap", f"{result['retrieval_overlap']}/{result['retrieval_union']}")
        m5.metric("Gold Chunks", len(result["gold_chunk_ids"]))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("R0 Generic RAG")
            st.write(r0["response"])
            st.caption(f"Latency {r0['latency_ms']} ms | Tokens {r0['token_count']} | Cost ${r0['tutor_cost']:.5f}")
        with col2:
            st.subheader("R1 FSLSM-Personalized RAG")
            st.write(r1["response"])
            st.caption(f"Latency {r1['latency_ms']} ms | Tokens {r1['token_count']} | Cost ${r1['tutor_cost']:.5f}")

        e1, e2 = st.columns(2)
        with e1:
            st.subheader("R0 Retrieved Evidence")
            for idx, chunk in enumerate(r0["retrieved_chunks"], 1):
                with st.expander(f"R0 Evidence {idx}: {chunk.get('chunk_id', 'source')}"):
                    st.write(chunk.get("text", ""))
        with e2:
            st.subheader("R1 Retrieved Evidence")
            for idx, chunk in enumerate(r1["retrieved_chunks"], 1):
                with st.expander(f"R1 Evidence {idx}: {chunk.get('chunk_id', 'source')}"):
                    st.write(chunk.get("text", ""))

        if show_internals:
            plan = result["reasoning_plan"]
            with st.expander("ProfileAgent Reasoning Plan", expanded=True):
                st.write(f"Retrieval directive: {plan['retrieval_directive']}")
                st.write(f"Generation directive: {plan['generation_directive']}")
                st.write(f"Reranking bias: `{plan['reranking_bias']}`")
                st.write(f"Deprioritize: `{plan['deprioritize']}`")
            with st.expander("Query Reformulation"):
                st.write(f"R0: {r0['reformulated_query']}")
                st.write(f"R1: {r1['reformulated_query']}")


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
    st.subheader("FSLSM-RAG-MCP")
    page = st.radio(
        "Dashboard Section",
        ["Overview", "Exp1", "Exp1 Live", "Exp2", "Exp2 Live", "Exp3", "Live Demo", "Replay Demo"],
        index=0,
    )

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
elif page == "Exp1 Demo":
    render_exp1_live(profile_labels)
elif page == "Exp2":
    render_exp2_tab()
elif page == "Exp2 Demo":
    render_exp2_live(profile_labels)
elif page == "Exp3":
    render_exp3_tab()
elif page == "Exp3 Demo":
    render_live_demo(profile_labels)
elif page == "Exp3 Replay Demo":
    render_replay_demo()

with st.sidebar:
    render_runtime_checklist(status)

with st.expander("Raw Exp3 Metrics JSON"):
    st.json(load_metrics())
