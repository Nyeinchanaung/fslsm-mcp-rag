# Project Evolution Summary

## Purpose

This file is a compact reconstruction of how `mcp-rag` evolved across the thesis work plans and what now exists in the codebase. It is intended as a durable stand-in for session memory.

## Project Identity

- Repository: `fslsm-mcp-rag`
- Core theme: FSLSM-based personalization for retrieval-augmented tutoring, followed by FSLSM-conditioned MCP tool selection
- Primary corpus: D2L / Dive into Deep Learning material

## Evolution Timeline

### Phase 0: Thesis architecture and repository foundation

Originating plan:
- `work-plan/1.THESIS_DEV_PLAN.md`

What was defined:
- canonical project structure
- PostgreSQL + pgvector + FAISS architecture
- FSLSM profile storage strategy
- scalable ingestion pipeline
- initial design stubs for Experiments 2 and 3

What exists now:
- database models and migrations in `db/`
- vector store support in `db/vector_store.py`
- ingestion and setup scripts in `scripts/`
- configuration modules in `config/`

### Phase 1: Experiment 1 build-out

Primary plans:
- `work-plan/1.1.exp1.md`
- `work-plan/1_1_exp1_phase2_v2_litellm.md`
- `work-plan/1_1_exp1_task2.8_baseline.md`
- `work-plan/exp1.1.walkthrough.md`
- `work-plan/exp1_phase2_small_LLM.md`

Goal:
- test whether virtual student agents preserve FSLSM learning-style fidelity

Major implementation moves:
- project setup, env config, and dependency layer
- D2L ingestion and QA import
- FSLSM profile seeding
- virtual student prompt and agent factory
- ILS answering and evaluator pipeline
- baseline non-personalized condition
- LiteLLM migration for multi-provider support
- expansion to small open-source models through Ollama

Code that reflects this phase:
- `src/agents/agent_factory.py`
- `src/agents/ils_evaluator.py`
- `src/agents/prompts/student_system.py`
- `src/agents/prompts/ils_answering.py`
- `src/evaluation/metrics.py`
- `src/evaluation/visualizer.py`
- `experiments/exp1_agent_fidelity/`

Current state:
- Exp1 appears complete, with runners, analysis, findings, figures, and saved raw responses in `results/exp1/`

### Phase 2: Experiment 2 initial architecture

Primary plans:
- `work-plan/2.exp2-overall.md`
- `work-plan/2.1.exp2_phase1_profile_agent.md`
- `work-plan/2.2.exp2_phase2_retrieval_agent.md`
- `work-plan/2.3.exp2_phase3_tutor_agent.md`
- `work-plan/2.4.exp2_phase4_ab_runner.md`
- `work-plan/2.5.exp2_phase5_evaluation.md`

Goal:
- compare generic RAG (`R0`) versus FSLSM-personalized RAG (`R1`) for tutoring

Planned architecture:
1. `ProfileAgent` converts binary FSLSM profile into natural-language directives
2. `RetrievalAgent` performs retrieval, with personalization injected into R1
3. `TutorAgent` generates the tutoring response and collects engagement
4. A/B runner executes large matched experiments
5. Evaluation computes SCS, RR, CR@k, ER, and Engagement

Code that reflects this phase:
- `src/tutor/profile_agent.py`
- `src/tutor/retrieval_agent.py`
- `src/tutor/tutor_agent.py`
- `src/tutor/prompts/judge_prompts.py`
- `experiments/exp2_tutor_personalization/run_exp2.py`
- `experiments/exp2_tutor_personalization/evaluate_exp2.py`
- `experiments/exp2_tutor_personalization/rr_only.py`
- `experiments/exp2_tutor_personalization/pairwise_eval.py`

Data and configs:
- `data/exp2/sampled_questions.json`
- `data/exp2/filtered_questions.json`
- `data/exp2/scs_style_anchors.json`
- `experiments/exp2_tutor_personalization/configs/r0_config.yaml`
- `experiments/exp2_tutor_personalization/configs/r1_config.yaml`
- `experiments/exp2_tutor_personalization/configs/judge_config.yaml`

### Phase 3: Experiment 2 corrections and hardening

Primary plans:
- `work-plan/2.EXP2_CODEBASE_UPDATE_GUIDE-updated.md`
- `work-plan/2.exp2_RR_IMPROVEMENT_FIXES.md`
- `work-plan/2.pairwise_eval_workplan.md`

Why this phase happened:
- initial Exp2 behavior exposed issues around fairness, retrieval scoring, and response relevance evaluation

Main corrections described in the work plans:
- wire reranking correctly into retrieval
- replace unstable additive rerank boosts with safer weighting
- fix prompt balance between factual completeness and stylistic control
- make R0 and R1 system prompts more comparable
- improve RR judge prompt by including student query and evidence chunks
- add factual anchoring to personalized generation
- adjust token budgets by profile when needed
- clean/filter sampled question set
- add pairwise evaluation as a second evaluation lens

What the repo now shows:
- full Exp2 results are present in `experiments/exp2_tutor_personalization/results/`
- pairwise evaluation outputs exist under `results/pairwise/`
- progress and findings indicate the experiment was completed and analyzed

Reported end state from documentation:
- strong gains in style conformance and engagement for `R1`
- no meaningful relevance gain
- small retrieval recall penalties from FSLSM-conditioned query augmentation

### Phase 4: Experiment 3 initial implementation

Primary plan:
- `work-plan/3.EXPERIMENT_3_WORKPLAN.md`

Goal:
- evaluate FSLSM-conditioned MCP tool selection

Initial design:
- tool registry + FAISS tool index
- dry run and full ablation
- TSA and PTS metrics
- database-backed result logging

Code that reflects the initial Exp3 path:
- `experiments/exp3_mcp_tool_selection/`
- `tool_registry.py`
- `tool_index.py`
- `ablation_runner.py`
- `fslsm_query_augmentor.py`
- `session_adapter.py`
- scripts for diagnostics, dry run, full run, metrics, and report generation

Current state:
- this initial Exp3 line appears complete enough to have generated results and figures
- diagnostics were added afterward because the first formulation exposed structural problems

### Phase 5: Experiment 3 diagnostics and root-cause analysis

Primary plans:
- `work-plan/3.DIAGNOSTIC_DATA_COLLECTION.md`
- `work-plan/3.EXP3_ROOT_CAUSE_DIAGNOSIS.md`

Diagnosed issues:
- FAISS tool-description embeddings were too generic
- profile encoding did not align with ground-truth mapping
- FSLSM augmentor received the wrong profile format
- some ceilings in the `S1a` setup were structural, not implementation bugs

Resulting conclusion:
- the initial Exp3 design needed a corrected second pass rather than small patch fixes

### Phase 6: Experiment 3 revised implementation

Primary plan:
- `work-plan/WORKPLAN_EXP3_REVISED.md`

What changed conceptually:
- rebuild tool registry around a stronger Appendix-B-aligned definition
- rebuild FAISS index using domain-bridging tool descriptions
- add explicit profile decoding
- make FSLSM augmentation operate on the correct 4D representation
- replace partial ground-truth logic with a full 4D mapping
- add coverage questions for tool reachability
- add Tavily-backed web search
- separate `S0`, `S1a`, and `S1b` cleanly
- optionally reuse R2a sessions for an Exp2 extension

Code that reflects the revised path:
- `experiments/exp3_revised/core/profile_decoder.py`
- `experiments/exp3_revised/core/fslsm_augmentor.py`
- `experiments/exp3_revised/core/ground_truth.py`
- `experiments/exp3_revised/core/s0_baseline.py`
- `experiments/exp3_revised/core/session_runner.py`
- `experiments/exp3_revised/tools/tool_registry.py`
- `experiments/exp3_revised/tools/tool_index.py`
- `experiments/exp3_revised/tools/tool_prompts.py`
- `experiments/exp3_revised/tools/tavily_search.py`
- `experiments/exp3_revised/scripts/`

Current state:
- the revised Exp3 implementation exists alongside the original Exp3 directory
- results databases and passive logs are already present under `experiments/exp3_revised/results/`

### Phase 7: Thesis packaging and figure production

Primary plan:
- `work-plan/figure_instructions.md`

Purpose:
- map finalized outputs from Experiments 1, 2, and 3 into thesis-ready figures across Chapters 5 to 7

Interpretation:
- by this point the project had shifted from pipeline construction to reporting, comparison, and presentation

## Current Codebase Shape

### Stable foundation

- `config/`: settings, constants, logging
- `db/`: schema, migrations, seeding, vector store
- `scripts/`: ingestion, profile creation, QA import, reannotation utilities
- `data/`: FSLSM profiles, processed chunks, Exp2 question sets, anchors

### Experiment 1 stack

- `src/agents/`
- `src/evaluation/`
- `experiments/exp1_agent_fidelity/`
- `results/exp1/`

### Experiment 2 stack

- `src/tutor/`
- `experiments/exp2_tutor_personalization/`
- `results/exp2/`

### Experiment 3 stack

- original line: `experiments/exp3_mcp_tool_selection/`
- revised line: `experiments/exp3_revised/`
- `results/exp3/`

## Practical Reading Order for New Work

If you need fast orientation before editing code, read in this order:

1. `experiments/exp2_tutor_personalization/PROGRESS_REPORT.md`
2. `work-plan/PROJECT_EVOLUTION_SUMMARY.md`
3. `work-plan/WORKPLAN_EXP3_REVISED.md`
4. `work-plan/2.EXP2_CODEBASE_UPDATE_GUIDE-updated.md`
5. the code entrypoints for the experiment you are touching

Suggested code entrypoints:
- Exp1: `experiments/exp1_agent_fidelity/run.py`
- Exp2: `experiments/exp2_tutor_personalization/run_exp2.py`
- Exp3 revised: `experiments/exp3_revised/core/session_runner.py`

## High-Confidence Project Status

- Experiment 1: implemented and analyzed
- Experiment 2: implemented, corrected, fully evaluated, and pairwise-evaluated
- Experiment 3 original: implemented and diagnosed
- Experiment 3 revised: implemented as a second-generation pipeline
- thesis figure/reporting assets: present

## Working Assumptions Going Forward

- `src/agents/` primarily belongs to Exp1
- `src/tutor/` primarily belongs to Exp2
- `experiments/exp3_mcp_tool_selection/` is the original Exp3 line
- `experiments/exp3_revised/` is the corrected Exp3 line and likely the safer basis for further Exp3 work
- repo docs are ahead of the minimal top-level `README.md`, so work-plan files and experiment reports are the reliable source of truth
