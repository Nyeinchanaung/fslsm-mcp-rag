# FSLSM-RAG-MCP

Thesis implementation for an FSLSM-personalized tutoring system over the Dive into Deep Learning (D2L) corpus.

The project studies whether Felder-Silverman Learning Style Model (FSLSM) profiles can be encoded into synthetic student agents, used to personalize RAG tutor responses, and operationalized through an MCP tool-selection runtime.

## Project

The thesis is structured as three linked experiments:

| Experiment | Purpose | Main Result |
|---|---|---|
| Exp1 | Validate FSLSM-conditioned virtual student agents | Strong models recover assigned profiles reliably; top PRA = 1.000 |
| Exp2 | Compare generic RAG (`R0`) vs FSLSM-personalized RAG (`R1`) | R1 improves style conformance and engagement while preserving relevance |
| Exp3 | Run FSLSM-aware tutoring through a real MCP tool layer | MCP runtime executes successfully with high prompt-token savings |

The overall thesis claim is not that learning styles are proven human traits. The claim is narrower: FSLSM can be used as a controllable personalization signal for synthetic agents, tutor-response style adaptation, and tool-selection/runtime analysis.

## Experiment 1: Virtual Student Agent Fidelity

Exp1 tests whether LLM-based virtual student agents can consistently express an assigned FSLSM profile when answering the 44-item ILS questionnaire.

Setup:

- 16 FSLSM profiles across four binary dimensions: `act_ref`, `sen_int`, `vis_ver`, `seq_glo`
- 5 agent instances per profile, giving 80 FSLSM agents per model
- 3 ILS trials per agent, giving 240 FSLSM records per model
- 5 no-profile baseline agents per model
- 15 API/local models evaluated through LiteLLM/Ollama

Metrics:

- `PRA`: Profile Recovery Accuracy; ties count as mismatches
- `DAS`: Dimension Alignment Score; continuous alignment strength

Overall result:

- 15 models evaluated
- 8/15 models pass the PRA >= 0.82 target
- 6/15 models pass both PRA >= 0.82 and DAS >= 0.75
- Best models: `claude-sonnet-4-20250514` PRA 1.000 / DAS 0.927, `gemma3:12b` PRA 1.000 / DAS 0.882, `gpt-4.1-mini` PRA 0.996 / DAS 0.924
- Smaller models around 1-2B parameters mostly collapse to chance-level PRA around 0.50

Run:

```bash
python experiments/exp1_agent_fidelity/run.py
python experiments/exp1_agent_fidelity/run_baseline.py
python experiments/exp1_agent_fidelity/analyze.py
python experiments/exp1_agent_fidelity/visualize.py
```

Main artifacts:

- `results/exp1/metrics/`
- `results/exp1/raw_responses/`
- `experiments/exp1_agent_fidelity/report.ipynb`
- `experiments/exp1_agent_fidelity/final_defense_report.ipynb`

## Experiment 2: FSLSM Tutor Personalization

Exp2 tests whether FSLSM-conditioned RAG improves tutor responses compared with a generic RAG baseline.

Setup:

- `R0`: generic D2L-grounded RAG
- `R1`: FSLSM-personalized RAG using ProfileAgent directives for retrieval and generation
- 80 synthetic student agents
- 72 D2L questions
- 11,520 sessions total, 5,760 matched R0/R1 pairs
- Tutor model: `gpt-4.1-mini`

Pipeline:

1. `ProfileAgent`: converts FSLSM vector into retrieval and generation directives
2. `RetrievalAgent`: hybrid BM25 + FAISS retrieval with RRF and multi-query decomposition
3. `TutorAgent`: generates D2L-grounded tutor response and engagement score

Overall result:

| Metric | R0 Mean | R1 Mean | Result |
|---|---:|---:|---|
| SCS | 0.261 | 0.469 | R1 +79.6%, large effect |
| Engagement | 3.247 | 3.890 | R1 +19.8%, large effect |
| RR | 3.788 | 3.785 | No significant difference |
| CR@5 | 0.159 | 0.155 | Small negative |
| CR@10 | 0.269 | 0.254 | Small negative |
| ER | 0.340 | 0.333 | Small negative |

Track B pairwise evaluation:

- 5,760 valid pairwise comparisons
- R1 wins: 5,419
- R0 wins: 341
- R1 win rate: 0.9408
- 95% CI: 0.9344-0.9466

Interpretation:

- Personalization mainly improves response style and perceived engagement.
- Response relevance is preserved.
- Retrieval recall does not improve; FSLSM query augmentation slightly shifts retrieval away from factual gold chunks.

Run:

```bash
python experiments/exp2_tutor_personalization/run_exp2.py --mode both
python experiments/exp2_tutor_personalization/evaluate_exp2.py
python experiments/exp2_tutor_personalization/pairwise_eval.py --full
```

Main artifacts:

- `experiments/exp2_tutor_personalization/results/exp2_results_summary.json`
- `experiments/exp2_tutor_personalization/results/exp2_session_metrics.csv`
- `experiments/exp2_tutor_personalization/results/pairwise/`
- `experiments/exp2_tutor_personalization/report.ipynb`

## Experiment 3: MCP Runtime Tool Selection

Exp3 tests whether the tutoring system can be operationalized through a real MCP-style tool runtime.

Setup:

- Real FastMCP-backed tool layer
- 15 registered MCP tools
- `Exp3-Core`: 60-question tool-balanced benchmark, 15 tools x 4 questions
- 16 canonical FSLSM profiles
- Three runtime conditions:
  - `S0`: prompt-bloat baseline with all tool schemas
  - `S1a`: unconditioned RAG-MCP tool retrieval
  - `S1b`: FSLSM-conditioned RAG-MCP tool retrieval

Primary metrics:

- `Task-TSA`: selected tool matches task-optimal tool
- `Profile-TSA`: selected tool matches profile-conditioned target
- `PTS`: prompt-token savings versus S0
- execution success and latency

Overall Exp3-Core result:

| Condition | n | Task-TSA | PTS | Execution Success | Mean Latency |
|---|---:|---:|---:|---:|---:|
| S0 | 960 | 0.783 | 0.0% | 1.000 | 983 ms |
| S1a | 960 | 0.833 | 93.5% | 1.000 | 772 ms |
| S1b | 960 | 0.820 | 93.5% | 1.000 | 802 ms |

Interpretation:

- The strongest Exp3 claim is runtime validity and efficiency.
- `S1a` is strongest on primary Task-TSA.
- `S1b` does not universally improve task tool selection, but supports limited profile-conditioned analysis while preserving high prompt-token savings.

Run:

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/01_build_tool_index.py
./venv/bin/python experiments/exp3_mcp_runtime/scripts/02_verify_setup.py
./venv/bin/python experiments/exp3_mcp_runtime/scripts/11_full_run_core.py --run-id exp3_core_YYYYMMDD_HHMMSS --log-passive
./venv/bin/python experiments/exp3_mcp_runtime/scripts/07_compute_metrics.py --run-id exp3_core_YYYYMMDD_HHMMSS
./venv/bin/python experiments/exp3_mcp_runtime/scripts/08_generate_report.py --run-id exp3_core_YYYYMMDD_HHMMSS
```

Main artifacts:

- `experiments/exp3_mcp_runtime/data/`
- `experiments/exp3_mcp_runtime/results/runs/`
- `experiments/exp3_mcp_runtime/report.ipynb`
- `experiments/exp3_mcp_runtime/final_defense_report.ipynb`

## Demo Web App

The shared Streamlit demo is now outside the experiment folders because it presents all three experiments.

Run:

```bash
./venv/bin/streamlit run demo/app.py
```

If using plain Streamlit outside the repo venv:

```bash
streamlit run demo/app.py
```

Demo sections:

- `Overview`: thesis story across all experiments
- `Exp1`: aggregate Exp1 results and figures
- `Exp1 Live`: mini/full ILS live checks and cached artifact explorer
- `Exp2`: aggregate Exp2 results
- `Exp2 Live`: live R0/R1 comparison, profile plan, retrieval evidence, and pairwise judge
- `Exp3`: aggregate MCP runtime results
- `Live Demo`: live Exp3 MCP tool-selection session
- `Replay Demo`: replay stored Exp3 sessions

Environment notes:

- API demos require relevant keys such as `OPENAI_API_KEY`.
- Exp3 web-search tool requires `TAVILY_API_KEY`.
- Local model demos require Ollama and the selected model to be available locally.
- `gemma3:12b` is intentionally disabled in Exp1 Live on this Mac because it can hang the local runtime.

## Repository Map

```text
demo/                                      Shared Streamlit dashboard
src/                                       Core tutor, agent, retrieval, and evaluation code
data/                                      Profiles, questions, processed chunks, Exp2 data
results/                                  Shared result artifacts
experiments/exp1_agent_fidelity/           Exp1 scripts, notebooks, reports
experiments/exp2_tutor_personalization/    Exp2 runner, evaluator, pairwise judge
experiments/exp3_mcp_runtime/              Exp3 MCP runtime, tools, server, scripts
tests/                                     Regression and runtime tests
```

## Test

```bash
python -m py_compile demo/service.py demo/app.py tests/test_exp3_mcp_runtime.py
pytest tests/test_exp3_mcp_runtime.py -q
```
