# Experiment 3 Summary

## Title

FastMCP-Based Instructional Tool Selection Runtime

## Research Role

Experiment 3 evaluates whether a real MCP runtime can support efficient and accurate instructional tool selection, with and without FSLSM conditioning.

This summary intentionally covers only the final canonical implementation:

- `experiments/exp3_mcp_runtime`

It excludes earlier prototype branches.

## Canonical Runtime Definition

Exp3 is now implemented as a real FastMCP setup with:

- repo-local FastMCP server
- runtime client
- tool index and selector
- session runner
- benchmark datasets
- metrics and report generation
- Streamlit presentation demo

The runtime exposes 15 instructional tools:

- tools `1..13`: generation or transformation tools
- tool `14`: D2L content retriever
- tool `15`: Tavily-backed web search

## Conditions

- `S0`: MCP-baseline
- `S1a`: RAG-MCP unconditioned
- `S1b`: RAG-MCP + FSLSM

Operational meaning:

- `S0`: selector sees all tool schemas at once with no profile signal during selection
- `S1a`: selector retrieves and reranks candidate tools from raw task intent
- `S1b`: selector uses the same task-preserving retrieval plus a small FSLSM reranking signal

Important reporting point:

`S0` should be described as an **MCP-baseline** or **prompt-bloat MCP baseline**, not as a no-MCP StandardRAG baseline.

## Benchmarks

### Exp3-Core

Primary benchmark:

- 60 questions
- 15 tools × 4 questions each
- balanced by tool coverage
- 16 canonical FSLSM profiles
- total full-run size: `60 × 16 × 3 = 2,880` sessions

Grounding modes:

- `d2l`
- `style_fixture`
- `search`

### R2a-Transfer

Secondary benchmark:

- old Exp2-style question set rerun under the full MCP runtime
- should be reported separately from Exp3-Core

## Ground Truth Design

Exp3 uses separated public and private benchmark artifacts:

- `exp3_core_questions.json`: public benchmark questions and grounding metadata
- `exp3_core_answer_key.json`: hidden evaluation labels

The answer key contains:

- `target_tool_id` for primary task-intent evaluation
- `profile_target_tool_ids` for secondary profile-conditioned evaluation
- `profile_eval_eligible` for identifying questions where profile-conditioned tool divergence is meaningful

Current profile-conditioned evaluation policy:

- only `concept_explain` questions are profile-eligible
- explicit task-intent questions keep the same expected tool across all profiles

## Metrics

### Primary

- `Task-TSA`: task-based tool selection accuracy
- `PTS`: prompt token savings

### Secondary

- `Profile-TSA All`
- `Profile-TSA Eligible`
- `execution_success_rate`
- `grounded_tool_output_rate`
- `latency_ms`

## Current Implementation Status

Implemented and verified:

- FastMCP server registration
- MCP client execution
- D2L-backed retrieval tool
- Tavily-backed search tool
- timestamped run isolation
- Task-TSA and Profile-TSA logging
- matched-session metrics
- Streamlit demo
- Exp3-Core dataset validation

Current run structure:

- each run writes under `experiments/exp3_mcp_runtime/results/runs/<run_id>/`

## Final Full-Run Result

Final full run used:

- run id: `exp3_core_real_20260503_1`
- 60 benchmark questions
- 16 canonical FSLSM profiles
- 2,880 total rows
- 960 matched sessions per condition
- 2,880 passive replay log rows

Condition-level metrics:

| Condition | n | Task-TSA | Profile-TSA All | Profile-TSA Eligible | PTS | Exec Success | Latency ms | Grounded Output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `S0` | 960 | 0.783 | 0.725 | 0.125 | 0.0 | 1.000 | 983.1 | 0.533 |
| `S1a` | 960 | 0.833 | 0.819 | 0.031 | 93.5 | 1.000 | 771.6 | 0.533 |
| `S1b` | 960 | 0.820 | 0.818 | 0.125 | 93.5 | 1.000 | 801.7 | 0.533 |

Paired comparisons:

| Comparison | Task-TSA Delta | Profile-TSA All Delta | Profile-TSA Eligible Delta | PTS Delta | Latency Delta ms |
|---|---:|---:|---:|---:|---:|
| `S1b - S1a` | -0.014 | -0.001 | +0.094 | -0.0 | +30.1 |
| `S1b - S0` | +0.036 | +0.093 | 0.000 | +93.5 | -181.4 |
| `S1a - S0` | +0.050 | +0.094 | -0.094 | +93.5 | -211.5 |

Interpretation:

- the runtime and metric pipeline completed successfully at the full benchmark scale
- `S1a` achieved the highest primary Task-TSA
- `S1b` preserved high token savings and improved over `S0`, but did not outperform `S1a` on Task-TSA
- `S1b` improved over `S1a` on Profile-TSA Eligible by +0.094, which supports a limited personalization-specific effect on profile-eligible questions
- both RAG-MCP conditions preserved approximately 93.5% prompt token savings relative to `S0`
- all conditions reached 100% execution success

## Current Limitation

The full run completed successfully, but the main thesis claim should be phrased carefully:

- `S1a` outperformed `S1b` on primary Task-TSA.
- `S1b` showed a positive personalization effect only on the secondary Profile-TSA Eligible metric.
- The strongest Exp3 result is therefore the efficiency and runtime-validity claim, plus a limited profile-conditioned benefit on eligible questions, not a broad claim that FSLSM conditioning improves all tool selection accuracy.

## Full-Run Status

Exp3-Core full run is complete:

- result DB: `experiments/exp3_mcp_runtime/results/runs/exp3_core_real_20260503_1/exp3_runtime_results.db`
- metrics JSON: `experiments/exp3_mcp_runtime/results/runs/exp3_core_real_20260503_1/exp3_runtime_metrics.json`
- report table: `experiments/exp3_mcp_runtime/results/runs/exp3_core_real_20260503_1/exp3_runtime_table.md`
- passive replay log: `experiments/exp3_mcp_runtime/results/runs/exp3_core_real_20260503_1/exp2_r2_passive_log.jsonl`

## Proposal Report Guidance

Emphasize:

- Exp3 is no longer a simulation-only tool-selection prototype.
- It is implemented as a real FastMCP-based execution layer.
- The evaluation separates primary task-oriented tool accuracy from secondary personalization-oriented tool accuracy.
- The benchmark is balanced by tool coverage rather than inherited Exp2 profile distribution.

Be careful not to overclaim:

- `S1b` does not beat `S1a` on primary Task-TSA
- the positive personalization result appears in Profile-TSA Eligible, a secondary metric
- the report should frame Exp3 as demonstrating efficient MCP-based tool selection with a limited FSLSM benefit, rather than universal superiority of FSLSM-conditioned selection
