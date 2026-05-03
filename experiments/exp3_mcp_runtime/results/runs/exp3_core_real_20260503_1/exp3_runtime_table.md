# Exp3 MCP Runtime Report

Source DB: `/Users/nyeinchanaung/Documents/GitHub/mcp-rag/experiments/exp3_mcp_runtime/results/runs/exp3_core_real_20260503_1/exp3_runtime_results.db`

## exp3_core

| Condition | n | Task-TSA | Profile-TSA All | Profile-TSA Eligible | Profile Eligible n | PTS | Exec Success | Latency ms | Grounded Output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 960 | 0.783 | 0.725 | 0.125 | 64 | 0.0 | 1.000 | 983.1 | 0.533 |
| S1a | 960 | 0.833 | 0.819 | 0.031 | 64 | 93.5 | 1.000 | 771.6 | 0.533 |
| S1b | 960 | 0.820 | 0.818 | 0.125 | 64 | 93.5 | 1.000 | 801.7 | 0.533 |

| Paired Comparison | Task-TSA Delta | Profile-TSA All Delta | Profile-TSA Eligible Delta | PTS Delta | Latency Delta ms | N Pairs | Profile Eligible Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1b_minus_S1a | -0.014 | -0.001 | 0.094 | -0.0 | 30.1 | 960 | 64 |
| S1b_minus_S0 | 0.036 | 0.093 | 0.000 | 93.5 | -181.4 | 960 | 64 |
| S1a_minus_S0 | 0.050 | 0.094 | -0.094 | 93.5 | -211.5 | 960 | 64 |
