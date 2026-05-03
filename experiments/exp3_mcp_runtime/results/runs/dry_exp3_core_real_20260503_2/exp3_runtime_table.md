# Exp3 MCP Runtime Report

Source DB: `/Users/nyeinchanaung/Documents/GitHub/mcp-rag/experiments/exp3_mcp_runtime/results/runs/dry_exp3_core_real_20260503_2/exp3_runtime_results.db`

## exp3_core

| Condition | n | Task-TSA | Profile-TSA All | Profile-TSA Eligible | Profile Eligible n | PTS | Exec Success | Latency ms | Grounded Output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 12 | 0.917 | 0.583 | 0.000 | 4 | 0.0 | 1.000 | 114.8 | 1.000 |
| S1a | 12 | 0.667 | 0.583 | 0.000 | 4 | 93.5 | 0.917 | 5.0 | 1.000 |
| S1b | 12 | 0.583 | 0.583 | 0.000 | 4 | 93.5 | 0.917 | 4.4 | 1.000 |

| Paired Comparison | Task-TSA Delta | Profile-TSA All Delta | Profile-TSA Eligible Delta | PTS Delta | Latency Delta ms | N Pairs | Profile Eligible Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1b_minus_S1a | -0.083 | 0.000 | 0.000 | -0.0 | -0.7 | 12 | 4 |
| S1b_minus_S0 | -0.333 | 0.000 | 0.000 | 93.5 | -110.4 | 12 | 4 |
| S1a_minus_S0 | -0.250 | 0.000 | 0.000 | 93.5 | -109.7 | 12 | 4 |
