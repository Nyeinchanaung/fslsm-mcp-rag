# Thesis Report Context

## Purpose

This handoff pack is a report-writing context bundle for the thesis proposal and related reporting tasks. It summarizes the current project state across Experiment 1, Experiment 2, and the final FastMCP-based Experiment 3 implementation.

This pack intentionally excludes earlier Exp3 prototype branches such as `exp3_mcp_tool_selection` and `exp3_revised`. For reporting purposes, Exp3 should be described using the canonical package:

- `experiments/exp3_mcp_runtime`

## Overall Research Arc

The thesis studies whether learning-style-aware tutoring can be grounded in a machine learning corpus and later operationalized through a real MCP tool-selection runtime.

The experiments connect as follows:

1. **Experiment 1** validates that LLM-based virtual student agents can faithfully represent FSLSM learning profiles.
2. **Experiment 2** tests whether FSLSM-conditioned tutor personalization improves tutoring outcomes in a D2L-grounded RAG setting.
3. **Experiment 3** moves from response personalization to MCP-based instructional tool selection, comparing an MCP baseline against RAG-MCP and FSLSM-conditioned RAG-MCP.

## Current Recommended Thesis Framing

### Experiment 1

- Claim: FSLSM-conditioned virtual student agents are valid enough to support later experiments.
- Status: complete
- Result: strong confirmation

### Experiment 2

- Claim: FSLSM-conditioned tutor generation substantially improves style conformance and engagement without harming relevance, though retrieval-style tension remains.
- Status: complete
- Result: partial confirmation

### Experiment 3

- Claim: a real MCP runtime can support tool-selection evaluation under three conditions:
  - `S0`: MCP-baseline
  - `S1a`: RAG-MCP unconditioned
  - `S1b`: RAG-MCP + FSLSM
- Status: implementation complete, full Exp3-Core run complete
- Primary metrics:
  - `Task-TSA`
  - `PTS`
- Secondary metrics:
  - `Profile-TSA`
  - execution success
  - grounded output rate
  - latency

## Definitions To Keep Stable In The Report

### FSLSM

Felder-Silverman Learning Style Model with four binary dimensions:

- Active / Reflective
- Sensing / Intuitive
- Visual / Verbal
- Sequential / Global

This yields 16 canonical profile combinations.

### D2L Corpus

The course-content grounding source is the chunked D2L machine-learning corpus:

- `d2l/output/d2l_corpus_chunks.json`

### Experiment 3 Conditions

- `S0`: MCP-baseline. All tool schemas are visible to the selector. No profile information is provided during selection.
- `S1a`: unconditioned RAG-MCP. Tool retrieval and reranking use raw task intent only.
- `S1b`: FSLSM-conditioned RAG-MCP. Tool retrieval uses the same task-preserving retrieval plus a small FSLSM rerank signal.

Important: `S0` should not be described as "StandardRAG with no MCP" in the current implementation. The implemented baseline is an MCP-enabled prompt-bloat baseline.

## Proposal Report Emphasis

The report should emphasize:

- Exp1 establishes valid FSLSM virtual students.
- Exp2 demonstrates clear personalization gains at the generation layer.
- Exp3 operationalizes these ideas in a real MCP runtime with tool-selection metrics and run isolation.

The report should not overclaim:

- Exp2 does not improve chunk recall metrics.
- Exp3 full-run results do not show `S1b > S1a` on primary Task-TSA.
- Exp3 does show strong prompt token savings and a limited `S1b` advantage on Profile-TSA Eligible.

## Recommended Files To Upload Into A ChatGPT Project

- `work-plan/report-handoff/00_thesis_report_context.md`
- `work-plan/report-handoff/01_project_overview.md`
- `work-plan/report-handoff/02_exp1_summary.md`
- `work-plan/report-handoff/03_exp2_summary.md`
- `work-plan/report-handoff/04_exp3_summary.md`

Optional supporting artifacts:

- `experiments/exp3_mcp_runtime/results/runs/dry_exp3_core_real_20260503_2/exp3_runtime_table.md`
- proposal PDF
- selected figures from Exp1 and Exp2
