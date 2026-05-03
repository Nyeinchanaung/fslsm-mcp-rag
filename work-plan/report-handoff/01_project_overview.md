# Project Overview

## Project Goal

This project develops and evaluates an FSLSM-aware tutoring system for machine learning education, grounded in a D2L corpus and extended into an MCP-based tool-selection runtime.

The work is organized into three experiments:

- **Exp1:** validate FSLSM virtual student agents
- **Exp2:** test FSLSM-conditioned tutor personalization in a RAG pipeline
- **Exp3:** test MCP-based instructional tool selection under baseline, unconditioned, and FSLSM-conditioned settings

## Shared Assets

### Learning-style representation

- 16 canonical FSLSM profiles
- 4 binary dimensions:
  - Active / Reflective
  - Sensing / Intuitive
  - Visual / Verbal
  - Sequential / Global

### Grounding corpus

- D2L machine-learning corpus
- Current runtime source:
  - `d2l/output/d2l_corpus_chunks.json`

### Synthetic student agents

- Validated through Exp1
- Reused in Exp2 and conceptually continued into Exp3 profile-conditioned evaluation

## Cross-Experiment Logic

### Experiment 1 role

Exp1 is a validity gate. It tests whether model-driven virtual students can reliably express assigned FSLSM learning profiles.

### Experiment 2 role

Exp2 is a tutoring-quality study. It tests whether tutor responses improve when the RAG pipeline is conditioned on the learner's FSLSM profile.

### Experiment 3 role

Exp3 is a runtime/selection study. It shifts the question from "does personalization improve the answer?" to "can an MCP runtime choose the right instructional tool efficiently and with defensible personalization?"

## Current Canonical Experiment Paths

- Exp1: `experiments/exp1_agent_fidelity`
- Exp2: `experiments/exp2_tutor_personalization`
- Exp3: `experiments/exp3_mcp_runtime`

For thesis reporting, use `exp3_mcp_runtime` as the only source of truth for Exp3.

## Current Result Snapshot

### Exp1

- complete
- strong success
- high PRA and DAS for top models

### Exp2

- complete
- partial confirmation
- strong gains in style conformance and engagement
- no relevance degradation
- slight retrieval-recall tradeoff

### Exp3

- full MCP runtime implemented
- FastMCP server + runtime client + session runner + metrics pipeline in place
- Exp3-Core benchmark created
- dry run completed
- full benchmark ready to run on a machine with working API/network access

## Main Reporting Recommendation

Position the thesis as a staged research program:

1. validate FSLSM student simulation
2. validate tutoring gains from FSLSM personalization
3. operationalize the system inside a real MCP tool-selection runtime

This gives a coherent narrative from representation fidelity to tutoring effect to deployment/runtime evaluation.
