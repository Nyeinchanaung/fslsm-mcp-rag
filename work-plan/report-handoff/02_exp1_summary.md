# Experiment 1 Summary

## Title

Virtual Student Agent Fidelity

## Research Role

Experiment 1 validates whether LLM-based virtual student agents can faithfully encode and express FSLSM learning-style profiles. This is the foundation for using such agents in later tutoring and evaluation experiments.

## Research Question And Hypothesis

- **Research Question:** RQ2
- **Hypothesis H2:** FSLSM-conditioned LLM agents achieve `PRA >= 0.82` and `DAS >= 0.75`

## Setup

- 16 FSLSM profiles
- 5 agent instances per profile
- 3 ILS trials per agent
- 80 FSLSM agents per model
- 240 FSLSM trial records per model
- 5 baseline agents without FSLSM encoding
- 15 baseline records per model
- 15 LLMs evaluated in total

FSLSM dimensions:

- Active / Reflective
- Sensing / Intuitive
- Visual / Verbal
- Sequential / Global

## Metrics

### PRA

Profile Recovery Accuracy measures whether the recovered profile pole matches the assigned pole for each dimension.

- binary accuracy over 4 dimensions
- random baseline: `0.50`

### DAS

Dimension Alignment Score measures the strength of alignment between the generated profile expression and the assigned FSLSM dimension.

- continuous alignment metric
- neutral baseline: `0.50`

## Main Results

Top models:

- `claude-sonnet-4`: PRA `1.000`, DAS `0.927`
- `gpt-4.1-mini`: PRA `0.996`, DAS `0.924`
- `gemma3:12b`: PRA `1.000`, DAS `0.882`

Key findings:

- H2 was confirmed and exceeded.
- 8 of 15 models surpassed the PRA target.
- A practical capability floor appeared around the low-single-digit billion parameter range.
- Very small models clustered around chance performance.
- Baseline no-persona runs converged around PRA `0.50`, confirming that FSLSM prompt conditioning caused the alignment rather than natural model bias alone.

## Interpretation

Exp1 supports the use of synthetic FSLSM student agents in later experiments. The main methodological consequence is that later evaluation does not depend on arbitrary synthetic personas; it depends on student agents that were first validated for profile fidelity.

## Cost Snapshot

Reported API cost examples:

- `gpt-4.1-mini`: about `$2.153`
- `claude-sonnet-4`: about `$18.434`
- local open-source models: effectively `$0` API cost

## Status

- complete
- hypothesis confirmed
- suitable as the methodological validity foundation for Exp2 and Exp3

## Proposal Report Guidance

Emphasize:

- Exp1 is a validity experiment, not a tutoring experiment.
- It justifies later use of FSLSM-based synthetic students.
- It shows that FSLSM conditioning is robust for strong frontier and some open-source models.

Do not overemphasize:

- individual low-performing models
- parameter-count-only explanations

The most important report takeaway is that validated FSLSM student agents are available for downstream experimental use.
