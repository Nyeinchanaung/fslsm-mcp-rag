# Experiment 2 Summary

## Title

FSLSM-Conditioned Tutor Personalization

## Research Role

Experiment 2 tests whether FSLSM-conditioned tutoring improves learner-facing outcomes in a D2L-grounded RAG tutoring system.

## Research Question And Hypothesis

- **Research Question:** RQ1
- **Hypothesis H1:** FSLSM-personalized responses improve style conformance, engagement, and retrieval-related quality without degrading relevance

## Conditions

- **R0:** generic RAG baseline
- **R1:** FSLSM-conditioned RAG

R1 modifies both retrieval and generation:

- retrieval uses FSLSM-informed query augmentation
- generation uses FSLSM-conditioned prompting

## Setup

- D2L-grounded tutoring dataset
- 16 FSLSM profiles
- 4 virtual student agents per profile across knowledge levels
- 90 questions per agent according to the experiment README
- reported matched evaluation scale: `5,760` R0/R1 pairs
- tutor model: `claude-sonnet-4`
- pairwise judge: `GPT-4o`

Evaluation tracks:

- **Track A:** automatic metrics
- **Track B:** pairwise blind preference judgement

## Metrics

- `SCS`: Style Conformance Score
- `RR`: Response Relevance
- `CR@5`: Chunk Recall at 5
- `CR@10`: Chunk Recall at 10
- `ER`: Essential Recall / exact-recall-style grounding metric
- `Engagement`: student-rated engagement

## Main Results

### Track A

- `SCS`: `0.261 -> 0.469`, delta `+0.208`, large effect
- `Engagement`: `3.247 -> 3.890`, delta `+0.643`, large effect
- `RR`: essentially unchanged, no significant degradation
- `CR@5`: slight negative effect
- `ER`: slight negative effect

### Track B

- `R1` preferred in `94.1%` of matched pairs
- strong blind preference support for personalization

## Interpretation

Exp2 shows that personalization works strongly at the **generation layer**:

- responses better match the learner's style
- responses are perceived as more engaging
- relevance is preserved

However, retrieval metrics show a small negative tradeoff. The most plausible interpretation is that adding style-oriented retrieval directives can slightly pull retrieval away from the factually defined gold chunks.

This means:

- personalization clearly helps answer presentation
- personalization does not clearly help factual retrieval ranking under the current gold-standard design

## Cost Snapshot

Reported totals:

- `R0`: about `$12.80`
- `R1`: about `$15.51`
- overhead: about `+21.2%`

This means the personalization gain came at relatively low per-session cost overhead.

## Status

- complete
- H1 partially confirmed

Confirmed:

- style conformance gain
- engagement gain
- no relevance degradation

Not confirmed:

- projected retrieval recall gain

## Proposal Report Guidance

Emphasize:

- Exp2 is the strongest evidence that FSLSM personalization materially changes tutoring quality.
- The personalization effect is large and consistent across learner types.
- The main benefit is stylistic and engagement-oriented, not retrieval-oriented.

Be explicit about the limitation:

- retrieval quality did not improve and may slightly decline under style-augmented retrieval

This limitation is important because it motivates Exp3: instead of only conditioning the response, the system can try to choose better instructional tools through an MCP runtime.
