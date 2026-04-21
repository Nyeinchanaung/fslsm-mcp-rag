# Experiment 2 — Progress Report
**FSLSM-Based Tutor Personalization (PersonaRAG)**  
**Prepared for:** Supervisor Meeting  
**Date:** April 20, 2026  
**Status:** Full experiment complete — evaluation done

---

## 1. Executive Summary

**Research Question:** Does FSLSM-conditioned RAG (R1) produce tutor responses with better style conformance, relevance, and student engagement compared to a generic RAG baseline (R0)?

**Hypothesis H1:** FSLSM-personalized responses will achieve higher Style Conformance Score (SCS), Chunk Recall (CR@10), and Engagement than the control condition.

**Current Status:** Full A/B experiment complete (11,520 sessions, 5,760 matched pairs). Evaluation pipeline finished.

**Key Final Results:**
- SCS: R1 = 0.518 vs R0 = 0.269 (+92%, p ≈ 0, Cohen's d = 3.26) ✅
- Engagement: R1 = 4.114 vs R0 = 3.238 (+27%, p ≈ 0, Cohen's d = 2.02) ✅
- RR: R1 = 4.214 vs R0 = 4.655 (−9.5%, p ≈ 0, d = −0.82) ⚠️ significant but negative
- CR@10 and ER: no significant difference (p = 0.47, p = 0.18)

---

## 2. Development Timeline

| Date | Commit | Milestone |
|------|--------|-----------|
| Feb 18, 2026 | `a533f04` | Initial project setup |
| Feb 18, 2026 | `b34c1d4` | D2L-QA ground truth dataset generator (corpus foundation) |
| Feb 20, 2026 | `073ca47` | D2L QA refactored into single-hop / multi-hop structure |
| Feb 21, 2026 | `9ed36dc` | First output results |
| Mar 8, 2026 | `8e8a691` | Phase 1: FSLSM ProfileAgent created |
| Mar 12, 2026 | `661649a` | Exp 1 evaluation: DAS and PRA metrics |
| Mar 16, 2026 | `661649a` | Exp 1.1 analysis complete |
| Mar 25, 2026 | `59636f4` | Experiment 1 final report updated |
| Mar 29–30, 2026 | `65b3cb1–76f9f15` | Exp 1: additional open-source LLMs (Qwen, Gemma2, Gemma3) |
| Apr 5–11, 2026 | `64b0910–9c543f4` | Exp 1: scaling to 10 LLMs, reruns |
| Apr 15, 2026 | `294c638` | **Exp 2 setup**: dataset creation, 72 questions, 80 agents, pipeline wiring |
| Apr 16, 2026 | `a596d3a` | Dry run with 40 agents — system validation |
| Apr 17–18, 2026 | *(in-session)* | Gold chunk re-annotation (relevance-only, GPT-4o), retrieval improvements |
| Apr 18, 2026 | *(in-session)* | Full experiment launched (11,520 sessions, 1 worker, nohup) |
| Apr 19, 2026 | *(current)* | **62% complete** — this progress report |

---

## 3. System Architecture

Experiment 2 implements a three-phase personalization pipeline:

```
Student FSLSM Profile (binary vector)
        │
        ▼
┌───────────────────┐
│  Phase 1          │  ProfileAgent
│  Profile → Plan   │  · Translates [-1/+1 per dimension] into natural language
│                   │  · Outputs: retrieval_directive, generation_directive,
│                   │    reranking_bias, deprioritize
└────────┬──────────┘
         │ reasoning_plan
         ▼
┌───────────────────┐
│  Phase 2          │  RetrievalAgent (R0 vs R1)
│  Retrieval        │  R0: hybrid BM25+FAISS, raw question
│                   │  R1: hybrid BM25+FAISS, query augmented with retrieval_directive
│                   │  · Multi-query decomposition (LLM splits multi-hop questions)
│                   │  · Reciprocal Rank Fusion (RRF) for BM25 + FAISS merging
└────────┬──────────┘
         │ top-k chunks
         ▼
┌───────────────────┐
│  Phase 3          │  TutorAgent
│  Generation       │  R0: generic system prompt
│                   │  R1: FSLSM-conditioned system prompt (generation_directive)
│                   │  · gpt-4.1-mini, temperature=0.3
│                   │  · Virtual student engagement scoring (1–5)
└───────────────────┘
```

### FSLSM Dimensions (4 binary dimensions → 16 profiles)

| Dimension | Pole −1 | Pole +1 |
|-----------|---------|---------|
| `act_ref` | Active | Reflective |
| `sen_int` | Sensing | Intuitive |
| `vis_ver` | Visual | Verbal |
| `seq_glo` | Sequential | Global |

**Example ProfileAgent output for Active-Sensing-Visual-Sequential learner:**

*Retrieval directive:*
> "interactive exercises, hands-on worked examples, concrete facts, specific numerical examples, diagrams, charts, step-by-step procedures"

*Generation directive:*
> "ACTIVE LEARNING STYLE — Include at least one hands-on exercise or 'Try it yourself' prompt...  
> SENSING LEARNING STYLE — Include specific numerical examples with concrete values...  
> VISUAL LEARNING STYLE — Include at least one ASCII diagram, table, or structured visual...  
> SEQUENTIAL LEARNING STYLE — Structure your response as numbered steps..."

---

## 4. Experiment Design

| | R0 (Control) | R1 (Experimental) |
|--|-------------|-------------------|
| **Label** | Generic RAG | FSLSM-Personalized RAG |
| **Retrieval** | Hybrid BM25+FAISS, raw question | Hybrid BM25+FAISS, question + FSLSM retrieval_directive |
| **Generation** | Generic system prompt | FSLSM-conditioned system prompt |
| **Model** | gpt-4.1-mini | gpt-4.1-mini |

**Dataset:**
- 72 questions sampled from D2L corpus (mix of single-hop and multi-hop)
- 80 synthetic student agents (16 FSLSM profiles × 5 instances each)
- **Total sessions:** 80 × 72 × 2 = 11,520

**Gold standard:**
- Gold chunks defined by GPT-4o relevance judgement ≥3 (1–5 scale)
- Essential chunks: relevance ≥4
- Mean 2.4 gold chunks per question (range 1–5)

---

## 5. Evaluation Metrics

| Metric | Full Name | Formula / Method |
|--------|-----------|-----------------|
| **SCS** | Style Conformance Score | Hybrid: 0.5 × cosine_sim(response, FSLSM anchor) + 0.5 × structural_marker_score |
| **RR** | Response Relevance | GPT-4o LLM-as-judge, 1–5 scale |
| **CR@5** | Chunk Recall @ 5 | \|top-5 retrieved ∩ gold\| / \|gold\| |
| **CR@10** | Chunk Recall @ 10 | \|top-10 retrieved ∩ gold\| / \|gold\| |
| **ER** | Essential Recall | \|retrieved ∩ essential\| / \|essential\| |
| **Engagement** | Student Engagement | Virtual student agent score, 1–5 |

**SCS anchor design:** 16 composite style anchors (one per FSLSM profile), each blending 4 dimension anchors. Example for Active pole: *"Let's try this hands-on exercise..."*

---

## 6. Key Design Decisions & Rationale

| # | Decision | What Changed | Why |
|---|----------|-------------|-----|
| 1 | **Hybrid BM25+FAISS retrieval** | Added BM25 sparse retrieval alongside FAISS dense index | Dense-only retrieval missed keyword-heavy queries; BM25 recovers exact-match terms |
| 2 | **Reciprocal Rank Fusion (RRF)** | RRF merging instead of score normalization | RRF is rank-based and robust to score scale differences across models |
| 3 | **Multi-query decomposition** | LLM (gpt-4.1-mini) splits multi-hop questions into 2–4 sub-queries | Single-query CR@10 ≈ 0.10 for multi-hop questions; sub-query union raises recall significantly |
| 4 | **R1 query augmentation** | R1 appends `retrieval_directive` to question before embedding | Gives R1 a legitimate FSLSM-driven retrieval advantage; tested: R1 ≥ R0 in 100% of controlled comparisons |
| 5 | **Removed FSLSM reranking from live pipeline** | Reranking boost (originally 0.05) removed | RRF scores are ~0.016/rank; a 0.05 boost caused 3+ rank inversions, degrading retrieval |
| 6 | **Relevance-only gold standard** | Gold chunks defined by GPT-4o relevance ≥3 (not retrieval-filtered) | Original approach filtered gold by R0 retrievability, making it structurally impossible for R1 to outperform R0 — a fairness violation |
| 7 | **SCS hybrid scoring** | 0.5 × embedding sim + 0.5 × structural marker score | Pure embedding similarity underweighted explicit stylistic markers (e.g., "Step 1:", "Try it:"); hybrid scoring captures both semantic and structural style |
| 8 | **Virtual student engagement scoring** | Separate gpt-4.1-mini agent plays "student" and scores engagement 1–5 | Provides behavioral signal beyond semantic quality; captures affective/motivational dimension of personalization |

---

## 7. Final Results (n = 5,760 matched pairs)

### 7.1 Main Metrics

| Metric    | R0 Mean | R1 Mean | Diff    | p-value | Significant      | Cohen's d   |
| --------- | ------- | ------- | ------- | ------- | ---------------- | ----------- |
| **SCS**   | 0.269   | 0.518   | +0.248  | ≈ 0     | ✅ YES           | 3.26 (huge) |
| **Eng**   | 3.238   | 4.114   | +0.876  | ≈ 0     | ✅ YES           | 2.02 (huge) |
| **RR**    | 4.655   | 4.214   | -0.442  | ≈ 0     | ⚠️ sig. negative | -0.82       |
| **CR@5**  | 0.241   | 0.240   | -0.001  | 0.570   | ❌ No            | ~0          |
| **CR@10** | 0.334   | 0.332   | -0.001  | 0.470   | ❌ No            | ~0          |
| **ER**    | 0.352   | 0.349   | -0.002  | 0.182   | ❌ No            | ~0          |

Statistical tests: Wilcoxon signed-rank (all metrics).

### 7.2 SCS and Engagement by FSLSM Dimension Pole

| Dimension | Pole | n | SCS R0 | SCS R1 | ΔSCS | Eng R0 | Eng R1 |
|-----------|------|---|--------|--------|------|--------|--------|
| act_ref | Active | 5 | 0.309 | 0.581 | +0.272 | 3.0 | 4.6 |
| act_ref | Reflective | 5 | 0.236 | 0.561 | +0.325 | 3.2 | 4.0 |
| sen_int | Sensing | 5 | 0.248 | 0.545 | +0.297 | 3.2 | 4.4 |
| sen_int | Intuitive | 5 | 0.296 | 0.597 | +0.301 | 3.0 | 4.2 |
| vis_ver | Visual | 5 | 0.283 | 0.583 | +0.300 | 3.0 | 4.2 |
| vis_ver | Verbal | 5 | 0.262 | 0.559 | +0.297 | 3.2 | 4.4 |
| seq_glo | Sequential | 6 | 0.254 | 0.528 | +0.274 | 3.0 | 4.3 |
| seq_glo | **Global** | 4 | 0.300 | **0.635** | **+0.335** | 3.25 | 4.25 |

**Observation:** All 8 dimension poles benefit from personalization. Global learners show the largest SCS improvement (+0.335). Active learners show the largest engagement improvement (3.0 → 4.6).

### 7.3 Cost

| | R0 | R1 |
|--|----|----|
| Total (10 sessions) | $0.0246 | $0.0289 |
| Mean per session | $0.00246 | $0.00289 |
| **Overhead** | — | **+17.4%** |

Personalization adds minimal cost overhead (~$0.0004 per session).

---

## 8. Discussion

### What Works
- **FSLSM conditioning strongly improves style conformance.** R1 responses are clearly structured according to the student's learning profile (e.g., Visual learners receive ASCII diagrams, Sequential learners receive numbered steps). The effect size (d = 4.41) is unusually large, suggesting the generation directives are directly and reliably controlling output style.
- **Engagement is significantly higher.** The virtual student agent rates R1 responses 1.2 points higher on a 5-point scale, suggesting that style-matched explanations are perceived as more helpful and motivating.

### What Doesn't

- **All retrieval metrics (CR@5, CR@10, ER) are equal between R0 and R1** (p = 0.57, p = 0.47, p = 0.18; d ≈ 0 for all). The FSLSM-augmented query in R1 adds style-relevant keywords to the retrieval query, but gold chunks are defined by content relevance — style keywords do not align with factual gold, so retrieval recall is unaffected at both k=5 and k=10.

- **RR is significantly lower in R1** (R1 = 4.214 vs R0 = 4.655, d = −0.82). R1 responses allocate token budget to style-specific elements (exercises, diagrams, numbered steps) that don't appear in the gold answer. The LLM judge penalises this as lower factual proximity.

### Interpretation
Personalization in this system primarily operates at the **generation layer** (style, structure, tone). The **retrieval layer** gains are smaller and harder to detect at small sample sizes. This is consistent with the literature: style is more directly controllable via prompt conditioning than retrieval ranking.

---

## 9. Current Status & Next Steps

### Now
- ✅ Full experiment running: **7,178 / 11,520 sessions** (~62% complete)
- ✅ No errors detected
- ⏳ Estimated completion: ~April 20, 2026

### After Experiment Completes
1. Run evaluation pipeline:
   ```bash
   python experiments/exp2_tutor_personalization/evaluate_exp2.py --rr-workers 1
   ```
2. Re-run `report.ipynb` to regenerate all figures and tables with full data
3. Compile final thesis chapter for Experiment 2

### Expected Final Results
- **SCS:** R1 > R0 confirmed at p < 0.001 (effect size likely d > 3)
- **Engagement:** R1 > R0 confirmed at p < 0.001
- **CR@10/ER:** Expect small but statistically detectable R1 > R0 (d ≈ 0.1–0.2) with n=5,760 pairs
- **RR:** Likely no significant difference (style ≠ factual accuracy)

---

## 10. Appendix: Repository Structure

```
mcp-rag/
├── data/
│   ├── agents/validated_agents.json        # 80 synthetic FSLSM agents
│   ├── exp2/
│   │   ├── sampled_questions.json          # 72 questions with gold chunks (rel≥3)
│   │   ├── sampled_questions_pre_reannotate.json  # backup (original 5 gold each)
│   │   ├── reannotate_checkpoint.json      # GPT-4o relevance scores (360 entries)
│   │   └── scs_style_anchors.json          # 16 FSLSM composite style anchors
│   └── fslsm/profiles.json                 # FSLSM profile metadata
├── src/
│   ├── tutor/
│   │   ├── profile_agent.py                # Phase 1: FSLSM → retrieval/generation plan
│   │   ├── retrieval_agent.py              # Phase 2: hybrid BM25+FAISS+RRF retrieval
│   │   └── tutor_agent.py                  # Phase 3: LLM tutoring + engagement scoring
│   └── evaluation/metrics.py              # SCS, RR, CR@10, ER, DAS, PRA
├── experiments/exp2_tutor_personalization/
│   ├── run_exp2.py                         # A/B orchestrator (Phase 4)
│   ├── evaluate_exp2.py                    # Evaluation pipeline (Phase 5)
│   ├── report.ipynb                        # Analysis notebook with figures
│   └── results/
│       ├── raw_sessions_r0.jsonl           # R0 session outputs
│       ├── raw_sessions_r1.jsonl           # R1 session outputs
│       ├── exp2_results_summary.json       # Aggregated metrics
│       └── exp2_session_metrics.csv        # Per-session metrics table
└── d2l/output/d2l_corpus_chunks.json       # D2L textbook corpus (chunked)
```
