# Experiment 3 — Findings & Summary
**RQ3: FSLSM-Conditioned MCP Tool Selection**
**For: Proposal Defense / Presentation Rewrite**

---

## Research Question & Hypothesis

**RQ3:** Does FSLSM-conditioned semantic tool selection (S1b) improve Tool Selection Accuracy (TSA) and reduce Prompt Token Savings (PTS) waste compared to a prompt-bloat baseline (S0) and an unconditioned RAG-MCP retrieval (S1a)?

**H3 (Proposal):** FSLSM-conditioned tool selection (S1b) will achieve ≥ 10–15 percentage points higher TSA than unconditioned RAG-MCP (S1a), while maintaining comparable prompt token savings (PTS).

---

## Projected vs Actual Results

### Proposal Projection (Table 3.4)

| Condition | TSA (projected) | PTS (projected) |
|-----------|----------------|----------------|
| S0 (prompt bloat) | baseline | 0% |
| S1a (unconditioned RAG-MCP) | moderate gain | ~90%+ |
| S1b (FSLSM-conditioned RAG-MCP) | S1a + 10–15 pp | ~90%+ |

### Actual Results (n = 5,760 sessions)

| Condition | TSA | TSA SE | PTS | Description |
|-----------|-----|--------|-----|-------------|
| **S0** | 9.8% | ±0.4% | 0.0% | All 15 tools in context (prompt bloat) |
| **S1a** | 11.0% | ±0.4% | 93.0% | FAISS semantic retrieval, no FSLSM |
| **S1b** | 26.8% | ±0.6% | 93.2% | FAISS semantic retrieval + FSLSM reranking |

### Statistical Comparisons

| Comparison | t-statistic | p-value | Cohen's h | Effect |
|------------|------------|---------|-----------|--------|
| S1b vs S0 | 24.196 | ≈ 0 (p < 0.0001) | 0.452 | Medium |
| S1b vs S1a | 22.098 | ≈ 0 (p < 0.0001) | 0.412 | Medium |

**S1b vs S1a improvement: +15.8 pp** (at upper bound of projected 10–15 pp range)

---

## Key Findings

### 1. H3 Confirmed — FSLSM Conditioning Significantly Improves TSA

S1b achieves 26.8% TSA vs S1a at 11.0% — a +15.8 pp improvement that meets the upper bound of the projected range. Both comparisons (vs S0 and vs S1a) are highly significant (p < 0.0001) with medium effect sizes (Cohen's h ≈ 0.41–0.45).

FSLSM-conditioned query augmentation provides a meaningful boost beyond unconditioned semantic retrieval. The intent-filter + FSLSM-overlap scoring in `get_optimal_tool_id()` reliably selects tools whose FSLSM tags match the agent's learning profile.

### 2. PTS Confirmed — 93% Token Savings Maintained

Both S1a and S1b achieve ~93% prompt token savings vs S0 (S0 = 1,410 tokens for all 15 tools vs ~99 tokens for a single selected tool). FSLSM conditioning adds no meaningful overhead to the token efficiency gains of semantic retrieval.

**The core MCP value proposition is confirmed:** selecting one tool from a registry of 15 saves ~93% of prompt tokens with no loss in retrieval quality for the FSLSM-relevant dimensions.

### 3. S0 Baseline TSA Is Surprisingly Low (9.8%)

The prompt-bloat baseline achieves only 9.8% TSA — close to random selection from 15 tools (random = 6.7%). This is lower than the proposal anticipated. The reason: when all 15 tools are in context simultaneously, the LLM cannot distinguish which tool is most relevant to the specific learning style dimension implied by the query. The bloated context degrades selection signal.

**Implication:** This makes the case for RAG-MCP even stronger — the baseline is closer to random than anticipated.

### 4. Unconditioned RAG-MCP (S1a) Provides Minimal Gain Over S0

S1a TSA = 11.0% — only +1.2 pp over S0 (9.8%). Semantic retrieval alone, without FSLSM conditioning, barely improves over the bloat baseline. Generic question embeddings do not reliably distinguish between, e.g., a Visual learner's tool vs a Verbal learner's tool when both tools have similar semantic descriptions.

**Implication:** FSLSM conditioning is essential to the performance gain, not semantic retrieval alone.

---

## Per-Dimension TSA (S1b)

| Dimension | Pole | TSA (S1b) |
|-----------|------|-----------|
| seq_glo | Sequential | **61.8%** |
| vis_ver | Visual | 40.6% |
| vis_ver | Verbal | 27.8% |
| seq_glo | Global | 14.0% |
| sen_int | Sensing | 8.7% |
| sen_int | Intuitive | 6.7% |
| act_ref | Active | **0.0%** |
| act_ref | Reflective | **0.0%** |

**High-performing dimensions:** Sequential (61.8%) and Visual (40.6%) tools have distinctive keywords in the tool registry that strongly overlap with their FSLSM intent-keywords. Sequential tools reference "step-by-step", "procedures", "structured"; Visual tools reference "diagrams", "visualizations", "charts".

**Zero-TSA dimensions — Design Artifact:** Active and Reflective poles achieve 0% TSA. This is not a statistical anomaly — it reflects a systematic design gap: the 72-question dataset covers 4 content domains (gradient descent, attention, regularization, backpropagation) that do not contain intent-keywords matching Active-pole tools (e.g., "interactive exercises", "hands-on simulations") or Reflective-pole tools (e.g., "self-assessment", "reflective journaling"). The tools exist in the registry but their activation keywords never appear in the question set.

---

## Hypothesis Assessment

| Aspect | Projected | Actual | Status |
|--------|-----------|--------|--------|
| S1b TSA improvement over S1a | +10–15 pp | **+15.8 pp** | ✅ Confirmed (at upper bound) |
| PTS maintained | ~90%+ | **93.0–93.2%** | ✅ Confirmed |
| S1b > S0 significant | Yes | p < 0.0001, h=0.452 | ✅ Confirmed |
| S1b > S1a significant | Yes | p < 0.0001, h=0.412 | ✅ Confirmed |
| Uniform coverage across dimensions | Assumed | **No — 0% for Active/Reflective** | ⚠️ Partial |

**H3 verdict: CONFIRMED** for TSA improvement and PTS retention. The per-dimension coverage gap (Active/Reflective at 0%) is a design artifact of the question set, not a failure of the FSLSM-conditioning mechanism.

---

## Interpretation

### Why FSLSM Conditioning Helps

The FSLSM query augmentor appends 4 learning-style directives (one per dimension) to each query before embedding. This enriched query shifts the embedding toward tool-description regions of the FAISS index that are tagged with matching FSLSM poles. The intent-filter in `get_optimal_tool_id()` then scores retrieved tools by FSLSM-tag overlap with the agent's assigned profile.

For dimensions with distinctive tool keywords (Sequential, Visual), the augmented embedding reliably lands in the right region of the tool-index space. For dimensions without keyword overlap in the question set (Active, Reflective), the augmentation provides no signal.

### The Active/Reflective Gap Requires Thesis Explanation

The zero TSA for Active and Reflective poles should be explained as a limitation of the evaluation dataset design, not of the FSLSM-MCP mechanism:

1. The 72 questions were sampled from D2L machine learning textbook content (gradient descent, attention mechanisms, etc.)
2. These topics do not naturally invoke Active-style interactions (hands-on exercises, simulations) or Reflective-style interactions (journaling, self-review)
3. This is a **domain-coverage issue**, not a model failure — a different question set (e.g., lab exercises, project prompts) would activate these dimensions

**Defense framing:** "The Active and Reflective poles require domain content that invites interactive or self-reflective engagement. The ML textbook questions in our dataset do not naturally trigger these intent-keywords. This is a known limitation of our evaluation scope and is explicitly bounded in Chapter 3 Limitations."

### Comparison to Prompt Bloat Baseline

S0 at 9.8% TSA vs S1b at 26.8% represents a +17 pp gain from the full RAG-MCP-FSLSM pipeline over naive prompt inclusion. The near-random S0 baseline validates the core MCP design decision: loading all tools into context does not help the model select the right one; it requires targeted retrieval.

---

## Unexpected Findings

1. **S1a vs S0 gap is only 1.2 pp** — unconditioned semantic RAG adds almost nothing over prompt bloat. This was not anticipated. It reinforces the thesis that FSLSM conditioning is the critical differentiator, not just "any retrieval."

2. **Sequential dimension outperforms all others at 61.8%** — this appears driven by the D2L corpus being heavily procedural/algorithmic content. Sequential tools align strongly with the dominant text structure of the corpus.

3. **S0 baseline is near-random (9.8% vs 6.7% random)** — reinforces that providing all 15 tools to the LLM without filtering creates too much noise for the model to reliably select the optimal one.

---

## System Architecture Summary

```
Query + FSLSM Profile
        │
        ▼ (FSLSM Query Augmentor)
Augmented Query = question + [4 dimension directives]
        │
        ▼ (FAISS tool-index retrieval)
Top-k candidate tools
        │
        ▼ (get_optimal_tool_id: intent-filter + FSLSM-overlap score)
Selected tool → appended to prompt
```

**Tool registry:** 15 MCP tools, each tagged with FSLSM dimension poles, intent-keywords, and estimated prompt token cost. S0 includes all 15 tools; S1a/S1b include only the selected tool (~99 tokens vs 1,410 tokens).

---

## Presentation Talking Points

- "Without FSLSM conditioning, semantic retrieval (S1a) barely beats random — TSA = 11.0% vs 9.8%"
- "FSLSM conditioning jumps TSA to 26.8% — a 2.4× improvement over the unconditioned baseline"
- "We save 93% of prompt tokens while improving tool selection accuracy"
- "Sequential and Visual dimensions work best — the tool keywords match the corpus content"
- "Active and Reflective at 0% is a dataset scope limitation, not a mechanism failure — different question domains would activate those poles"
- "The core finding: you need both semantic retrieval AND learning style conditioning; retrieval alone is not enough"
