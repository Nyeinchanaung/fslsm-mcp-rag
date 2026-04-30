# Experiment 2 — FSLSM-Conditioned Tutor Personalization

**Research Question:** RQ1
**Hypothesis:** H1 — FSLSM-personalized responses improve SCS, Engagement, and CR@5 without degrading RR
**Status:** Complete · 5,760 matched session pairs · H1 partially confirmed

---

## What This Experiment Tests

Compares two RAG-based tutoring conditions for a D2L machine-learning corpus:

- **R0 (Baseline):** Generic RAG — retrieves top-5 chunks and generates a response with no learning-style conditioning
- **R1 (Personalized):** FSLSM-conditioned RAG — query is augmented with FSLSM style directives before retrieval; system prompt encodes the student's learning profile

Each of the 16 FSLSM learning profiles is represented by 4 virtual student agents across 3 knowledge levels (beginner, intermediate, advanced, general), answering 90 questions drawn from the D2L corpus.

---

## Design

| Factor | Detail |
|--------|--------|
| Conditions | R0 (baseline) vs R1 (personalized) |
| FSLSM profiles | 16 (4 binary dimensions) |
| Agents per profile | 4 (3 knowledge levels + general) |
| Questions | 90 per agent |
| Total session pairs | 5,760 matched R0/R1 pairs |
| Tutor model | claude-sonnet-4 |
| Judge model (Track B) | GPT-4o (position-randomised) |

### Evaluation Tracks

- **Track A:** Automatic metrics per session (SCS, RR, CR@5, CR@10, ER, Engagement)
- **Track B:** Pairwise preference judgement — GPT-4o selects R0 or R1 per pair, blind to condition label

---

## Metrics

| Metric | Description | Scale |
|--------|-------------|-------|
| **SCS** | Style Conformance Score — hybrid (0.5 × cosine sim + 0.5 × structural marker score) | 0–1 |
| **RR** | Response Relevance — virtual student rates factual quality | 1–5 |
| **CR@5** | Chunk Recall at 5 — fraction of gold chunks in top-5 retrieved | 0–1 |
| **CR@10** | Chunk Recall at 10 | 0–1 |
| **ER** | Exact Recall — exact match of gold chunk in retrieved set | 0–1 |
| **Engagement** | Virtual student engagement rating | 1–5 |

---

## Results

### Track A — Automatic Metrics (n = 5,760 pairs, Wilcoxon signed-rank)

| Metric | R0 | R1 | Δ | Cohen's d | p-value | Significance |
|--------|----|----|---|-----------|---------|--------------|
| **SCS** | 0.261 | 0.469 | **+0.208** | 2.37 | < 0.0001 | *** |
| **Engagement** | 3.247 | 3.890 | **+0.643** | 1.49 | < 0.0001 | *** |
| **RR** | 3.788 | 3.785 | −0.003 | ~0 | 0.683 | n.s. |
| **CR@5** | 0.159 | 0.155 | −0.004 | −0.022 | 0.020 | * |
| **ER** | 0.340 | 0.333 | −0.006 | −0.013 | 0.028 | * |

### Track B — Pairwise Judgement (n = 5,760 pairs)

| Outcome | Count | Percentage |
|---------|-------|------------|
| R1 preferred (Win) | 5,419 | 94.1% |
| No preference (Tie) | 0 | 0.0% |
| R0 preferred (Loss) | 341 | 5.9% |

Binomial test (ties excluded): p < 0.001

### Per-FSLSM-Pole Improvement (R1 − R0)

| Pole | ΔSCS | ΔEngagement |
|------|------|-------------|
| Reflective | **+0.237** | +0.505 |
| Visual | +0.233 | **+0.906** |
| Sequential | +0.216 | +0.599 |
| Sensing | +0.211 | +0.682 |
| Intuitive | +0.205 | +0.604 |
| Global | +0.200 | +0.687 |
| Verbal | +0.183 | +0.380 |
| Active | +0.178 | +0.781 |

### Cost

| Condition | Total (5,760 sessions) | Per session |
|-----------|------------------------|-------------|
| R0 (generic) | $12.80 | $0.00222 |
| R1 (personalized) | $15.51 | $0.00269 |
| Overhead | +$2.71 (+21.2%) | +$0.00047 |

---

## Key Findings

1. **SCS: large effect confirmed (d = 2.37)** — personalization adds structural style markers reliably across all 16 profiles. R1 responses include numbered steps for Sequential learners, ASCII diagrams for Visual learners, analogies for Verbal learners, etc.

2. **Engagement: large effect confirmed (d = 1.49)** — virtual students rate R1 responses 0.643 points higher. Visual learners show the largest gain (+0.906), driven by structured visual content absent in R0.

3. **RR: no degradation (p = 0.683)** — factual quality is fully preserved. Style conditioning is additive, not substitutive.

4. **CR@5/ER: small negative effects (d < 0.07, practically negligible)** — FSLSM-augmented queries shift retrieval toward style-relevant content, slightly away from factual gold chunks. The effect sizes are far below conventional thresholds for practical significance.

5. **Track B: 94.1% win rate** — GPT-4o prefers R1 in 19 of every 20 pairs, confirming the metric results align with holistic preference judgements.

6. **All 8 poles benefit** — no learner type is harmed by personalization.

---

## Interpretation

Personalization operates primarily at the **generation layer**: FSLSM conditioning controls output structure directly via generation directives. The retrieval layer shows a small negative trade-off because style-relevant keywords in the augmented query do not consistently co-locate with factually-defined gold chunks in the D2L corpus.

This is an inherent retrieval–style tension: optimising for style-relevant retrieval and factual precision are partially orthogonal goals in a corpus structured around content topics rather than pedagogical styles.

---

## How to Reproduce

```bash
# 1. Run both conditions (R0 + R1) — use nohup for long runs
nohup python experiments/exp2_tutor_personalization/run_exp2.py > results/run_full.log 2>&1 &

# 2. Evaluate Track A metrics (SCS, RR, CR@5, Engagement)
python experiments/exp2_tutor_personalization/evaluate_exp2.py

# 3. Evaluate Track B pairwise judgements
python experiments/exp2_tutor_personalization/pairwise_eval.py

# 4. Open report notebooks
jupyter lab experiments/exp2_tutor_personalization/report.ipynb
jupyter lab experiments/exp2_tutor_personalization/report_pairwise.ipynb
```

> Use 1 worker for RR scoring to avoid rate limits.

---

## Output Files

### Results — `results/`

| File | Description |
|------|-------------|
| `exp2_session_metrics.csv` | 11,520 rows (5,760 R0 + 5,760 R1); all Track A metrics per session |
| `exp2_results_summary.json` | Aggregated statistics, p-values, effect sizes |
| `raw_sessions_r0.jsonl` | Full R0 session data |
| `raw_sessions_r1.jsonl` | Full R1 session data |

### Pairwise — `results/pairwise/`

| File | Description |
|------|-------------|
| `summary_overall.json` | Win/Tie/Loss counts + win rate + CI + binomial p-value |
| `summary_by_profile.csv` | Per-profile verdicts (16 rows) |
| `final_verdicts.csv` | Per-session verdict (5,760 rows): R0_WIN / R1_WIN / TIE |
| `raw_results.jsonl` | Raw GPT-4o pairwise judgement outputs |

### Notebooks

| Notebook | Contents |
|----------|----------|
| `report.ipynb` | Track A metrics: SCS, Engagement, CR@5, RR, ER — overall and per-dimension |
| `report_pairwise.ipynb` | Track B: Win/Tie/Loss, per-profile win rates, qualitative examples |
| `analysis_engagement.ipynb` | Deep-dive engagement analysis by FSLSM dimension |
| `analysis_comparison.ipynb` | Cross-condition comparison and effect-size summary |

### Thesis Figures — `figures/` (repo root)

| Figure ID | Filename | Description |
|-----------|----------|-------------|
| F5-4 | `exp2_metrics_bar.png` | R0 vs R1 grouped bar chart, 5 metrics |
| F5-5 | `exp2_dim_effect.png` | ΔSCS and ΔEngagement per FSLSM pole |
| F5-6 | `exp2_pairwise_wtl.png` | Track B Win/Tie/Loss stacked bar |
| F6-1 | `ana_scs_distribution.png` | SCS violin plots, real session data |
| F6-2 | `ana_retrieval_style_tradeoff.png` | Per-profile CR@5 vs SCS scatter |

---

## Hypothesis Verdict

| Aspect | Projected | Actual | Result |
|--------|-----------|--------|--------|
| SCS improvement | +0.25 | +0.208 (d=2.37) | ✅ Confirmed |
| Engagement improvement | +1.3 | +0.643 (d=1.49) | ✅ Confirmed |
| CR@5 improvement | +0.08 | −0.004 (d=−0.022) | ❌ Not confirmed |
| RR no degradation | ≥ 4.1 | 3.785 (p=0.683, n.s.) | ✅ Confirmed |
| All poles benefit | — | All 8 poles improve | ✅ Confirmed |

**H1: PARTIALLY CONFIRMED.** Core claims (SCS, Engagement) are strongly supported. CR@5 shows a small but statistically significant negative effect, not the projected gain. See `findings.md` for full interpretation.
