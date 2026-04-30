# Experiment 1 — Virtual Student Agent Fidelity

**Research Question:** RQ2  
**Hypothesis:** H2 — FSLSM-conditioned LLM agents achieve PRA ≥ 0.82 and DAS ≥ 0.75  
**Status:** Complete · 15 models evaluated · H2 confirmed (exceeded)

---

## What This Experiment Tests

Validates that LLM-based virtual student agents faithfully encode and express FSLSM (Felder-Silverman Learning Style Model) learning profiles. Each agent answers the 44-item ILS questionnaire across multiple independent trials; the recovered profile is compared against the assigned one.

---

## FSLSM Profile Design

Each agent is assigned one of **16 FSLSM profiles** defined by four binary dimensions:

| Dimension | Pole A (−) | Pole B (+) |
|-----------|------------|------------|
| Active/Reflective | Active | Reflective |
| Sensing/Intuitive | Sensing | Intuitive |
| Visual/Verbal | Visual | Verbal |
| Sequential/Global | Sequential | Global |

Dimension scores are encoded in the range [−11, +11] in the agent's system prompt.

**Agent configuration per model:**
- 16 profiles × 5 instances = 80 FSLSM agents
- Knowledge levels: beginner (I01), intermediate (I02), advanced (I03), general (I04, I05)
- 3 ILS trials per agent → **240 FSLSM records** per model
- 5 baseline agents (no FSLSM encoding) × 3 trials → **15 baseline records** per model

---

## Metrics

### Profile Recovery Accuracy (PRA)
Fraction of dimensions where the detected pole matches the assigned pole:
```
PRA = (1/4) × Σ 𝟙[sign(raw_score_d) == assigned_d]
```
- Ties (raw_score = 0) count as mismatches
- Range: [0, 1] · Random chance = 0.50 · Perfect = 1.00

### Dimension Alignment Score (DAS)
Continuous alignment strength per dimension:
```
DAS_d = (raw_score_d × assigned_d + 11) / 22
```
- Range: [0, 1] · Neutral = 0.50 · Maximum alignment = 1.00
- Captures *how strongly* the agent expresses each dimension (vs. PRA's binary pass/fail)

---

## Results — All 15 Models

| Model | Params | PRA | DAS | Tier |
|-------|--------|-----|-----|------|
| claude-sonnet-4 | API | **1.000** | **0.927** | High |
| gemma3:12b | 12B | **1.000** | 0.882 | High |
| gpt-4.1-mini | API | 0.996 | 0.924 | High |
| qwen2.5:7b | 7B | 0.956 | 0.785 | High |
| gemma2:9b | 9B | 0.948 | 0.837 | High |
| phi4-mini | 3.8B | 0.915 | 0.722 | High |
| llama3.2:3b | 3B | 0.902 | 0.708 | High |
| qwen2.5:3b | 3B | 0.890 | 0.752 | High |
| llama3.1:8b | 8B | 0.724 | 0.741 | Mid |
| gemma3:4b | 4B | 0.718 | 0.718 | Mid |
| mistral:7b | 7B | 0.544 | 0.625 | Failed |
| qwen2.5:1.5b | 1.5B | 0.508 | 0.602 | Failed |
| llama3.2:1b | 1B | 0.500 | 0.500 | Failed |
| gemma2:2b | 2B | 0.500 | 0.531 | Failed |
| gemma3:1b | 1B | 0.500 | 0.500 | Failed |

*PRA = 0.50 = random chance (4 binary dimensions). H2 target = PRA ≥ 0.82.*

### Per-Dimension PRA — Top 3 Models

| Dimension | claude-sonnet-4 | gpt-4.1-mini | gemma3:12b |
|-----------|:--------------:|:------------:|:---------:|
| act_ref | 1.000 | 1.000 | 1.000 |
| sen_int | 1.000 | 1.000 | 1.000 |
| vis_ver | 1.000 | 1.000 | 1.000 |
| seq_glo | 1.000 | 0.983 | 1.000 |

### Baseline Natural Bias (No FSLSM Encoding)

Without a persona prompt, each model defaults to its own natural style:

| Model | Natural Tendency |
|-------|-----------------|
| gpt-4.1-mini | Reflective–Intuitive–Visual–Global |
| claude-sonnet-4 | Reflective–Intuitive–Verbal–Global |
| llama3.1:8b | Active–Sensing–Visual–Sequential |

All models converge to PRA = 0.500 and DAS = 0.500 on the baseline, confirming that FSLSM conditioning — not inherent model bias — drives high scores in the FSLSM condition.

### API Cost

| Model | FSLSM (240 trials) | Baseline (15 trials) | Total |
|-------|--------------------|----------------------|-------|
| gpt-4.1-mini | $2.089 | $0.064 | $2.153 |
| claude-sonnet-4 | $17.881 | $0.553 | $18.434 |
| llama3.1:8b | $0.000 | $0.000 | $0.000 |

---

## Key Findings

1. **H2 exceeded**: 8/15 models surpass the PRA ≥ 0.82 target; top models reach PRA = 1.000.
2. **Hard floor at ~2B parameters**: Models with ≤ 2B parameters universally fail (PRA ≈ 0.50 = chance). Sub-2B models cannot hold 44-item ILS context while maintaining FSLSM-aligned responses.
3. **Open-source matches cloud**: gemma3:12b achieves PRA = 1.000 at zero API cost — matching the best cloud model.
4. **seq_glo is consistently hardest**: Lowest DAS across all models; even GPT-4.1-mini scores DAS_seq_glo = 0.868 vs DAS_act_ref = 0.984.
5. **Baseline confirms causality**: PRA = 0.500 for all models without FSLSM encoding rules out inherent model bias as an explanation.
6. **Model family matters more than parameter count**: mistral:7b (PRA = 0.544) fails despite 7B parameters, while llama3.2:3b (PRA = 0.902) succeeds at 3B.

---

## How to Reproduce

```bash
# 1. Run FSLSM experiment (all models)
python experiments/exp1_agent_fidelity/run.py

# 2. Run baseline (no FSLSM encoding)
python experiments/exp1_agent_fidelity/run_baseline.py

# 3. Compute PRA, DAS, cost, and baseline analysis
python experiments/exp1_agent_fidelity/analyze.py

# 4. Generate visualisations
python experiments/exp1_agent_fidelity/visualize.py
```

All models run at `temperature=0.3` via LiteLLM (cloud models use API keys; local models use Ollama).

---

## Output Files

### Metrics — `results/exp1/metrics/`

| File | Contents |
|------|----------|
| `pra_das_summary.csv` | PRA by model, condition, knowledge level, dimension |
| `das_summary.csv` | DAS by model, condition, knowledge level, dimension (376 rows) |
| `baseline_analysis.csv` | Per-model natural bias direction and magnitude |
| `cost_summary.csv` | API cost per model/condition (30 rows) |
| `per_question_alignment.csv` | Per-question analysis across models (660 rows) |
| `{model}_results.json` | Raw FSLSM trial results per model |
| `{model}_baseline_results.json` | Raw baseline trial results per model |

### Figures — `results/exp1/figures/`

| File | Description |
|------|-------------|
| `model_comparison_pra.png` | PRA by model and dimension |
| `knowledge_level_pra.png` | PRA by knowledge level |
| `fslsm_vs_baseline_pra.png` | FSLSM vs baseline PRA |
| `das_comparison_bar.png` | DAS by model and dimension |
| `das_fslsm_vs_baseline.png` | FSLSM vs baseline DAS |
| `cost_per_model.png` | API cost breakdown |
| `baseline_bias_radar.png` | Natural dimension bias overlay |
| `heatmap_{model}.png` | Profile × dimension alignment heatmaps |
| `heatmap_baseline_{model}.png` | Baseline raw ILS score heatmaps |

### Thesis Figures — `figures/` (repo root)

| Figure ID | Filename | Description |
|-----------|----------|-------------|
| F5-1 | `exp1_pra_bar.png` | PRA bar chart, all 15 models, tier-coloured |
| F5-2 | `exp1_radar.png` | Agent #001 profile fidelity radar chart |
| F5-3 | `exp1_pra_das_scatter.png` | PRA vs DAS scatter with H2 target zone |

---

## Hypothesis Verdict

| Criterion | Target | Actual | Result |
|-----------|--------|--------|--------|
| PRA (cloud/API models) | ≥ 0.82 | 0.996–1.000 | ✅ Exceeded |
| DAS (cloud/API models) | ≥ 0.75 | 0.882–0.927 | ✅ Exceeded |
| Coverage | Frontier only | 15 models, 3 tiers | ✅ Expanded |
| Minimum viable model | Not specified | ~3B params | New finding |

**H2: CONFIRMED and exceeded.** See `findings.md` for full analysis.
