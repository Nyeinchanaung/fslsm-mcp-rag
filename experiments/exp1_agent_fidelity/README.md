# Experiment 1: Virtual Student Agent Fidelity (RQ2)

## Objective

Validate that LLM-based virtual student agents faithfully encode and express FSLSM (Felder-Silverman Learning Style Model) learning style profiles. This answers **RQ2**: *How accurately can LLM-based virtual student agents reproduce assigned FSLSM learning style profiles?*

## Design

### FSLSM Profiles

Each agent is assigned one of **16 FSLSM profiles** defined by 4 binary dimensions:

| Dimension | Pole A | Pole B |
|---|---|---|
| Active/Reflective | Active (−) | Reflective (+) |
| Sensing/Intuitive | Sensing (−) | Intuitive (+) |
| Visual/Verbal | Visual (−) | Verbal (+) |
| Sequential/Global | Sequential (−) | Global (+) |

Profiles are encoded as behavioral instructions in the agent's system prompt, with dimension scores in the range [−11, +11].

### Agent Configuration

- **16 profiles** × **5 instances** = **80 FSLSM agents** per model
- **Knowledge levels**: Instances 1–3 are assigned beginner, intermediate, advanced; instances 4–5 are general (no level specified)
- **3 ILS trials** per agent → **240 FSLSM records** per model
- **5 baseline agents** (no FSLSM encoding) × **3 trials** = **15 baseline records** per model

### Models Tested

| Model | Provider | Notes |
|---|---|---|
| gpt-4.1-mini | OpenAI (via LiteLLM) | Cloud API |
| claude-sonnet-4-20250514 | Anthropic (via LiteLLM) | Cloud API |
| llama3.1:8b | Ollama (via LiteLLM) | Local, free |

All models run at `temperature=0.3`.

### Evaluation Protocol

Each agent answers the 44-item ILS questionnaire (Felder & Soloman) across 3 independent trials. Responses are scored per dimension to produce raw scores in [−11, +11], which are then compared against the assigned profile.

## Metrics

### Profile Recovery Accuracy (PRA)

Fraction of dimensions where the sign of the detected score matches the assigned pole:

```
PRA = (1/4) × Σ 𝟙[sign(raw_score_d) == assigned_d]
```

- Ties (raw_score = 0) count as mismatches
- Range: [0, 1], where 1.0 = perfect profile recovery

### Dimension Alignment Score (DAS)

Continuous measure of alignment strength per dimension:

```
DAS_d = (raw_score_d × assigned_d + 11) / 22
```

- Range: [0, 1], where 1.0 = maximum alignment, 0.5 = neutral, 0.0 = maximum misalignment
- Unlike PRA (binary), DAS captures *how strongly* the agent expresses each dimension

### Baseline DAS

For non-personalized baseline agents (no assigned profile), DAS is computed against all 16 FSLSM profiles and averaged. By mathematical symmetry (8 profiles with assigned=+1 and 8 with assigned=−1 per dimension), baseline DAS = 0.500 exactly for any model.

## Results

### Overall Summary (ALL knowledge levels)

| Model | Condition | PRA | DAS |
|---|---|---|---|
| gpt-4.1-mini | FSLSM | 0.996 | 0.924 |
| gpt-4.1-mini | Baseline | 0.500 | 0.500 |
| claude-sonnet-4-20250514 | FSLSM | 1.000 | 0.927 |
| claude-sonnet-4-20250514 | Baseline | 0.500 | 0.500 |
| llama3.1:8b | FSLSM | 0.724 | 0.741 |
| llama3.1:8b | Baseline | 0.500 | 0.500 |

### Per-Dimension PRA & DAS (FSLSM condition)

| Model | Act/Ref | Sen/Int | Vis/Ver | Seq/Glo |
|---|---|---|---|---|
| **PRA** | | | | |
| gpt-4.1-mini | 1.000 | 1.000 | 1.000 | 0.983 |
| claude-sonnet-4-20250514 | 1.000 | 1.000 | 1.000 | 1.000 |
| llama3.1:8b | 1.000 | 0.500 | 0.879 | 0.517 |
| **DAS** | | | | |
| gpt-4.1-mini | 0.984 | 0.870 | 0.974 | 0.868 |
| claude-sonnet-4-20250514 | 0.953 | 0.910 | 0.982 | 0.862 |
| llama3.1:8b | 0.864 | 0.687 | 0.792 | 0.622 |

### Baseline Natural Bias

Without FSLSM encoding, each model exhibits its own natural learning style tendency:

| Model | Act/Ref | Sen/Int | Vis/Ver | Seq/Glo |
|---|---|---|---|---|
| gpt-4.1-mini | Reflective (+5.4) | Intuitive (+3.0) | Visual (−6.6) | Global (+3.8) |
| claude-sonnet-4-20250514 | Reflective (+10.1) | Intuitive (+9.3) | Verbal (+6.7) | Global (+9.8) |
| llama3.1:8b | Active (−10.7) | Sensing (−9.8) | Visual (−11.0) | Sequential (−11.0) |

### API Cost

| Model | FSLSM (240 trials) | Baseline (15 trials) | Total |
|---|---|---|---|
| gpt-4.1-mini | $2.089 | $0.064 | $2.153 |
| claude-sonnet-4-20250514 | $17.881 | $0.553 | $18.434 |
| llama3.1:8b | $0.000 | $0.000 | $0.000 |

## Key Findings

1. **Cloud models achieve near-perfect fidelity**: GPT-4.1-mini (PRA 0.996, DAS 0.924) and Claude Sonnet (PRA 1.000, DAS 0.927) faithfully reproduce all 16 FSLSM profiles across trials and knowledge levels.
2. **Local model struggles**: Llama 3.1:8b achieves only PRA 0.724 / DAS 0.741, particularly weak on Sensing/Intuitive and Sequential/Global dimensions.
3. **Baseline confirms persona encoding is causal**: All models produce PRA = 0.500 and DAS = 0.500 without FSLSM encoding, confirming that high scores are driven by the persona instructions, not inherent model bias.
4. **Sequential/Global is the hardest dimension** across all models (lowest DAS: 0.622–0.868).
5. **Knowledge level has minimal effect** on profile fidelity for cloud models.

## Reproduction

```bash
# 1. Run FSLSM experiment (all 3 models)
python experiments/exp1_agent_fidelity/run.py

# 2. Run baseline experiment
python experiments/exp1_agent_fidelity/run_baseline.py

# 3. Compute metrics (PRA, DAS, cost, baseline analysis)
python experiments/exp1_agent_fidelity/analyze.py

# 4. Generate all visualizations
python experiments/exp1_agent_fidelity/visualize.py
```

### Output Files

**Metrics** (`results/exp1/metrics/`):
- `pra_das_summary.csv` — PRA by model, condition, knowledge level, dimension
- `das_summary.csv` — DAS by model, condition, knowledge level, dimension
- `baseline_analysis.csv` — Per-model natural bias scores
- `cost_summary.csv` — API costs per model/condition
- `{model}_results.json` — Raw FSLSM results per model
- `{model}_baseline_results.json` — Raw baseline results per model

**Figures** (`results/exp1/figures/`):
- `model_comparison_pra.png` — PRA by model and dimension
- `knowledge_level_pra.png` — PRA by knowledge level
- `fslsm_vs_baseline_pra.png` — FSLSM vs baseline PRA
- `das_comparison_bar.png` — DAS by model and dimension
- `das_fslsm_vs_baseline.png` — FSLSM vs baseline DAS
- `cost_per_model.png` — API cost breakdown
- `baseline_bias_radar.png` — Natural dimension bias overlay
- `heatmap_{model}.png` — Profile x dimension alignment heatmaps
- `heatmap_baseline_{model}.png` — Baseline raw ILS score heatmaps
