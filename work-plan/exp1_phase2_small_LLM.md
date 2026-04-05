# Experiment 1 Phase 2 — Small LLM Models (1–5B Parameters)

> **Motivation:** The initial 6-model experiment (3 local 7B–12B + 2 API) yielded results clustering at either ~1.0 PRA (high fidelity) or ~0.72 (Llama 3.1:8b), with no clear gradient in between. Adding smaller models from the same families tests whether parameter count is the primary bottleneck for maintaining multi-dimensional FSLSM personas. The goal is to identify the **capability cliff** — the minimum model size that can reliably hold 4 simultaneous learning-style traits across 44 ILS questions.

---

## Model Selection

| New Model | Params | Ollama Tag | Same-Family Comparison | Rationale |
|---|---|---|---|---|
| **qwen2.5:3b** | 3B | `ollama/qwen2.5:3b` | qwen2.5:7b (7B, PRA=0.956) | Direct size comparison within Qwen2.5 family |
| **gemma3:4b** | 4B | `ollama/gemma3:4b` | gemma3:12b (12B, PRA=1.000) | Perfect-scoring family — does 1/3 the params collapse? |
| **llama3.2:3b** | 3B | `ollama/llama3.2:3b` | llama3.1:8b (8B, PRA=0.724) | Newer generation + smaller — does architecture improvement compensate? |
| **phi4-mini** | 3.8B | `ollama/phi4-mini` | — (new family) | Microsoft's efficiency-focused small model, cross-family data point |

### Expected outcome
```
~12B (gemma3:12b)      → ~1.000 PRA
~7-9B (qwen, gemma2)   → ~0.72–0.96 PRA
~3-4B (new models)     → ~0.50–0.80 PRA  ← the interesting zone
```

This creates a **parameter-count vs fidelity curve** across 3 model sizes within two families (Qwen, Gemma), plus cross-family comparisons.

---

## Experiment Design

Identical to the existing experiment — no changes to prompts, questionnaire, or evaluation metrics:

- **80 FSLSM agents** per model (16 profiles x 5 instances)
- **5 baseline agents** per model (no persona)
- **3 trials** per agent (total: 240 FSLSM + 15 baseline records per model)
- **44 ILS questions** per trial
- **Temperature**: 0.3 (same as all other models)
- **System prompt**: identical template via `build_student_system_prompt()`
- **Evaluation**: PRA (Profile Recovery Accuracy) + DAS (Dimension Alignment Score)
- **Per-question alignment**: 44-question breakdown via `compute_per_question_alignment()`

---

## Implementation Steps

### 1. Pull Ollama Models
```bash
ollama pull qwen2.5:3b      # ~2 GB
ollama pull llama3.2:3b      # ~2 GB  
ollama pull phi4-mini         # ~2.5 GB
# gemma3:4b already available locally (3.3 GB)
```

### 2. Register Models
**`src/utils/llm_client.py`** — add to `MODEL_REGISTRY`:
```python
"qwen2.5:3b":   "ollama/qwen2.5:3b",
"gemma3:4b":    "ollama/gemma3:4b",
"llama3.2:3b":  "ollama/llama3.2:3b",
"phi4-mini":    "ollama/phi4-mini",
```

**`experiments/exp1_agent_fidelity/config.yaml`** — add under `models:`:
```yaml
- name: qwen2.5:3b
  temperature: 0.3
- name: gemma3:4b
  temperature: 0.3
- name: llama3.2:3b
  temperature: 0.3
- name: phi4-mini
  temperature: 0.3
```

### 3. Run Experiments (one model at a time)
```bash
# Qwen 2.5:3b
python experiments/exp1_agent_fidelity/run.py --model qwen2.5:3b
python experiments/exp1_agent_fidelity/run_baseline.py --model qwen2.5:3b

# Gemma3:4b
python experiments/exp1_agent_fidelity/run.py --model gemma3:4b
python experiments/exp1_agent_fidelity/run_baseline.py --model gemma3:4b

# Llama 3.2:3b
python experiments/exp1_agent_fidelity/run.py --model llama3.2:3b
python experiments/exp1_agent_fidelity/run_baseline.py --model llama3.2:3b

# Phi4-mini
python experiments/exp1_agent_fidelity/run.py --model phi4-mini
python experiments/exp1_agent_fidelity/run_baseline.py --model phi4-mini
```

**Estimated time**: ~1–2h per model at 3-4B scale (shorter than 7B+ runs).
**Resume support**: built into `ils_evaluator.py` — if Ollama crashes, restart and rerun the same command.

### 4. Update Analysis Pipeline
Add all 4 new models to the MODELS list in:
- `experiments/exp1_agent_fidelity/analyze.py`
- `experiments/exp1_agent_fidelity/visualize.py`
- `experiments/exp1_agent_fidelity/report.ipynb` (Cell 1)

### 5. Regenerate Metrics & Figures
```bash
python experiments/exp1_agent_fidelity/analyze.py
python experiments/exp1_agent_fidelity/visualize.py
```

### 6. Update Report
- Update Key Findings with 10-model results table
- Add "Parameter Scaling Analysis" subsection
- Re-execute notebook: `jupyter nbconvert --execute --inplace report.ipynb`

---

## Output Files (per new model)

| File | Content |
|---|---|
| `results/exp1/metrics/{safe}_results.json` | 240 FSLSM agent records |
| `results/exp1/metrics/{safe}_baseline_results.json` | 15 baseline agent records |
| `results/exp1/raw_responses/{agent_uid}_trial{1,2,3}.json` | Raw per-question responses |
| `results/exp1/figures/heatmap_{safe}.png` | FSLSM profile heatmap |

Regenerated aggregates (all 10 models):
- `results/exp1/metrics/pra_das_summary.csv`
- `results/exp1/metrics/das_summary.csv`
- `results/exp1/metrics/baseline_natural_style.csv`
- `results/exp1/metrics/per_question_alignment.csv`
- `results/exp1/metrics/cost_summary.csv`
- `results/exp1/figures/per_question_alignment_heatmap.png`
- `results/exp1/figures/model_comparison_pra.png`
- All other aggregate figures

---

## Verification Checklist

- [ ] `ollama list` shows all 4 new models
- [ ] Each model produces 240 FSLSM + 15 baseline raw response files
- [ ] `pra_das_summary.csv` has rows for all 10 models
- [ ] `baseline_natural_style.csv` has 10 rows
- [ ] `per_question_alignment.csv` has 440 rows (44 questions x 10 models)
- [ ] `report.ipynb` executes cleanly with all 10 models
- [ ] PRA gradient visible: higher-param models > lower-param models within same family

---

## Analysis Questions to Answer

1. **Within-family scaling**: Does qwen2.5:3b < qwen2.5:7b? Does gemma3:4b < gemma3:12b? By how much?
2. **Cross-generation**: Is llama3.2:3b better or worse than llama3.1:8b despite being smaller?
3. **Per-dimension failure**: Do small models fail on the same dimensions (sen_int, seq_glo) as llama3.1:8b?
4. **Baseline natural style**: Do small models show the same Active-Sensing-Visual-Sequential baseline as their larger siblings?
5. **Capability cliff**: Is there a sharp drop-off at some parameter count, or is degradation gradual?
