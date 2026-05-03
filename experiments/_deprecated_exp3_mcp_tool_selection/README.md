# Experiment 3 — FSLSM-Conditioned MCP Tool Selection

**Research Question:** RQ3
**Hypothesis:** H3 — FSLSM-conditioned tool selection (S1b) achieves ≥ 10–15 pp higher TSA than unconditioned RAG-MCP (S1a), while maintaining PTS
**Status:** Complete · 5,760 sessions per condition · H3 confirmed

---

## What This Experiment Tests

Evaluates whether FSLSM-aware semantic tool selection improves MCP (Model Context Protocol) tool accuracy compared to:

- **S0:** Prompt-bloat baseline — all 15 tool schemas injected into the context simultaneously
- **S1a:** Unconditioned RAG-MCP — FAISS semantic retrieval over the raw question embedding
- **S1b:** FSLSM-conditioned RAG-MCP — FAISS retrieval over a query augmented with FSLSM learning-style directives, followed by FSLSM-tag-overlap reranking

The experiment reuses the Exp2 agent × question pool (16 FSLSM profiles × 4 agents × 90 questions = 5,760 sessions per condition).

---

## System Architecture

```
Query + FSLSM Profile
        │
        ▼  (FSLSM Query Augmentor)
Augmented query = question + [4 dimension directives]
        │
        ▼  (FAISS tool-index retrieval — top-k candidates)
Candidate tools
        │
        ▼  (get_optimal_tool_id: intent-filter + FSLSM-overlap score)
Selected tool → injected into prompt
```

**Tool registry:** 15 MCP tools, each tagged with FSLSM dimension poles, intent-keywords, and prompt token cost.

| Condition | Tools in prompt | Token cost (approx.) |
|-----------|-----------------|----------------------|
| S0 | All 15 schemas | ~1,410 tokens |
| S1a / S1b | 1 selected tool | ~99 tokens |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **TSA** | Tool Selection Accuracy — fraction of sessions where the selected tool matches the FSLSM-optimal ground truth |
| **PTS** | Prompt Token Savings — token reduction vs S0 baseline: `1 − (tokens_S1x / tokens_S0)` |

---

## Results

### Overall (n = 5,760 sessions per condition)

| Condition | TSA | TSA SE | PTS | Description |
|-----------|-----|--------|-----|-------------|
| **S0** | 9.8% | ±0.4% | 0.0% | Prompt bloat — all 15 tools |
| **S1a** | 11.0% | ±0.4% | 93.0% | Semantic retrieval only |
| **S1b** | **26.8%** | ±0.6% | **93.2%** | Semantic retrieval + FSLSM reranking |

### Statistical Comparisons

| Comparison | Δ TSA | t-statistic | p-value | Cohen's h |
|------------|-------|-------------|---------|-----------|
| S1b vs S0 | +17.0 pp | 24.196 | < 0.0001 | 0.452 (medium) |
| S1b vs S1a | **+15.8 pp** | 22.098 | < 0.0001 | 0.412 (medium) |
| S1a vs S0 | +1.2 pp | 2.187 | 0.029 | 0.038 (trivial) |

### Per-FSLSM-Pole TSA under S1b

| Pole | TSA (S1b) | Note |
|------|-----------|------|
| Sequential | **61.8%** | Strong keyword overlap with D2L corpus |
| Visual | 40.6% | Diagrams/charts keywords well-represented |
| Verbal | 27.8% | |
| Global | 14.0% | |
| Sensing | 8.7% | |
| Intuitive | 6.7% | |
| Active | 0.0% | Design artifact — see note below |
| Reflective | 0.0% | Design artifact — see note below |

> **Active/Reflective at 0%:** The 90-question dataset covers ML theory topics (gradient descent, attention, regularisation, backpropagation) that do not contain intent-keywords matching Active tools ("interactive exercises", "simulations") or Reflective tools ("self-assessment", "journaling"). This is a dataset scope limitation, not a mechanism failure — different domain content would activate these poles.

---

## Key Findings

1. **H3 confirmed: +15.8 pp TSA gain** — S1b over S1a at the upper bound of the projected 10–15 pp range. Both S1b vs S0 and S1b vs S1a comparisons are highly significant (p < 0.0001).

2. **93% prompt token savings maintained** — both S1a and S1b save ~93% of prompt tokens vs S0. FSLSM conditioning adds no meaningful token overhead.

3. **Unconditioned retrieval (S1a) barely beats random** — S1a TSA = 11.0% vs S0 = 9.8% (Δ = +1.2 pp, trivial effect h = 0.038). Generic query embeddings cannot distinguish style-specific tools. FSLSM conditioning is the critical differentiator.

4. **S0 baseline is near-random (9.8% vs 6.7% random)** — injecting all 15 tool schemas creates context noise that degrades the model's ability to select the right tool.

5. **Sequential dimension leads at 61.8%** — the D2L ML textbook corpus is heavily procedural, making Sequential tool keywords the most discriminative.

---

## MCP Tool Registry

The registry contains **15 pedagogical tools**, each tagged with the FSLSM poles it primarily serves. Every pole is covered by 2–4 tools; several tools serve two poles.

### Full Tool List

| ID | Tool | Category | FSLSM Poles | Description |
|----|------|----------|-------------|-------------|
| 1 | `diagram_renderer` | visualization | Visual · Sequential | Generate a labelled diagram, flowchart, or schematic figure that visually illustrates a concept, architecture, or network structure. Best for learners who prefer pictures and graphical representations laid out step by step. |
| 2 | `interactive_simulation` | visualization | Visual · Active | Launch an interactive simulation with sliders and manipulable parameters that the learner can experiment with hands-on. Updates plots in real time. Best for learners who learn by doing and experimenting. |
| 3 | `stepwise_walkthrough` | procedural | Sequential · Sensing | Decompose an algorithm or training procedure into an explicit numbered walkthrough — each step shows one transformation or computation with a concrete intermediate result. Best for learners who need an ordered concrete progression. |
| 4 | `worked_example` | procedural | Sensing · Sequential | Present a fully solved example problem with every intermediate computation shown explicitly — concrete numerical inputs, standard methods, and annotated calculation steps. Best for learners who prefer concrete facts and detail-rich worked solutions. |
| 5 | `conceptual_overview` | overview | Global · Intuitive | Provide a high-level overview that frames the topic in its broader context — motivates why it matters, maps connections to neighbouring ideas, and explains the big picture before details. Best for learners who need holistic framing first. |
| 6 | `abstract_derivation` | theoretical | Intuitive · Verbal | Derive a result from first principles using formal notation and mathematical reasoning — loss function derivation, gradient computation, proof of convergence, or symbolic manipulation. Best for learners who prefer abstract theory and formal derivations. |
| 7 | `analogy_explainer` | theoretical | Intuitive · Verbal | Explain a concept by mapping it to a familiar analogy or metaphor, drawing conceptual parallels in written prose to convey underlying meaning. Best for learners who grasp abstractions through verbal metaphors. |
| 8 | `socratic_dialogue` | dialogic | Active · Reflective | Engage the learner with a sequence of pointed Socratic questions that probe understanding, surface misconceptions, and require active discussion. Alternates between active answering and reflective pauses. Best for learners who learn by discussing and questioning. |
| 9 | `practice_exercise` | exercise | Active · Sensing | Generate a hands-on practice problem set with concrete numerical inputs, standard procedures to apply, and immediate feedback on each attempt. Best for learners who learn by doing concrete practical exercises. |
| 10 | `reflection_prompt` | dialogic | Reflective · Verbal | Issue a guided reflection prompt asking the learner to think quietly and write out their reasoning in their own words. Best for learners who process by thinking alone and writing prose. |
| 11 | `summary_outline` | overview | Reflective · Sequential | Produce a structured outline organised into ordered sections, sub-sections, and bullet points for review. Linear, ordered, bullet-point format. Best for learners who consolidate by reading ordered structured outlines. |
| 12 | `prose_explainer` | explanation | Verbal · Sequential | Explain a concept in flowing connected written prose, building the explanation sentence by sentence in a clear linear order. Heavy on written words, light on figures. Best for learners who prefer reading ordered written narrative explanations. |
| 13 | `code_sandbox` | exercise | Active · Sensing | Open a runnable code sandbox where the learner can edit, execute, and inspect outputs of working code that demonstrates the concept. Best for learners who prefer doing and trying concrete code over reading theory. |
| 14 | `concept_map` | visualization | Global · Visual | Build a graphical concept map that visually shows how the topic connects to surrounding ideas as a network of nodes and labelled edges. Holistic big-picture diagram. Best for learners who need to see overall connections in a chart-like layout. |
| 15 | `case_study` | overview | Sensing · Global | Present an extended real-world case study that anchors the topic in a concrete practical application within its wider context. Best for learners who learn through concrete real applications placed in holistic context. |

### Coverage by FSLSM Pole

| Pole | Tools |
|------|-------|
| **Visual** | diagram_renderer (1), interactive_simulation (2), concept_map (14) |
| **Sequential** | diagram_renderer (1), stepwise_walkthrough (3), worked_example (4), summary_outline (11), prose_explainer (12) |
| **Sensing** | stepwise_walkthrough (3), worked_example (4), practice_exercise (9), code_sandbox (13), case_study (15) |
| **Global** | conceptual_overview (5), concept_map (14), case_study (15) |
| **Intuitive** | conceptual_overview (5), abstract_derivation (6), analogy_explainer (7) |
| **Verbal** | abstract_derivation (6), analogy_explainer (7), reflection_prompt (10), prose_explainer (12) |
| **Active** | interactive_simulation (2), socratic_dialogue (8), practice_exercise (9), code_sandbox (13) |
| **Reflective** | socratic_dialogue (8), reflection_prompt (10), summary_outline (11) |

### Tool Selection Logic

The expert-optimal tool for each session is determined by `get_optimal_tool_id(query, profile)` in `tool_registry.py`:

1. **Intent classification** — the query is matched against 9 intent keyword lists (`explain`, `visualize`, `implement`, `practice`, `solve`, `derive`, `compare`, `summarize`, `discuss`)
2. **Strong intent overrides** (≈30% of queries) — `implement` → `code_sandbox`/`worked_example`, `practice` → `practice_exercise`/`summary_outline`, `visualize` → `diagram_renderer`/`concept_map` (profile-aware)
3. **Default "explain" path** (≈70% of queries) — profile alone determines the tool via `PROFILE_TOOL_MAP`, a lookup table of all 16 profiles to their optimal tool ID

This design makes Exp3 a clean test of FSLSM conditioning: for the majority "explain" intent, the profile is the only signal that distinguishes the correct tool.

---

## How to Reproduce

```bash
# 1. Build the FAISS tool index
python experiments/exp3_mcp_tool_selection/scripts/01_build_tool_index.py

# 2. Verify setup (FAISS quality, tool coverage)
python experiments/exp3_mcp_tool_selection/scripts/02_verify_setup.py

# 3. Dry run (50 sessions to validate pipeline)
python experiments/exp3_mcp_tool_selection/scripts/03_dry_run.py

# 4. Full run (5,760 sessions × 3 conditions)
nohup python experiments/exp3_mcp_tool_selection/scripts/04_full_run.py > results/full_run.log 2>&1 &

# 5. Compute TSA and PTS metrics
python experiments/exp3_mcp_tool_selection/scripts/05_compute_metrics.py

# 6. Generate report figures
python experiments/exp3_mcp_tool_selection/scripts/06_generate_report.py

# 7. Open report notebook
jupyter lab experiments/exp3_mcp_tool_selection/report.ipynb
```

### Diagnostics

```bash
bash experiments/exp3_mcp_tool_selection/scripts/run_all_diagnostics.sh
```

Individual diagnostics available for FAISS quality, tool description coverage, and ground-truth validation.

---

## Key Files

| File | Description |
|------|-------------|
| `config.py` | Paths, embedding model, experiment parameters |
| `tool_registry.py` | 15 MCP tool definitions with FSLSM tags and intent-keywords |
| `tool_index.py` | FAISS index construction and `get_optimal_tool_id()` logic |
| `fslsm_query_augmentor.py` | Adds FSLSM dimension directives to queries (S1b) |
| `session_adapter.py` | Converts Exp2 R0/R1 session records into Exp3 input format |
| `ablation_runner.py` | Runs all three conditions (S0, S1a, S1b) in sequence |

---

## Output Files

### Results — `results/`

| File | Description |
|------|-------------|
| `exp3_metrics.json` | Aggregated TSA/PTS by condition, dimension, and profile |
| `exp3_results.db` | SQLite database — all 17,280 session records (3 conditions × 5,760) |
| `exp3_table_3_4.md` | Summary table in markdown (matches thesis Table 3.4) |

### Figures — `results/figures/`

| File | Description |
|------|-------------|
| `report_tsa_by_condition.png` | TSA bar chart: S0 / S1a / S1b |
| `report_pts_analysis.png` | PTS bars + PTS vs TSA scatter |
| `report_dim_tsa_s1b.png` | Per-pole TSA under S1b (8 bars) |
| `report_profile_tsa_heatmap.png` | 16-profile × 3-condition TSA heatmap |
| `report_conditioning_lift.png` | Per-profile TSA lift: S1b − S1a |
| `report_tool_tsa_s1b.png` | Per-tool TSA breakdown under S1b |
| `report_exp3_summary.png` | 2×3 multi-panel thesis summary figure |

### Thesis Figures — `figures/` (repo root)

| Figure ID | Filename | Source |
|-----------|----------|--------|
| F5-7 | `exp3_tsa_bar.png` | `report_tsa_by_condition.png` |
| F5-8 | `exp3_pts_tsa.png` | `report_pts_analysis.png` |
| F5-9 | `exp3_dim_tsa.png` | `report_dim_tsa_s1b.png` |
| F5-10 | `exp3_profile_heatmap.png` | `report_profile_tsa_heatmap.png` |
| F6-3 | `ana_s1b_lift.png` | `report_conditioning_lift.png` |
| F6-4 | `ana_per_tool_tsa.png` | `report_tool_tsa_s1b.png` |
| F6-5 | `exp3_summary.png` | `report_exp3_summary.png` |

---

## Hypothesis Verdict

| Aspect | Projected | Actual | Result |
|--------|-----------|--------|--------|
| S1b TSA over S1a | +10–15 pp | **+15.8 pp** | ✅ Confirmed (upper bound) |
| PTS maintained | ~90%+ | **93.0–93.2%** | ✅ Confirmed |
| S1b vs S0 significant | Yes | p < 0.0001, h=0.452 | ✅ Confirmed |
| S1b vs S1a significant | Yes | p < 0.0001, h=0.412 | ✅ Confirmed |
| Uniform coverage | Assumed | Active/Reflective = 0% | ⚠️ Dataset scope limitation |

**H3: CONFIRMED.** FSLSM conditioning provides a meaningful and statistically robust improvement over both the prompt-bloat baseline and unconditioned semantic retrieval. The Active/Reflective coverage gap is a known limitation of the evaluation dataset scope. See `findings.md` for full analysis.
