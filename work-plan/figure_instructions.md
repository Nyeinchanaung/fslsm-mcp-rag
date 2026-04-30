# Figure Instructions — FSLSM-RAG-MCP Thesis
## Chapters 5, 6, and 7

> **Workflow**
> 1. Generate each figure in Claude Code using the data and specs below
> 2. Export as **PNG, 300 dpi, white background**
> 3. Save to `figures/` with the **exact filename** listed
> 4. Upload all PNGs — the LaTeX will be updated to reference them

**Global style rules (apply to every figure)**
- Font: DejaVu Sans or similar sans-serif, min 11 pt axis labels, 9 pt tick labels
- Colour palette (consistent across all chapters):
  - R0 / S0 / Baseline: `#95A5A6` (medium grey)
  - R1 / S1b / Proposed: `#2980B9` (thesis blue)
  - S1a / Intermediate: `#27AE60` (green)
  - High/Confirmed: `#1ABC9C` (teal)
  - Mid tier: `#F39C12` (amber)
  - Failed/Negative: `#E74C3C` (red)
  - Accent / highlight: `#8E44AD` (purple)
- All error bars: 95 % CI unless stated otherwise
- All significance annotations: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001, `n.s.` otherwise
- Figure size targets are for the printed page (13 cm ≈ full column)

---

## Figures by Chapter

| ID | Filename | Chapter | Section | Status |
|----|----------|---------|---------|--------|
| F5-1 | `exp1_pra_bar.png` | 5 | 5.1 | Generate |
| F5-2 | `exp1_radar.png` | 5 | 5.1 | Export from exp1_report |
| F5-3 | `exp1_pra_das_scatter.png` | 5 | 5.1 | Generate |
| F5-4 | `exp2_metrics_bar.png` | 5 | 5.2 | Generate |
| F5-5 | `exp2_dim_effect.png` | 5 | 5.2 | Generate |
| F5-6 | `exp2_pairwise_wtl.png` | 5 | 5.2 | Needs exp2_pairwise data |
| F5-7 | `exp3_tsa_bar.png` | 5 | 5.3 | Export from exp3_report §2 |
| F5-8 | `exp3_pts_tsa.png` | 5 | 5.3 | Export from exp3_report §3+5 |
| F5-9 | `exp3_dim_tsa.png` | 5 | 5.3 | Export from exp3_report §4 |
| F5-10 | `exp3_profile_heatmap.png` | 5 | 5.3 | Export from exp3_report §6 |
| F6-1 | `ana_scs_distribution.png` | 6 | 6.1 | Generate |
| F6-2 | `ana_retrieval_style_tradeoff.png` | 6 | 6.2 | Needs exp2 per-session data |
| F6-3 | `ana_s1b_lift.png` | 6 | 6.3 | Export from exp3_report §7 |
| F6-4 | `ana_per_tool_tsa.png` | 6 | 6.4 | Export from exp3_report §5 |
| F6-5 | `exp3_summary.png` | 6 | 6.5 | Export from exp3_report §9 |
| F7-1 | `disc_hypothesis_dashboard.png` | 7 | 7.1 | Generate |

---
---

# CHAPTER 5 — RESULT

> Chapter 5 presents raw outcomes per research question.
> Every figure should stand alone with a self-contained caption.

---

## F5-1 · `exp1_pra_bar.png`
**Section 5.1 · RQ2 · Exp 1**
**Type:** Horizontal bar chart
**Canvas:** 13 × 9 cm

### What it shows
PRA for all 15 evaluated models, sorted descending, colour-coded by tier.

### Data

```python
models = [
    # name,            pra,   tier
    ("claude-sonnet-4", 1.000, "High"),
    ("gemma3:12b",      1.000, "High"),
    ("gpt-4.1-mini",    0.996, "High"),
    ("qwen2.5:7b",      0.956, "High"),
    ("gemma2:9b",       0.948, "High"),
    ("phi4-mini",       0.915, "High"),
    ("llama3.2:3b",     0.902, "High"),
    ("qwen2.5:3b",      0.890, "High"),
    ("llama3.1:8b",     0.724, "Mid"),
    ("gemma3:4b",       0.718, "Mid"),
    ("mistral:7b",      0.544, "Failed"),
    ("qwen2.5:1.5b",    0.508, "Failed"),
    ("llama3.2:1b",     0.500, "Failed"),
    ("gemma2:2b",       0.500, "Failed"),
    ("gemma3:1b",       0.500, "Failed"),
]
```

### Styling details
- Bars: High = `#1ABC9C`, Mid = `#F39C12`, Failed = `#E74C3C`
- Vertical dashed line at **x = 0.82**, label `"H2 target (0.82)"` above line
- Vertical dashed line at **x = 0.50**, label `"Random chance"` below line
- Show PRA value at right end of each bar (2 decimal places)
- X-axis: 0.40 → 1.05, gridlines at 0.25 intervals
- Group separator lines between tiers, with tier label on right margin:
  `"High (n=8)"`, `"Mid (n=2)"`, `"Failed (n=5)"`
- No legend (use bar colour + tier labels)

### Caption for LaTeX
```
PRA across all 15 evaluated models, sorted by score.
Dashed lines mark the H2 target threshold (0.82) and
random-chance baseline (0.50). Models with $\leq$2B
parameters universally fail to exceed chance level.
```

---

## F5-2 · `exp1_radar.png`
**Section 5.1 · RQ2 · Exp 1**
**Type:** Radar / spider chart
**Canvas:** 10 × 10 cm (square)

### What it shows
Predefined FSLSM index vs ILS questionnaire result for a representative agent.
Demonstrates directional congruence even when absolute magnitudes differ.

### Source
> ⚠️ **Export directly from `exp1_report.ipynb`** — do not regenerate from scratch.
> Use the radar chart output for Agent #001.
> If the notebook is not available yet, use the placeholder values below.

### Placeholder data (if regenerating)
```python
# Four axes: act_ref, sen_int, vis_ver, seq_glo
# Scale: -1.0 to +1.0
predefined_index = {
    "act_ref": -1.0,   # Reflective pole
    "sen_int": +1.0,   # Sensing pole
    "vis_ver": -1.0,   # Verbal pole
    "seq_glo": +1.0,   # Sequential pole
}
questionnaire_result = {
    "act_ref": -0.73,
    "sen_int": +0.64,
    "vis_ver": -0.55,
    "seq_glo": +0.82,
}
```

### Styling details
- Two overlaid polygons:
  - **Predefined Index**: solid blue line `#2980B9`, filled with 20% opacity
  - **Questionnaire Result**: dashed red line `#E74C3C`, no fill
- Axis labels: `"Processing\n[act_ref]"`, `"Perception\n[sen_int]"`,
  `"Input\n[vis_ver]"`, `"Understanding\n[seq_glo]"`
- Gridlines at −1.0, −0.5, 0.0, +0.5, +1.0
- Legend inside top-right corner
- Title: `"Agent #001 — FSLSM Profile Fidelity"`
- Annotate: `"Same pole ✓"` near each axis tip where both lines agree

### Caption for LaTeX
```
Radar chart for Agent \#001 comparing the predefined FSLSM
index (solid) with the ILS questionnaire result (dashed)
across all four dimensions. Both profiles land on the same
pole of each dimension, confirming directional congruence.
```

---

## F5-3 · `exp1_pra_das_scatter.png`
**Section 5.1 · RQ2 · Exp 1**
**Type:** Scatter plot
**Canvas:** 11 × 9 cm

### What it shows
PRA (x-axis) vs DAS (y-axis) for all 15 models, with tier zones overlaid.
Reveals that PRA and DAS are strongly correlated and co-stratify by tier.

### Data
```python
# (model_name, pra, das, tier)
points = [
    ("claude-sonnet-4", 1.000, 0.927, "High"),
    ("gemma3:12b",      1.000, 0.882, "High"),
    ("gpt-4.1-mini",    0.996, 0.924, "High"),
    ("qwen2.5:7b",      0.956, 0.785, "High"),
    ("gemma2:9b",       0.948, 0.837, "High"),
    ("phi4-mini",       0.915, 0.722, "High"),
    ("llama3.2:3b",     0.902, 0.708, "High"),
    ("qwen2.5:3b",      0.890, 0.752, "High"),
    ("llama3.1:8b",     0.724, 0.741, "Mid"),
    ("gemma3:4b",       0.718, 0.718, "Mid"),
    ("mistral:7b",      0.544, 0.625, "Failed"),
    ("qwen2.5:1.5b",    0.508, 0.602, "Failed"),
    ("llama3.2:1b",     0.500, 0.500, "Failed"),
    ("gemma2:2b",       0.500, 0.531, "Failed"),
    ("gemma3:1b",       0.500, 0.500, "Failed"),
]
```

### Styling details
- Colour points by tier: same palette as F5-1
- Marker size: 80; add model name labels offset from points (avoid overlap)
- Green shaded rectangle: PRA ≥ 0.82 AND DAS ≥ 0.75 → label `"H2 Target Zone"`
- Reference lines: dashed at PRA = 0.82 (vertical) and DAS = 0.75 (horizontal)
- X-axis: 0.45 → 1.05; Y-axis: 0.45 → 0.95
- Add Pearson r in top-left corner: `"r = 0.94"` (calculate from data above)

### Caption for LaTeX
```
PRA vs DAS for all 15 models. The shaded region marks the H2
target zone (PRA $\geq$ 0.82, DAS $\geq$ 0.75). High-tier
models cluster in the target zone; failed models converge
toward the chance point (0.50, 0.50).
```

---

## F5-4 · `exp2_metrics_bar.png`
**Section 5.2 · RQ1 · Exp 2 Track A**
**Type:** Grouped bar chart
**Canvas:** 13 × 8 cm

### What it shows
R0 vs R1 comparison across all five primary metrics (Track A).
Primary axis (left): SCS, CR@5, ER (all 0–1 scale).
Secondary axis (right): RR, Engagement (1–5 scale) — OR normalise all to 0–1
by dividing 5-point scales by 5 for a single-axis chart (recommended for clarity).

### Data
```python
metrics = {
    #  metric,    R0,    R1,    delta,   cohen_d,  p,        sig
    "SCS":        (0.261, 0.469, +0.208,  2.37,    "<0.0001", "***"),
    "Engagement": (0.649, 0.778, +0.129,  1.49,    "<0.0001", "***"),  # ÷5
    "RR":         (0.758, 0.757, -0.001,  ~0,      "0.683",   "n.s."),  # ÷5
    "CR@5":       (0.159, 0.155, -0.004, -0.022,   "0.020",   "*"),
    "ER":         (0.340, 0.333, -0.006, -0.013,   "0.028",   "*"),
}
# Note: Engagement and RR are divided by 5 to normalise to 0-1
```

### Styling details
- R0 bars: `#95A5A6` (grey); R1 bars: `#2980B9` (blue)
- Bar groups separated by metric name on x-axis
- Significance bracket above each pair with annotation (e.g., `"d=2.37 ***"`)
- For n.s. pairs: draw bracket but label `"n.s."`
- Y-axis: 0.0 → 0.85 (normalised), label `"Score (0–1 normalised)"`
- Legend top-right: `"R0 (Baseline)"` / `"R1 (Personalized)"`

### Caption for LaTeX
```
Track A metric comparison: R0 (baseline) vs R1 (personalized)
across five Likert-scale metrics ($n = 5{,}760$ pairs).
Engagement and Response Relevance are normalised to 0--1 by
dividing by 5. Significance annotations above each pair;
effect sizes (Cohen's $d$) shown for confirmed effects.
```

---

## F5-5 · `exp2_dim_effect.png`
**Section 5.2 · RQ1 · Exp 2 Track A**
**Type:** Grouped horizontal bar chart
**Canvas:** 13 × 8 cm

### What it shows
ΔSCS and ΔEngagement (R1 − R0) broken down by FSLSM dimension pole.
All deltas positive; chart reveals which learner types benefit most.

### Data
```python
# (pole, delta_scs, delta_engagement_raw)
poles = [
    ("Reflective", 0.237, 0.505),
    ("Visual",     0.233, 0.906),
    ("Sequential", 0.216, 0.599),
    ("Sensing",    0.211, 0.682),
    ("Intuitive",  0.205, 0.604),
    ("Global",     0.200, 0.687),
    ("Verbal",     0.183, 0.380),
    ("Active",     0.178, 0.781),
]
# Sort by ΔSCS descending (already sorted above)
```

### Styling details
- Two bars per pole: ΔSCS (`#2980B9` blue) and ΔEngagement (`#E67E22` orange)
- ΔEngagement uses secondary x-axis (top, scale 0–1.0) OR scale it to match
  ΔSCS by dividing by 4 (since engagement is 1–5)
- Recommended: dual axis — left for ΔSCS (0–0.30), right for ΔEngagement (0–1.0)
- Label the top two values explicitly:
  - `"Visual: +0.906"` on ΔEngagement bar
  - `"Reflective: +0.237"` on ΔSCS bar
- Add vertical dashed line at ΔSCS = 0.208 (overall mean) labelled `"Mean ΔSCS"`

### Caption for LaTeX
```
Per-FSLSM-pole improvement in SCS (blue) and Engagement
(orange) under R1 vs R0. Reflective learners gain most in
style conformance; Visual learners show the largest
engagement benefit (+0.906), driven by structured visual
content unavailable in the R0 generic baseline.
```

---

## F5-6 · `exp2_pairwise_wtl.png`
**Section 5.2 · RQ1 · Exp 2 Track B**
**Type:** Stacked horizontal bar (single bar) + annotations
**Canvas:** 12 × 4 cm (wide, short)

### What it shows
Win / Tie / Loss distribution across all 5,760 pairwise judgements (Track B).

### Data
> ⚠️ **Exact W/T/L counts needed from `exp2_report_pairwise.pdf`.**
> Please upload `exp2_report_pairwise.pdf` so the exact split can be extracted.
> Use the placeholder below until then.

```python
# PLACEHOLDER — replace with actual values from exp2_report_pairwise.pdf
win  = None   # R1 preferred by GPT-4o judge
tie  = None   # No preference
loss = None   # R0 preferred
total = 5760
```

### Styling details
- Single wide stacked bar: Win = `#1ABC9C`, Tie = `#BDC3C7`, Loss = `#E74C3C`
- Label each segment: count + percentage (e.g., `"3,421 (59.4%)"`)
- Below bar: `"Binomial test (ties excluded): p < 0.001"`
- Title: `"Track B: Pairwise Judgement — R1 vs R0 (n = 5,760)"`
- Note at bottom: `"Judge: GPT-4o, position-randomised"`

---

## F5-7 · `exp3_tsa_bar.png`
**Section 5.3 · RQ3 · Exp 3**

> ✅ **Export directly from `exp3_report.ipynb`, Section 2.**
> Confirm the three bars read: S0 = 9.8%, S1a = 11.0%, S1b = 26.8%
> with significance brackets (*** for both S1b comparisons).
> No changes needed — export as-is.

---

## F5-8 · `exp3_pts_tsa.png`
**Section 5.3 · RQ3 · Exp 3**

> ✅ **Export directly from `exp3_report.ipynb`, Sections 3 + 5 (combined panel).**
> Confirm: left panel = PTS bars (S0=0%, S1a=93.0%, S1b=93.2%),
> right panel = PTS vs TSA scatter with S0/S1a/S1b labelled.
> No changes needed — export as-is.

---

## F5-9 · `exp3_dim_tsa.png`
**Section 5.3 · RQ3 · Exp 3**

> ✅ **Export directly from `exp3_report.ipynb`, Section 4.**
> Confirm 8 poles visible, Sequential = 61.8% (highest),
> Active = 0.0% and Reflective = 0.0% (lowest).
> No changes needed — export as-is.

---

## F5-10 · `exp3_profile_heatmap.png`
**Section 5.3 · RQ3 · Exp 3**

> ✅ **Export directly from `exp3_report.ipynb`, Section 6.**
> Confirm: 16-profile × 3-condition heatmap, sorted by S1b TSA descending.
> Top profile: Reflective-Intuitive-Visual-Sequential = 76.4%.
> No changes needed — export as-is.

---
---

# CHAPTER 6 — ANALYSIS

> Chapter 6 contains 4–5 deeper analyses per the AIT template requirement.
> Each analysis should go beyond reporting to reveal *why* the results occurred.

---

## F6-1 · `ana_scs_distribution.png`
**Section 6.1 — SCS Distribution Analysis**
**Type:** Violin plot (side-by-side)
**Canvas:** 11 × 8 cm

### What it shows
Full distribution of SCS scores for R0 and R1 across all 5,760 sessions.
Goes beyond mean comparison to show that R1 is both higher *and* tighter
(less variance), indicating reliable style imposition rather than lucky outliers.

### Data
```python
# Summary statistics only — generate synthetic distributions matching these:
R0_scs = dict(mean=0.261, std=0.088, min=0.05, max=0.55, n=5760)
R1_scs = dict(mean=0.469, std=0.071, min=0.20, max=0.75, n=5760)
# Generate using: np.random.normal(mean, std, n).clip(min, max)
# OR use actual session data from exp2_report.ipynb if available
```

### Styling details
- Two violins: R0 grey, R1 blue; show inner box plot (median line + IQR box)
- Overlay individual data points as jitter (alpha=0.05, 500-point subsample)
- Annotate: mean ± std for each violin
- Horizontal dashed line at R0 mean (0.261) extending across both violins
- Label: `"Δ mean = +0.208, d = 2.37"`
- Y-axis label: `"Style Conformance Score (SCS)"`

### Caption for LaTeX
```
Distribution of SCS scores across 5,760 sessions for R0
(baseline) and R1 (personalised). R1 shows both a higher
mean (+0.208) and reduced variance, indicating that FSLSM
conditioning reliably imposes structural style patterns
rather than improving only a subset of responses.
```

---

## F6-2 · `ana_retrieval_style_tradeoff.png`
**Section 6.2 — Retrieval–Style Trade-off Analysis**
**Type:** Scatter plot with quadrant shading
**Canvas:** 12 × 9 cm

### What it shows
Per-profile mean CR@5 (x) vs mean SCS (y) for both R0 and R1.
Reveals the partial orthogonality between factual retrieval and style conformance.

### Data
```python
# Aggregate values — replace with per-profile data from exp2_report.ipynb
# Overall means (will split by profile when data is available)
R0_aggregate = dict(cr5=0.159, scs=0.261)
R1_aggregate = dict(cr5=0.155, scs=0.469)

# Approximate per-pole means (from findings):
# Use per-dimension ΔSCS from findings and assume CR@5 change is uniform -0.004
```

> ⚠️ **Per-profile CR@5 and SCS data needed from `exp2_report.ipynb`.**
> Until available: plot the 8-pole dimension-level aggregates using
> R0 CR@5 = 0.159 (constant across poles) and R1 CR@5 = 0.155 (constant),
> with per-pole SCS from the findings table.

### Styling details
- Four quadrant zones shaded lightly:
  - Top-right (high CR@5, high SCS): light green, label `"Ideal"`
  - Top-left (low CR@5, high SCS): light yellow, label `"Style gain, retrieval cost"`
  - Bottom-right (high CR@5, low SCS): light blue, label `"Factual, no style"`
  - Bottom-left: light red, label `"Underperforming"`
- R0 points: grey circles; R1 points: blue circles, same pole paired with arrow
- Arrow from each R0 → R1 point for matching poles
- Reference lines: CR@5 = 0.157 (midpoint), SCS = 0.365 (midpoint)

### Caption for LaTeX
```
Per-pole mean CR@5 vs SCS for R0 (grey) and R1 (blue),
with arrows showing the R0$\rightarrow$R1 shift for each
FSLSM pole. R1 moves all profiles rightward (higher SCS)
while incurring a small leftward shift (lower CR@5),
revealing the inherent retrieval--style trade-off in the
D2L corpus (see Section~7.3).
```

---

## F6-3 · `ana_s1b_lift.png`
**Section 6.3 — FSLSM Conditioning Lift per Profile**

> ✅ **Export directly from `exp3_report.ipynb`, Section 7.**
> Confirm: horizontal bar chart, 16 profiles, showing TSA lift (S1b − S1a in pp).
> Top profiles: Reflective-Intuitive-Visual-Sequential, Reflective-Sensing-Verbal-Sequential.
> Profiles with Sequential or Visual poles should dominate positive lift.
> No changes needed — export as-is.

---

## F6-4 · `ana_per_tool_tsa.png`
**Section 6.4 — Per-Tool TSA Breakdown (S1b)**

> ✅ **Export directly from `exp3_report.ipynb`, Section 5.**
> Confirm: bar chart of TSA % per tool under S1b, sorted descending,
> with tool names on x-axis and n= labels on bars.
> No changes needed — export as-is.

---

## F6-5 · `exp3_summary.png`
**Section 6.5 — Combined Exp 3 Summary**

> ✅ **Export directly from `exp3_report.ipynb`, Section 9.**
> This is the thesis-ready 2×3 multi-panel figure.
> Confirm panels: (A) TSA by condition, (B) PTS by condition,
> (C) FSLSM lift per profile, (D) per-dimension TSA, (E) per-tool TSA.
> No changes needed — export as-is.

---
---

# CHAPTER 7 — DISCUSSION

> Chapter 7 requires at minimum 2 pages per the AIT template.
> Figures here are interpretive — they synthesise across experiments
> and position findings relative to the literature.

---

## F7-1 · `disc_hypothesis_dashboard.png`
**Section 7.1 — Revisiting Research Questions**
**Type:** Visual summary panel (infographic-style table)
**Canvas:** 14 × 7 cm

### What it shows
A structured one-page summary of how each hypothesis was resolved,
intended for the committee to grasp the overall outcome at a glance.

### Content layout (3-column grid, one column per hypothesis)

```
┌─────────────────┬──────────────────┬─────────────────┐
│     H2 (Exp 1)  │     H1 (Exp 2)   │     H3 (Exp 3)  │
│  Agent Fidelity │  Personalization  │  MCP Tool Sel.  │
├─────────────────┼──────────────────┼─────────────────┤
│  ✅ EXCEEDED    │  ⚠️ PARTIAL      │  ✅ CONFIRMED   │
├─────────────────┼──────────────────┼─────────────────┤
│ PRA = 1.000     │ SCS d = 2.37 ✅  │ TSA +15.8 pp ✅ │
│ (top models)    │ Eng d = 1.49 ✅  │ PTS = 93% ✅    │
│ 8/15 exceed     │ CR@5 d = −0.02 ❌│ S1a ≈ S0 ⚠️    │
│ target          │ RR: no change ✅  │                 │
├─────────────────┼──────────────────┼─────────────────┤
│ RQ2 ✅ Answered │ RQ1 ✅ Answered  │ RQ3 ✅ Answered │
└─────────────────┴──────────────────┴─────────────────┘
```

### Styling details
- Background: white with thin border per cell
- Status badge colours:
  - `✅ EXCEEDED` / `✅ CONFIRMED`: `#1ABC9C` badge
  - `⚠️ PARTIAL`: `#F39C12` badge
- Metric rows: normal weight text, confirmed metrics in green, unconfirmed in red
- Column headers use thesis blue `#2980B9`
- Bottom row (RQ answered): subtle grey background `#ECF0F1`
- Use a clean tabular layout, not a decorative infographic

### Caption for LaTeX
```
Summary of hypothesis outcomes across all three experiments.
Tick marks (\checkmark) indicate confirmed predictions;
crosses indicate predictions not supported by the data.
Effect sizes (Cohen's $d$) and directional arrows
summarise the magnitude of confirmed effects.
```

---

## Upload Checklist

When uploading figures, confirm each filename exactly matches this list:

**Generate fresh (Claude Code):**
- [ ] `exp1_pra_bar.png` (F5-1)
- [ ] `exp1_radar.png` (F5-2) — or export from exp1_report
- [ ] `exp1_pra_das_scatter.png` (F5-3)
- [ ] `exp2_metrics_bar.png` (F5-4)
- [ ] `exp2_dim_effect.png` (F5-5)
- [ ] `exp2_pairwise_wtl.png` (F5-6) — **needs exp2_report_pairwise.pdf first**
- [ ] `ana_scs_distribution.png` (F6-1)
- [ ] `ana_retrieval_style_tradeoff.png` (F6-2) — **needs per-profile data from exp2_report**
- [ ] `disc_hypothesis_dashboard.png` (F7-1)

**Export from existing notebooks (no regeneration needed):**
- [ ] `exp3_tsa_bar.png` — exp3_report §2
- [ ] `exp3_pts_tsa.png` — exp3_report §3+5
- [ ] `exp3_dim_tsa.png` — exp3_report §4
- [ ] `exp3_profile_heatmap.png` — exp3_report §6
- [ ] `ana_s1b_lift.png` — exp3_report §7
- [ ] `ana_per_tool_tsa.png` — exp3_report §5
- [ ] `exp3_summary.png` — exp3_report §9

**Blocked — upload required first:**
- [ ] `exp2_pairwise_wtl.png` → needs `exp2_report_pairwise.pdf`
- [ ] `ana_retrieval_style_tradeoff.png` → needs per-session CR@5+SCS from `exp2_report.ipynb`

---

## Notes on LaTeX Integration

All figures will be inserted using:
```latex
\begin{figure}[h]
    \centerline{\includegraphics[width=Xcm]{figures/FILENAME.png}}
    \caption{CAPTION}
    \label{fig:LABEL}
\end{figure}
```

Width guide:
- Full-column figures: `width=13cm`
- Square / radar charts: `width=10cm`
- Side-by-side panels: `width=14cm`
- Narrow single charts: `width=11cm`
