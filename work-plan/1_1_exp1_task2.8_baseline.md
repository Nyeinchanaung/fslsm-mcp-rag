# Task 2.8 — Non-Personalized Baseline (Addendum to Phase 2 v2)

> **Purpose:** Establish a lower-bound PRA measurement that proves FSLSM persona encoding
> is actually responsible for the agents' learning-style alignment. Without this baseline,
> a reviewer can argue that LLMs naturally exhibit style-consistent behavior regardless
> of prompting. Table 3.2 in the thesis projects Baseline PRA ≈ 0.60 ± 0.13.

---

## 2.8.1 — Baseline Design Rationale

The baseline uses a **single generic "student" profile** with no FSLSM dimensions.
Five instances carry knowledge-level variations (matching the FSLSM agents), run
across all 3 LLM providers:

```
Baseline agents: 5 instances × 3 models = 15 agent records

Instance 1 → beginner
Instance 2 → intermediate
Instance 3 → advanced
Instance 4 → general (no knowledge level)
Instance 5 → general (no knowledge level)
```

**Why 5 agents instead of 80?** The baseline has no FSLSM dimensions to vary — there
is only one "non-personalized" profile. Running 5 instances gives enough statistical
signal for mean ± std across knowledge levels, and repeating across 3 trials (like
the FSLSM agents) yields 15 ILS runs per model (5 agents × 3 trials = 15 data points).

**How PRA is computed for baseline agents:** Baseline agents have no assigned FSLSM
profile. After running ILS, we detect each agent's dimension poles from its ILS
scores. Then we compute:
1. **PRA vs. each of the 16 profiles** — for each baseline agent trial, count how many
   of the 4 detected dimensions match each of the 16 profiles. Average across all
   16 gives the "chance-level" PRA.
2. **PRA vs. best-matching profile** — find which of the 16 profiles the baseline agent
   most closely matches. This reveals the LLM's natural style bias.
3. **Dimension-level analysis** — report the raw ILS scores per dimension to show
   which poles the LLM defaults to without prompting.

The expected result: average PRA across all 16 profiles ≈ 0.50–0.60 (near random
for a 4-dimension binary match). The key finding is that FSLSM agents score
0.75–0.82 against their *assigned* profile, which is significantly higher than the
baseline's match against *any* profile.

---

## 2.8.2 — DB Setup: Baseline Profile

Add one special profile to `fslsm_profiles` with neutral (zero) dimension values.
This profile exists purely as a foreign key anchor — its dimension values are only
used for the "best-match" PRA variant.

**Option A: Add to `data/fslsm/profiles.json`:**

```json
{
  "profile_code": "P00_Baseline",
  "label": "Non-Personalized Baseline",
  "dimensions": {"act_ref": 0, "sen_int": 0, "vis_ver": 0, "seq_glo": 0},
  "behavioral_instructions": {
    "act_ref": "",
    "sen_int": "",
    "vis_ver": "",
    "seq_glo": ""
  },
  "style_descriptor_graf": "No learning style preference assigned."
}
```

**Option B: Insert directly via seed script** (if you don't want to touch the 16-profile JSON):

```python
# In scripts/create_profiles.py — add after seeding the 16 FSLSM profiles:
session.add(FslsmProfile(
    profile_code="P00_Baseline",
    act_ref=0, sen_int=0, vis_ver=0, seq_glo=0,
    description="Non-Personalized Baseline",
    style_descriptor="No learning style preference assigned.",
))
```

> **Note on `CHECK` constraint:** The existing `fslsm_profiles` schema has
> `CHECK (act_ref IN (-1, 1))`. You need to relax this to `CHECK (act_ref IN (-1, 0, 1))`
> for the baseline row, or use a migration:
>
> ```sql
> ALTER TABLE fslsm_profiles DROP CONSTRAINT IF EXISTS fslsm_profiles_act_ref_check;
> ALTER TABLE fslsm_profiles ADD CONSTRAINT fslsm_profiles_act_ref_check
>     CHECK (act_ref IN (-1, 0, 1));
> -- Repeat for sen_int, vis_ver, seq_glo
> ```

**Update `config/constants.py`:**

```python
# Add after existing constants
BASELINE_PROFILE_CODE = "P00_Baseline"
NUM_BASELINE_INSTANCES = 5
NUM_BASELINE_AGENTS_PER_MODEL = 5  # 5 instances × 1 profile
```

**Done:** `SELECT * FROM fslsm_profiles WHERE profile_code = 'P00_Baseline';` returns one row with all dimensions = 0.

---

## 2.8.3 — Baseline System Prompt

**`src/agents/prompts/student_system.py`** — add a baseline prompt builder:

```python
def build_baseline_system_prompt(knowledge_level: str | None = None) -> str:
    """
    System prompt for Non-Personalized Baseline agents.
    No FSLSM dimensions, no behavioral instructions.
    Optionally includes knowledge_level.
    """
    if knowledge_level and knowledge_level in KNOWLEDGE_LEVEL_INSTRUCTIONS:
        knowledge_block = f"""

Your Knowledge Level: {knowledge_level.capitalize()}
{KNOWLEDGE_LEVEL_INSTRUCTIONS[knowledge_level]}"""
    else:
        knowledge_block = ""

    return f"""You are a virtual undergraduate student studying Introductory Machine Learning.
You are participating in a learning simulation. Respond naturally based on your own preferences — you have no specific learning style assignment.
{knowledge_block}

Interaction Rules:
1. Always respond AS the student, not as a tutor or assistant.
2. Answer questions based on your own natural inclinations and preferences.
3. When receiving explanations, react authentically based on what feels natural to you.
4. Maintain consistency across all turns in the conversation.

Topic domain: Introductory Machine Learning (neural networks, optimization, gradient descent).
"""
```

**Key design principle:** The prompt is deliberately neutral. It says "respond naturally" and
"your own preferences" — it does NOT say "be balanced" or "have no preference", which
would itself be a form of steering. The agent should reveal the LLM's natural default
tendencies.

**Sanity check:**

```python
from src.agents.prompts.student_system import build_baseline_system_prompt

prompt = build_baseline_system_prompt(knowledge_level="beginner")
assert "FSLSM" not in prompt
assert "learning style" not in prompt.lower().split("assignment")[0]  # no style before "assignment"
assert "Knowledge Level: Beginner" in prompt
print("Baseline prompt OK — no FSLSM leakage")
```

---

## 2.8.4 — Baseline Agent Factory

**`src/agents/agent_factory.py`** — add a `create_baseline_agents()` function:

```python
from config.constants import KNOWLEDGE_LEVEL_MAP, BASELINE_PROFILE_CODE
from src.agents.prompts.student_system import build_baseline_system_prompt

def create_baseline_agents(llm_model: str):
    """
    Create 5 Non-Personalized Baseline agents for a given model.
    Same knowledge_level mapping as FSLSM agents (beg, int, adv, gen, gen).
    Profile FK points to P00_Baseline (all dimensions = 0).
    """
    with get_session() as session:
        baseline_profile = (
            session.query(FslsmProfile)
            .filter_by(profile_code=BASELINE_PROFILE_CODE)
            .one()
        )

        for instance in range(1, 6):
            knowledge_level = KNOWLEDGE_LEVEL_MAP[instance]

            if knowledge_level:
                agent_uid = f"agent_Baseline_I{instance:02d}_{knowledge_level[:3]}"
            else:
                agent_uid = f"agent_Baseline_I{instance:02d}_gen"

            system_prompt = build_baseline_system_prompt(
                knowledge_level=knowledge_level,
            )

            session.add(Agent(
                agent_uid=agent_uid,
                profile_id=baseline_profile.id,
                instance_num=instance,
                llm_model=llm_model,
                system_prompt=system_prompt,
                knowledge_level=knowledge_level,
            ))

        session.commit()

    print(f"Created 5 baseline agents for {llm_model}")
```

**Done:** `SELECT agent_uid FROM agents WHERE agent_uid LIKE 'agent_Baseline%' AND llm_model = 'gpt-4o';` returns 5 rows.

---

## 2.8.5 — Baseline ILS Evaluator

The existing `run_ils_for_agent()` works **unchanged** — it just uses the agent's
system prompt (which happens to be the baseline prompt). The only new piece is a
dedicated runner that creates baseline agents and processes them:

**`src/agents/ils_evaluator.py`** — add:

```python
def run_baseline_experiment(llm_model: str, num_trials: int = 3, temperature: float = 0.3):
    """
    Run ILS questionnaire on 5 baseline agents for a given model.
    Returns results with detected poles for downstream PRA-vs-all-profiles analysis.
    """
    questions = json.load(open("data/fslsm/ils_questionnaire.json"))
    client = LLMClient(llm_model, temperature=temperature)

    with get_session() as session:
        agents = (
            session.query(Agent)
            .filter(
                Agent.llm_model == llm_model,
                Agent.agent_uid.like("agent_Baseline%"),
            )
            .all()
        )
        agent_data = [
            {"agent": a, "knowledge_level": a.knowledge_level}
            for a in agents
        ]

    results = []
    cumulative_cost = 0.0

    for ad in tqdm(agent_data, desc=f"Baseline [{llm_model}]"):
        agent = ad["agent"]
        for trial in range(1, num_trials + 1):
            dim_scores, call_cost = run_ils_for_agent(agent, questions, client, trial)
            cumulative_cost += call_cost

            detected = {
                d: (1 if dim_scores[d] > 0 else (-1 if dim_scores[d] < 0 else 0))
                for d in FSLSM_DIMENSIONS
            }

            results.append({
                "agent_uid": agent.agent_uid,
                "knowledge_level": ad["knowledge_level"],
                "profile_code": "Baseline",
                "instance_num": agent.instance_num,
                "trial": trial,
                "detected": detected,
                "raw_scores": dim_scores,
                "cost_usd": call_cost,
            })

    print(f"\n  Baseline cost for {llm_model}: ${cumulative_cost:.4f}")
    return results
```

---

## 2.8.6 — Baseline PRA Computation

The baseline PRA requires a different calculation than the FSLSM agents. Since
baseline agents have no assigned profile, we compute PRA **against all 16 profiles**
and report the average.

**`src/evaluation/metrics.py`** — add:

```python
def baseline_pra_vs_all_profiles(
    baseline_results: list[dict],
    all_profiles: list[dict],
) -> dict:
    """
    Compute how well baseline agents match EACH of the 16 FSLSM profiles.

    For each baseline result, count dimension matches against each profile.
    Report: overall average PRA, per-profile PRA, best-matching profile,
    and per-dimension natural bias.

    Args:
        baseline_results: list of dicts with "detected" dimension poles
        all_profiles: list of 16 profile dicts with "dimensions" {dim: -1/+1}

    Returns:
        {
            "avg_pra_vs_all": float,         # average match rate across all 16 profiles
            "std_pra_vs_all": float,
            "per_profile_pra": {code: float}, # PRA against each specific profile
            "best_match_profile": str,        # which profile the LLM naturally resembles
            "best_match_pra": float,
            "dimension_bias": {dim: float},   # avg raw ILS score per dim (shows natural tendency)
        }
    """
    # --- PRA against each of the 16 profiles ---
    per_profile_scores = {}
    for profile in all_profiles:
        code = profile["profile_code"]
        if code == "P00_Baseline":
            continue  # skip the baseline profile itself
        assigned = profile["dimensions"]
        matches = []
        for r in baseline_results:
            detected = r["detected"]
            dim_match_count = sum(
                1 for d in FSLSM_DIMENSIONS
                if detected[d] != 0 and detected[d] == assigned[d]
            )
            # Count non-tie dimensions for normalization
            non_tie_dims = sum(1 for d in FSLSM_DIMENSIONS if detected[d] != 0)
            if non_tie_dims > 0:
                matches.append(dim_match_count / non_tie_dims)
            else:
                matches.append(0.5)  # all ties → coin flip
        per_profile_scores[code] = float(np.mean(matches))

    all_pra_values = list(per_profile_scores.values())
    best_code = max(per_profile_scores, key=per_profile_scores.get)

    # --- Natural dimension bias ---
    dim_bias = {d: [] for d in FSLSM_DIMENSIONS}
    for r in baseline_results:
        for d in FSLSM_DIMENSIONS:
            dim_bias[d].append(r["raw_scores"][d])

    return {
        "avg_pra_vs_all": float(np.mean(all_pra_values)),
        "std_pra_vs_all": float(np.std(all_pra_values)),
        "per_profile_pra": per_profile_scores,
        "best_match_profile": best_code,
        "best_match_pra": per_profile_scores[best_code],
        "dimension_bias": {
            d: {
                "mean_score": float(np.mean(dim_bias[d])),
                "detected_pole": int(np.sign(np.mean(dim_bias[d]))) or 0,
                "pole_label": (
                    FSLSM_DIM_LABELS[d][0] if np.mean(dim_bias[d]) < 0
                    else FSLSM_DIM_LABELS[d][1] if np.mean(dim_bias[d]) > 0
                    else "Neutral"
                ),
            }
            for d in FSLSM_DIMENSIONS
        },
    }
```

**What this tells you for the thesis:**
- `avg_pra_vs_all ≈ 0.50–0.60` → LLM's natural responses match any random profile about half the time (expected baseline)
- `best_match_profile` → reveals the LLM's inherent style bias (e.g., GPT-4o might naturally lean Reflective-Verbal)
- `dimension_bias` → raw numbers showing which pole each model defaults to

---

## 2.8.7 — Updated Experiment Runner

**`experiments/exp1_agent_fidelity/run.py`** — add baseline run before or after FSLSM runs:

```python
import yaml, json
from pathlib import Path
from src.agents.agent_factory import create_agents, create_baseline_agents
from src.agents.ils_evaluator import run_experiment1, run_baseline_experiment

CONFIG = yaml.safe_load(open("experiments/exp1_agent_fidelity/config.yaml"))

for model_cfg in CONFIG["models"]:
    model_name = model_cfg["name"]
    temperature = model_cfg.get("temperature", 0.3)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    # ── FSLSM agents (80 per model) ──
    print(f"\n--- Creating FSLSM agents ---")
    create_agents(model_name)

    print(f"\n--- Running FSLSM ILS questionnaire ---")
    fslsm_results = run_experiment1(
        llm_model=model_name,
        num_trials=CONFIG["ils_questionnaire"]["num_trials"],
        temperature=temperature,
    )
    safe = model_name.replace("/", "_").replace(":", "_")
    Path(f"results/exp1/metrics/{safe}_results.json").write_text(
        json.dumps(fslsm_results, indent=2)
    )

    # ── Baseline agents (5 per model) ──
    print(f"\n--- Creating Baseline agents ---")
    create_baseline_agents(model_name)

    print(f"\n--- Running Baseline ILS questionnaire ---")
    baseline_results = run_baseline_experiment(
        llm_model=model_name,
        num_trials=CONFIG["ils_questionnaire"]["num_trials"],
        temperature=temperature,
    )
    Path(f"results/exp1/metrics/{safe}_baseline_results.json").write_text(
        json.dumps(baseline_results, indent=2)
    )

    # ── Quick summary ──
    from src.evaluation.metrics import profile_recovery_accuracy
    pra = profile_recovery_accuracy(fslsm_results)
    print(f"\nFSLSM PRA (overall 4D): {pra['overall_4d']:.3f}")
    print(f"Baseline: {len(baseline_results)} records saved for analysis")
```

**Updated `config.yaml`** — add baseline section:

```yaml
# ... (existing config unchanged) ...

baseline:
  num_agents: 5
  profile_code: "P00_Baseline"
  description: >
    Non-personalized agents with no FSLSM profile.
    Same knowledge_level distribution as FSLSM agents.
    Used to establish lower-bound PRA for Table 3.2.
```

---

## 2.8.8 — Updated Analyze Script

**`experiments/exp1_agent_fidelity/analyze.py`** — add baseline analysis:

```python
import json, pandas as pd
from pathlib import Path
from src.evaluation.metrics import (
    profile_recovery_accuracy, pra_by_knowledge_level,
    cost_summary, baseline_pra_vs_all_profiles,
)
from config.constants import FSLSM_DIMENSIONS

MODELS = ["gpt-4o", "claude-sonnet-4-5-20251001", "llama3.1_8b"]

def run_analysis():
    all_rows = []
    cost_rows = []
    baseline_rows = []

    profiles = json.loads(Path("data/fslsm/profiles.json").read_text())

    for model in MODELS:
        safe = model.replace("/", "_").replace(":", "_")

        # ── FSLSM analysis (unchanged) ──
        fslsm_file = Path(f"results/exp1/metrics/{safe}_results.json")
        if fslsm_file.exists():
            results = json.loads(fslsm_file.read_text())
            pra = profile_recovery_accuracy(results)
            # ... (existing PRA export logic — unchanged from v2) ...

            for dim in FSLSM_DIMENSIONS:
                all_rows.append({
                    "model": model, "condition": "FSLSM",
                    "knowledge_level": "ALL", "dimension": dim,
                    "pra": pra["per_dimension"][dim],
                })
            all_rows.append({
                "model": model, "condition": "FSLSM",
                "knowledge_level": "ALL", "dimension": "overall_4d",
                "pra": pra["overall_4d"],
            })

            # PRA by knowledge level (FSLSM)
            for level, lpra in pra_by_knowledge_level(results).items():
                all_rows.append({
                    "model": model, "condition": "FSLSM",
                    "knowledge_level": level, "dimension": "overall_4d",
                    "pra": lpra["overall_4d"],
                })

            cost_rows.append({"model": model, "condition": "FSLSM",
                              **cost_summary(results)})

        # ── Baseline analysis ──
        baseline_file = Path(f"results/exp1/metrics/{safe}_baseline_results.json")
        if baseline_file.exists():
            bl_results = json.loads(baseline_file.read_text())
            bl_pra = baseline_pra_vs_all_profiles(bl_results, profiles)

            # Overall baseline PRA (average match vs all 16 profiles)
            all_rows.append({
                "model": model, "condition": "Baseline",
                "knowledge_level": "ALL", "dimension": "overall_4d",
                "pra": bl_pra["avg_pra_vs_all"],
            })

            # Baseline by knowledge level
            for level in ["beginner", "intermediate", "advanced", "general"]:
                level_results = [
                    r for r in bl_results
                    if (r.get("knowledge_level") or "general") == level
                ]
                if level_results:
                    level_bl_pra = baseline_pra_vs_all_profiles(level_results, profiles)
                    all_rows.append({
                        "model": model, "condition": "Baseline",
                        "knowledge_level": level, "dimension": "overall_4d",
                        "pra": level_bl_pra["avg_pra_vs_all"],
                    })

            # Detailed baseline report
            baseline_rows.append({
                "model": model,
                "avg_pra_vs_all": bl_pra["avg_pra_vs_all"],
                "std_pra_vs_all": bl_pra["std_pra_vs_all"],
                "best_match_profile": bl_pra["best_match_profile"],
                "best_match_pra": bl_pra["best_match_pra"],
                **{f"bias_{d}": bl_pra["dimension_bias"][d]["pole_label"]
                   for d in FSLSM_DIMENSIONS},
                **{f"bias_{d}_score": bl_pra["dimension_bias"][d]["mean_score"]
                   for d in FSLSM_DIMENSIONS},
            })

            cost_rows.append({"model": model, "condition": "Baseline",
                              **cost_summary(bl_results)})

    # ── Export everything ──
    df_pra = pd.DataFrame(all_rows)
    df_pra.to_csv("results/exp1/metrics/pra_das_summary.csv", index=False)

    df_baseline = pd.DataFrame(baseline_rows)
    df_baseline.to_csv("results/exp1/metrics/baseline_analysis.csv", index=False)

    df_cost = pd.DataFrame(cost_rows)
    df_cost.to_csv("results/exp1/metrics/cost_summary.csv", index=False)

    print("\n=== FSLSM vs Baseline PRA ===")
    print(df_pra[df_pra["dimension"] == "overall_4d"].to_string(index=False))
    print("\n=== Baseline Natural Bias ===")
    print(df_baseline.to_string(index=False))

    return df_pra, df_baseline, df_cost


if __name__ == "__main__":
    run_analysis()
```

**Output files:**
- `results/exp1/metrics/pra_das_summary.csv` — now includes `condition` column: `FSLSM` or `Baseline`
- `results/exp1/metrics/baseline_analysis.csv` — per-model natural bias report
- `results/exp1/metrics/cost_summary.csv` — cost for both conditions

---

## 2.8.9 — Baseline Visualizations

### FSLSM vs Baseline Bar Chart (the main Table 3.2 figure)

```python
def fslsm_vs_baseline_bar(df, save_path: str):
    """
    Grouped bar chart reproducing Table 3.2 from the thesis:
    Baseline PRA vs FSLSM PRA per model.

    df: pra_das_summary.csv filtered to dimension='overall_4d', knowledge_level='ALL'
    """
    models = df["model"].unique()
    conditions = ["Baseline", "FSLSM"]
    colors = {"Baseline": "#9E9E9E", "FSLSM": "#2196F3"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.3
    x = np.arange(len(models))

    for i, cond in enumerate(conditions):
        subset = df[df["condition"] == cond]
        values = [
            subset[subset["model"] == m]["pra"].values[0]
            if len(subset[subset["model"] == m]) > 0 else 0
            for m in models
        ]
        bars = ax.bar(x + i * bar_width, values, bar_width,
                       label=cond, color=colors[cond])
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model")
    ax.set_ylabel("Profile Recovery Accuracy (PRA)")
    ax.set_title("Non-Personalized Baseline vs. FSLSM-Encoded Agents")
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([m.split(":")[0] for m in models], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.50, color='red', linestyle=':', alpha=0.4, label="Random chance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

### Baseline Dimension Bias Radar Chart

```python
def baseline_bias_radar(df_baseline, save_path: str):
    """
    Radar chart showing each model's natural dimension bias (no FSLSM prompting).
    Reveals whether GPT-4o/Claude/Llama have inherent learning-style tendencies.
    """
    dims = ["act_ref", "sen_int", "vis_ver", "seq_glo"]
    labels = ["Processing\n[Act/Ref]", "Perception\n[Sen/Int]",
              "Input\n[Vis/Ver]", "Understanding\n[Seq/Glo]"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))

    for i, (_, row) in enumerate(df_baseline.iterrows()):
        values = [row[f"bias_{d}_score"] for d in dims]
        values += values[:1]
        model_label = row["model"].split(":")[0]
        ax.plot(angles, values, '-o', label=model_label, color=colors[i], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("LLM Natural Style Bias (No FSLSM Prompting)", pad=15)
    ax.set_ylim(-11, 11)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
```

**Done:** `results/exp1/figures/` now also contains:
- `fslsm_vs_baseline.png` — the key comparison figure for Table 3.2
- `baseline_bias_radar.png` — each model's natural learning-style tendency

---

## 2.8.10 — API Cost Estimate for Baseline

```
Baseline calls per model: 5 agents × 44 questions × 3 trials = 660 API calls
Tokens per call: ~50 input + 5 output = 55 tokens
Total per model: ~36K tokens

At gpt-4o-mini pricing ($0.15/1M input + $0.60/1M output):
  → ~$0.01 per model
  → ~$0.03 total across 3 models

Negligible additional cost.
```

---

## Updated Agent Count Summary

| Condition | Profiles | Instances | Agents/Model | × 3 Models | Total |
|-----------|----------|-----------|-------------|------------|-------|
| FSLSM | 16 | 5 (3 leveled + 2 general) | 80 | 240 | 240 |
| Baseline | 1 | 5 (3 leveled + 2 general) | 5 | 15 | 15 |
| **Total** | **17** | | **85** | | **255** |

ILS runs (with 3 trials): 255 agents × 3 trials = **765 ILS runs** across all models.

---

## Updated Final Checklist (with Baseline)

| Check | Command / Criterion |
|---|---|
| Baseline profile exists | `SELECT * FROM fslsm_profiles WHERE profile_code = 'P00_Baseline';` → 1 row |
| 5 baseline agents per model | `SELECT COUNT(*) FROM agents WHERE agent_uid LIKE 'agent_Baseline%' AND llm_model = 'gpt-4o';` → 5 |
| Baseline raw responses | `ls results/exp1/raw_responses/agent_Baseline*.json \| wc -l` → 15 per model (5 × 3 trials) |
| Baseline results JSON | `ls results/exp1/metrics/*_baseline_results.json` → 3 files |
| Baseline analysis CSV | `cat results/exp1/metrics/baseline_analysis.csv` |
| PRA summary has both conditions | `cut -d, -f2 results/exp1/metrics/pra_das_summary.csv \| sort -u` → `Baseline, FSLSM, condition` |
| Baseline PRA < FSLSM PRA | `avg_pra_vs_all` < `overall_4d` for each model |
| FSLSM vs Baseline figure | `ls results/exp1/figures/fslsm_vs_baseline.png` exists |
| Baseline bias radar | `ls results/exp1/figures/baseline_bias_radar.png` exists |

---

## Updated Execution Order

| Day | Tasks |
|---|---|
| Day 1 | Pre-Phase 2: LiteLLM migration, DB migration (knowledge_level + baseline profile) |
| Day 1 | Task 2.1–2.2: Prompt builders (FSLSM + baseline), agent factories |
| Day 2 | Task 2.3–2.4: ILS evaluator (handles both FSLSM and baseline) |
| Day 2 | Sanity test: 1 FSLSM agent + 1 baseline agent, 1 trial each |
| Day 3 | Task 2.5: Run GPT-4o — 80 FSLSM + 5 baseline (~2.5h) |
| Day 4 | Task 2.5: Run Claude Sonnet — 80 FSLSM + 5 baseline |
| Day 4 | Task 2.6 + 2.8: All metrics (PRA, baseline PRA, cost) |
| Day 5 | Task 2.5: Ollama / Llama 3 — 80 FSLSM + 5 baseline |
| Day 5 | Task 2.7 + 2.8.9: All visualizations (including baseline comparisons) |
| Day 6 | Buffer: prompt refinement, final analysis |

---

## What This Produces for the Thesis

The baseline run directly populates **Table 3.2** with empirical data:

| Model | PRA | DAS | Condition |
|-------|-----|-----|-----------|
| Non-Personalized Baseline (GPT-4o) | 0.5x | 0.5x | No FSLSM |
| Non-Personalized Baseline (Claude) | 0.5x | 0.5x | No FSLSM |
| Non-Personalized Baseline (Llama 3) | 0.5x | 0.5x | No FSLSM |
| FSLSM + GPT-4o | 0.8x | 0.7x | FSLSM-encoded |
| FSLSM + Claude Sonnet | 0.8x | 0.7x | FSLSM-encoded |
| FSLSM + Llama 3 8B | 0.7x | 0.7x | FSLSM-encoded |

The gap (Δ PRA ≈ 0.20–0.25) between baseline and FSLSM agents is the central
finding of Experiment 1: persona encoding works.

Additionally, the `baseline_bias_radar.png` provides a novel insight: which
learning-style dimensions each LLM naturally gravitates toward. This can be
discussed in Section 4 (Results) as a finding about inherent model biases —
e.g., "GPT-4o exhibits a natural Reflective-Verbal tendency, while Llama 3
leans Active-Visual."
