# Phase 2 (Updated v2) — Experiment 1: Virtual Student Agent Fidelity

> **What changed in v2 (this version):**
> 1. **LiteLLM replaces the custom `LLMClient`** — all LLM calls now go through `litellm.completion()`, an open-source, MIT-licensed, provider-agnostic SDK (BerriAI/litellm). This strengthens methodological defensibility (see rationale below).
> 2. **knowledge_level** integrated into agent instances (from v1): instances 1–3 = `beginner`, `intermediate`, `advanced`; instances 4–5 = `general`.
>
> **Phases 0–1 are already complete.** This document replaces Phase 2 in `1_1_exp1.md`.

**Duration estimate:** 3–5 days
**Done when:** `results/exp1/metrics/pra_das_summary.csv` exists with PRA and DAS scores sliced by model AND by knowledge_level, and radar charts exist in `results/exp1/figures/`.

---

## Why LiteLLM? (Methodological Justification)

Your thesis compares agent fidelity across three LLM providers (GPT-4o, Claude Sonnet, Llama 3 8B). A reviewer or committee member can legitimately ask: *"Are the performance differences due to the models themselves, or due to differences in how you implemented each provider's API?"* The custom `LLMClient` from Phase 0 has three separate methods (`_openai_chat`, `_anthropic_chat`, `_ollama_chat`) with subtly different parameter handling, error paths, and token counting. This is a confound.

**LiteLLM eliminates this confound** by providing a single `completion()` function that internally normalizes prompt formatting, token counting, and response parsing across all providers. Specifically:

1. **Standardized prompt delivery.** LiteLLM translates a single `messages=[...]` format into each provider's native format. Your system prompts and ILS questions reach every model through the same transformation pipeline — not through three hand-written adapters.

2. **Consistent token accounting.** LiteLLM returns `response.usage.prompt_tokens` and `response.usage.completion_tokens` in a unified schema regardless of provider. This means your token-usage comparisons across models (important for Experiment 3) are apples-to-apples.

3. **Citable and auditable.** LiteLLM is an established open-source library (MIT license, 18k+ GitHub stars, actively maintained). You can cite it in the thesis methodology section as: *"All LLM interactions were mediated through LiteLLM v1.x (BerriAI, 2024), an open-source provider-agnostic SDK, to ensure uniform prompt delivery and token accounting across providers."* This is far more defensible than citing custom wrapper code.

4. **Built-in retry and rate-limit handling.** For 10,560 API calls per model (80 agents × 44 questions × 3 trials), provider rate limits are a real concern. LiteLLM handles backoff/retry natively, eliminating the need for custom retry logic.

5. **Cost tracking.** LiteLLM logs cost per call via `response._hidden_params["response_cost"]`, which you can record alongside PRA metrics for a cost-efficiency analysis across models.

6. **Future-proofing.** If a reviewer suggests testing a fourth model (e.g., Gemini, Mistral), you add one line to `config.yaml` — no new `_gemini_chat()` method needed.

**In the thesis, this appears in Section 3.3 (System Architecture):** add a sentence to the Data and Content Layer noting that all LLM interactions are routed through a provider-agnostic abstraction layer to ensure fair cross-model comparison.

---

## Pre-Phase 2 — Migration Steps

### Pre-2.A — Install LiteLLM + Remove Redundant SDK Deps

```bash
source venv/bin/activate
pip install litellm
```

LiteLLM bundles `openai` and `httpx` internally. You can keep `anthropic` installed (LiteLLM uses it under the hood for Anthropic calls), but you no longer need to import it directly.

**Update `pyproject.toml`** — add `litellm` to dependencies, mark direct provider SDKs as optional:

```toml
dependencies = [
    # --- LLM (unified interface) ---
    "litellm>=1.40.0",

    # --- Provider SDKs (pulled by litellm, listed for clarity) ---
    "openai>=1.40.0",
    "anthropic>=0.34.0",

    # ... (rest unchanged)
]
```

**Done:** `python -c "import litellm; print(litellm.__version__)"` prints a version ≥ 1.40.

---

### Pre-2.B — Replace LLMClient with LiteLLM Wrapper

**`src/utils/llm_client.py`** — rewrite to wrap `litellm.completion()`:

```python
"""
Unified LLM client backed by LiteLLM.

All provider-specific formatting (OpenAI, Anthropic, Ollama) is handled
internally by LiteLLM, ensuring identical prompt delivery across models.
This design choice is documented in the thesis methodology to eliminate
provider-adapter confounds in cross-model comparisons.

Reference: https://github.com/BerriAI/litellm
"""

from dataclasses import dataclass, field
from typing import Optional
import litellm

# Suppress LiteLLM's verbose startup logs; keep warnings/errors
litellm.suppress_debug_info = True


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost: float = 0.0               # USD cost of this call (from LiteLLM)
    raw_response: Optional[dict] = field(default=None, repr=False)


# ── Model string mapping ──────────────────────────────────────────
# LiteLLM requires provider-prefixed model names.
# This dict maps the short names used in config.yaml to LiteLLM model strings.
MODEL_REGISTRY = {
    # API models
    "gpt-4o":                       "openai/gpt-4o",
    "gpt-4o-mini":                  "openai/gpt-4o-mini",
    "claude-sonnet-4-5-20251001":   "anthropic/claude-sonnet-4-5-20251001",
    # Local via Ollama
    "llama3.1:8b":                  "ollama/llama3.1:8b",
    "llama3.1:70b":                 "ollama/llama3.1:70b",
}


def get_litellm_model(model_name: str) -> str:
    """Resolve short config name → LiteLLM prefixed model string."""
    return MODEL_REGISTRY.get(model_name, model_name)


class LLMClient:
    """
    Provider-agnostic LLM client for the FSLSM-RAG-MCP thesis.

    Usage:
        client = LLMClient("gpt-4o", temperature=0.3)
        resp = client.chat(system="You are...", user="Question?")
        print(resp.content, resp.prompt_tokens, resp.cost)
    """

    def __init__(self, model: str, temperature: float = 0.3):
        self.model_name = model
        self.litellm_model = get_litellm_model(model)
        self.temperature = temperature

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Send a single system+user turn to the model.
        All provider differences are handled by litellm.completion().
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        response = litellm.completion(
            model=self.litellm_model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens,
        )

        # Extract cost (LiteLLM tracks this per-call)
        cost = response._hidden_params.get("response_cost", 0.0) or 0.0

        return LLMResponse(
            content=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model=response.model,
            cost=cost,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )
```

**Key differences from the old custom LLMClient:**

| Aspect | Old (custom) | New (LiteLLM) |
|---|---|---|
| Provider routing | 3 separate `_openai_chat`, `_anthropic_chat`, `_ollama_chat` methods | Single `litellm.completion()` call |
| Prompt formatting | Manual per-provider (OpenAI uses `messages`, Anthropic uses `system` kwarg) | LiteLLM normalizes internally |
| Token counting | Different fields per provider (`usage.prompt_tokens` vs `usage.input_tokens`) | Unified `response.usage.prompt_tokens` |
| Cost tracking | Not available | Built-in `response._hidden_params["response_cost"]` |
| Rate limit handling | Not implemented (would need custom retry) | LiteLLM retries with exponential backoff |
| Adding new models | Write new `_provider_chat()` method | Add one line to `MODEL_REGISTRY` |
| Citable in thesis | "Custom Python wrapper" | "LiteLLM (BerriAI, 2024), open-source SDK" |

**Sanity test** (replaces the old Phase 0.4 test):

```python
from src.utils.llm_client import LLMClient

client = LLMClient("gpt-4o-mini", temperature=0.3)
resp = client.chat(system="You are helpful.", user="Say hi in one word.")
print(f"Content: {resp.content}")
print(f"Tokens: {resp.prompt_tokens} in / {resp.completion_tokens} out")
print(f"Cost: ${resp.cost:.6f}")
print(f"Model: {resp.model}")
```

**Done:** Test prints response from GPT-4o-mini with token counts and cost.

---

### Pre-2.C — DB Migration for knowledge_level

*(Unchanged from v1 — included for completeness)*

```bash
alembic revision --autogenerate -m "add knowledge_level to agents"
alembic upgrade head
```

**Update `db/models.py`** — add to `Agent`:
```python
knowledge_level = Column(String(20), nullable=True)
```

**Update `config/constants.py`** — add:
```python
KNOWLEDGE_LEVELS = ["beginner", "intermediate", "advanced"]
KNOWLEDGE_LEVEL_MAP = {
    1: "beginner",
    2: "intermediate",
    3: "advanced",
    4: None,   # general
    5: None,   # general
}
NUM_LEVELED_AGENTS = 48
NUM_GENERAL_AGENTS = 32
NUM_AGENTS = 80
```

**Done:** `knowledge_level` column exists in `agents` table.

---

### Pre-2.D — Update `.env` for LiteLLM

LiteLLM reads API keys from environment variables automatically. Ensure `.env` has:

```bash
# LLM API Keys (read by LiteLLM automatically)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local, no key needed — LiteLLM defaults to http://localhost:11434)
# Optional: OLLAMA_API_BASE=http://localhost:11434
```

No code changes needed — LiteLLM picks up `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from the environment. The `config/settings.py` Pydantic settings class still loads `.env` via `python-dotenv`, so the keys are available in the environment when the process starts.

---

## Task 2.1 — Student Agent System Prompt Builder (Updated)

*(Unchanged from v1 — knowledge_level integration)*

**`src/agents/prompts/student_system.py`** — takes optional `knowledge_level`:

```python
from config.constants import FSLSM_DIM_LABELS

KNOWLEDGE_LEVEL_INSTRUCTIONS = {
    "beginner": (
        "You are a beginner-level student with limited prior exposure to this topic. "
        "You struggle with technical jargon and need concepts explained from first "
        "principles. You often ask for simpler analogies and may confuse related "
        "terms. You feel overwhelmed by advanced notation or multi-step derivations."
    ),
    "intermediate": (
        "You are an intermediate-level student with a reasonable foundation in the "
        "basics. You understand core concepts like gradient descent and loss functions "
        "but need help connecting ideas across topics. You can follow moderately "
        "complex derivations but ask for clarification on non-obvious steps."
    ),
    "advanced": (
        "You are an advanced student who is comfortable with the mathematical "
        "foundations and standard algorithms. You ask about edge cases, theoretical "
        "nuances, implementation trade-offs, and recent research extensions. You "
        "prefer depth over simplification and may challenge explanations."
    ),
}


def build_student_system_prompt(
    profile: dict,
    knowledge_level: str | None = None,
) -> str:
    dims = profile["dimensions"]
    instructions = profile["behavioral_instructions"]

    dim_lines = []
    for dim_key, (neg_label, pos_label) in FSLSM_DIM_LABELS.items():
        pole_label = neg_label if dims[dim_key] == -1 else pos_label
        dim_lines.append(f"- **{pole_label}** ({dim_key}): {instructions[dim_key]}")

    fslsm_block = f"""Your Felder-Silverman Learning Style Profile is:
{chr(10).join(dim_lines)}"""

    if knowledge_level and knowledge_level in KNOWLEDGE_LEVEL_INSTRUCTIONS:
        knowledge_block = f"""

Your Knowledge Level: {knowledge_level.capitalize()}
{KNOWLEDGE_LEVEL_INSTRUCTIONS[knowledge_level]}"""
    else:
        knowledge_block = ""

    return f"""You are a virtual undergraduate student studying Introductory Machine Learning.
You have a specific learning style and must consistently behave according to your assigned profile in ALL interactions.

{fslsm_block}
{knowledge_block}

Interaction Rules:
1. Always respond AS the student, not as a tutor or assistant.
2. Ask questions, express confusion, and request clarification in ways consistent with your learning style{' and knowledge level' if knowledge_level else ''}.
3. When receiving explanations, react authentically:
   - If the content matches your style, express satisfaction and engagement.
   - If the content mismatches your style, express mild frustration or request adaptation.
4. Do not explicitly state your FSLSM scores or label yourself. Express your preferences through natural behavior.
5. Maintain consistency across all turns in the conversation.

Topic domain: Introductory Machine Learning (neural networks, optimization, gradient descent).
"""
```

---

## Task 2.2 — Agent Factory (Updated)

*(Unchanged from v1)*

**`src/agents/agent_factory.py`**:

```python
import json
from db.models import Agent, FslsmProfile
from db import get_session
from src.agents.prompts.student_system import build_student_system_prompt
from config.constants import KNOWLEDGE_LEVEL_MAP

def create_agents(llm_model: str, profiles_json: str = "data/fslsm/profiles.json"):
    profiles = json.load(open(profiles_json))

    with get_session() as session:
        db_profiles = {p.profile_code: p.id for p in session.query(FslsmProfile).all()}

        for profile in profiles:
            code = profile["profile_code"]
            profile_id = db_profiles[code]

            for instance in range(1, 6):
                knowledge_level = KNOWLEDGE_LEVEL_MAP[instance]

                if knowledge_level:
                    agent_uid = f"agent_{code}_I{instance:02d}_{knowledge_level[:3]}"
                else:
                    agent_uid = f"agent_{code}_I{instance:02d}_gen"

                system_prompt = build_student_system_prompt(
                    profile, knowledge_level=knowledge_level,
                )

                session.add(Agent(
                    agent_uid=agent_uid,
                    profile_id=profile_id,
                    instance_num=instance,
                    llm_model=llm_model,
                    system_prompt=system_prompt,
                    knowledge_level=knowledge_level,
                ))

        session.commit()

    print(f"Created 80 agents for {llm_model} (48 leveled + 32 general)")
```

---

## Task 2.3 — ILS Answering Prompt (Unchanged)

```python
def build_ils_question_prompt(question: dict) -> str:
    return f"""You are taking a learning style survey. Answer the following question honestly based on your learning preferences as described in your profile.

Question {question['q_num']}: {question['text']}

Options:
  (a) {question['option_a']['text']}
  (b) {question['option_b']['text']}

Respond with ONLY the letter "a" or "b". Do not explain your answer."""
```

---

## Task 2.4 — ILS Evaluator (Updated for LiteLLM)

**`src/agents/ils_evaluator.py`** — the key change is that `LLMClient` is now LiteLLM-backed, so the constructor no longer needs a `provider` argument. It also captures `cost` per call:

```python
import json, re
from pathlib import Path
from tqdm import tqdm
from db.models import Agent, FslsmProfile
from db import get_session
from src.utils.llm_client import LLMClient   # Now backed by LiteLLM
from src.agents.prompts.ils_answering import build_ils_question_prompt
from config.constants import FSLSM_DIMENSIONS

def run_ils_for_agent(agent: Agent, questions: list, client: LLMClient, trial: int) -> dict:
    """Run 44 ILS questions for a single agent. Returns dimension scores + cost."""
    dim_scores = {d: 0 for d in FSLSM_DIMENSIONS}
    raw = []
    total_cost = 0.0

    for q in questions:
        response = client.chat(
            system=agent.system_prompt,
            user=build_ils_question_prompt(q),
            max_tokens=10,
        )
        total_cost += response.cost

        answer_text = response.content.strip().lower()
        match = re.search(r'[ab]', answer_text)
        if match:
            answer = match.group()
            pole = q[f"option_{answer}"]["pole"]
            dim_scores[q["dimension"]] += pole
        else:
            answer = None

        raw.append({
            "q_num": q["q_num"],
            "answer": answer,
            "raw_text": answer_text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost,
        })

    # Save raw response file
    out_path = Path(f"results/exp1/raw_responses/{agent.agent_uid}_trial{trial}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "agent_uid": agent.agent_uid,
        "knowledge_level": agent.knowledge_level,
        "model": client.model_name,
        "litellm_model": client.litellm_model,    # LiteLLM's resolved model string
        "trial": trial,
        "dim_scores": dim_scores,
        "total_cost_usd": total_cost,
        "raw": raw,
    }, indent=2))

    return dim_scores, total_cost


def run_experiment1(llm_model: str, num_trials: int = 3, temperature: float = 0.3):
    """
    Run ILS questionnaire across all 80 agents for a given model.

    NOTE: No `provider` argument needed — LiteLLM resolves the provider
    from the model string via MODEL_REGISTRY in llm_client.py.
    """
    questions = json.load(open("data/fslsm/ils_questionnaire.json"))
    client = LLMClient(llm_model, temperature=temperature)

    with get_session() as session:
        agents = (
            session.query(Agent)
            .join(FslsmProfile)
            .filter(Agent.llm_model == llm_model)
            .all()
        )
        agent_data = []
        for a in agents:
            agent_data.append({
                "agent": a,
                "assigned": {
                    "act_ref": a.profile.act_ref,
                    "sen_int": a.profile.sen_int,
                    "vis_ver": a.profile.vis_ver,
                    "seq_glo": a.profile.seq_glo,
                },
                "knowledge_level": a.knowledge_level,
            })

    results = []
    cumulative_cost = 0.0

    for ad in tqdm(agent_data, desc=f"Exp1 [{llm_model}]"):
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
                "profile_code": agent.agent_uid.split("_I")[0].replace("agent_", ""),
                "instance_num": agent.instance_num,
                "trial": trial,
                "assigned": ad["assigned"],
                "detected": detected,
                "raw_scores": dim_scores,
                "cost_usd": call_cost,
            })

    print(f"\n  Total cost for {llm_model}: ${cumulative_cost:.4f}")
    return results
```

**Changes from v1:**
- `run_experiment1()` no longer takes a `provider` parameter — LiteLLM handles routing.
- Each raw response file logs `litellm_model` (the resolved provider-prefixed string) for auditability.
- Cost is tracked per-call and accumulated per-model for the cost summary.

---

## Task 2.5 — Experiment Runner (Updated for LiteLLM)

**`experiments/exp1_agent_fidelity/run.py`** — simplified because `LLMProvider` enum is gone:

```python
import yaml, json
from pathlib import Path
from src.agents.agent_factory import create_agents
from src.agents.ils_evaluator import run_experiment1

CONFIG = yaml.safe_load(open("experiments/exp1_agent_fidelity/config.yaml"))

for model_cfg in CONFIG["models"]:
    model_name = model_cfg["name"]
    temperature = model_cfg.get("temperature", 0.3)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  (LiteLLM handles provider routing automatically)")
    print(f"{'='*60}")

    # Step 1: Create 80 agents (48 leveled + 32 general)
    print(f"\n--- Creating agents ---")
    create_agents(model_name)

    # Step 2: Run ILS questionnaire
    num_trials = CONFIG["ils_questionnaire"]["num_trials"]
    print(f"\n--- Running ILS questionnaire ({num_trials} trials) ---")
    results = run_experiment1(
        llm_model=model_name,
        num_trials=num_trials,
        temperature=temperature,
    )

    # Step 3: Save results
    out_dir = Path("results/exp1/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(":", "_")
    out_file = out_dir / f"{safe_name}_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} records → {out_file}")

    # Step 4: Quick PRA summary
    from src.evaluation.metrics import profile_recovery_accuracy
    pra = profile_recovery_accuracy(results)
    print(f"PRA (overall 4D): {pra['overall_4d']:.3f}")
    for dim, score in pra["per_dimension"].items():
        print(f"  {dim}: {score:.3f}")

    # Step 5: Cost summary
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    print(f"Total cost: ${total_cost:.4f}")
```

**`experiments/exp1_agent_fidelity/config.yaml`** (updated — no `provider` field needed):

```yaml
experiment: exp1_agent_fidelity
description: "Fidelity of Virtual Student Agent Encoding (RQ2) — LiteLLM + knowledge_level"

agents:
  num_profiles: 16
  instances_per_profile: 5
  total_agents: 80
  knowledge_level_map:
    1: beginner
    2: intermediate
    3: advanced
    4: null
    5: null

# LiteLLM resolves provider from model name prefix.
# MODEL_REGISTRY in llm_client.py maps these short names
# to LiteLLM's provider-prefixed format.
models:
  - name: gpt-4o
    temperature: 0.3
  - name: claude-sonnet-4-5-20251001
    temperature: 0.3
  - name: llama3.1:8b
    temperature: 0.3

ils_questionnaire:
  num_questions: 44
  num_trials: 3

evaluation:
  metrics:
    - profile_recovery_accuracy
    - dimension_alignment_score
  analysis_slices:
    - by_model
    - by_knowledge_level
    - by_dimension
    - by_profile
    - by_cost                          # NEW: cost comparison across models
  visualization:
    - radar_chart_per_agent
    - heatmap_profiles_x_dimensions
    - model_comparison_bar_chart
    - knowledge_level_pra_comparison
    - cost_per_model_bar_chart         # NEW
```

---

## Task 2.6 — Metrics Computation (Updated)

**`src/evaluation/metrics.py`** — add cost analysis alongside PRA:

```python
import numpy as np
import pandas as pd
from config.constants import FSLSM_DIMENSIONS

def profile_recovery_accuracy(results: list[dict]) -> dict:
    """PRA per dimension and overall (4D). Ties (detected=0) count as mismatches."""
    dim_matches = {d: [] for d in FSLSM_DIMENSIONS}
    ties = {d: 0 for d in FSLSM_DIMENSIONS}

    for r in results:
        for d in FSLSM_DIMENSIONS:
            detected = r["detected"][d]
            assigned = r["assigned"][d]
            if detected == 0:
                ties[d] += 1
                dim_matches[d].append(0)
            else:
                dim_matches[d].append(int(assigned == detected))

    pra_per_dim = {d: float(np.mean(dim_matches[d])) for d in FSLSM_DIMENSIONS}
    pra_4d = float(np.mean(list(pra_per_dim.values())))
    return {"per_dimension": pra_per_dim, "overall_4d": pra_4d, "ties_per_dimension": ties}


def pra_by_knowledge_level(results: list[dict]) -> dict:
    """Slice PRA by knowledge_level group."""
    grouped = {}
    for r in results:
        level = r.get("knowledge_level") or "general"
        grouped.setdefault(level, []).append(r)
    return {level: profile_recovery_accuracy(group) for level, group in grouped.items()}


def cost_summary(results: list[dict]) -> dict:
    """Aggregate cost from LiteLLM per-call tracking."""
    costs = [r.get("cost_usd", 0) for r in results]
    return {
        "total_usd": sum(costs),
        "mean_per_agent_trial_usd": np.mean(costs) if costs else 0,
        "num_calls": len(costs),
    }


def dimension_alignment_score(agent_embeddings: np.ndarray,
                               trait_embeddings: np.ndarray) -> float:
    return float(np.dot(agent_embeddings, trait_embeddings))
```

**`experiments/exp1_agent_fidelity/analyze.py`** — now exports cost alongside PRA:

```python
import json, pandas as pd
from pathlib import Path
from src.evaluation.metrics import (
    profile_recovery_accuracy, pra_by_knowledge_level, cost_summary,
)
from config.constants import FSLSM_DIMENSIONS

MODELS = ["gpt-4o", "claude-sonnet-4-5-20251001", "llama3.1_8b"]

def run_analysis():
    all_rows = []
    cost_rows = []

    for model in MODELS:
        safe = model.replace("/", "_").replace(":", "_")
        results_file = Path(f"results/exp1/metrics/{safe}_results.json")
        if not results_file.exists():
            print(f"Skipping {model}")
            continue

        results = json.loads(results_file.read_text())

        # --- PRA overall ---
        pra = profile_recovery_accuracy(results)
        for dim in FSLSM_DIMENSIONS:
            all_rows.append({
                "model": model, "knowledge_level": "ALL",
                "dimension": dim, "pra": pra["per_dimension"][dim],
                "ties": pra["ties_per_dimension"][dim],
            })
        all_rows.append({
            "model": model, "knowledge_level": "ALL",
            "dimension": "overall_4d", "pra": pra["overall_4d"],
            "ties": sum(pra["ties_per_dimension"].values()),
        })

        # --- PRA by knowledge_level ---
        for level, level_pra in pra_by_knowledge_level(results).items():
            for dim in FSLSM_DIMENSIONS:
                all_rows.append({
                    "model": model, "knowledge_level": level,
                    "dimension": dim, "pra": level_pra["per_dimension"][dim],
                    "ties": level_pra["ties_per_dimension"][dim],
                })
            all_rows.append({
                "model": model, "knowledge_level": level,
                "dimension": "overall_4d", "pra": level_pra["overall_4d"],
                "ties": sum(level_pra["ties_per_dimension"].values()),
            })

        # --- Cost (from LiteLLM) ---
        cs = cost_summary(results)
        cost_rows.append({"model": model, **cs})

    # Export PRA
    df_pra = pd.DataFrame(all_rows)
    pra_path = Path("results/exp1/metrics/pra_das_summary.csv")
    df_pra.to_csv(pra_path, index=False)
    print(f"PRA → {pra_path}")

    # Export cost
    df_cost = pd.DataFrame(cost_rows)
    cost_path = Path("results/exp1/metrics/cost_summary.csv")
    df_cost.to_csv(cost_path, index=False)
    print(f"Cost → {cost_path}")

    print(f"\n{df_pra.to_string(index=False)}")
    print(f"\n{df_cost.to_string(index=False)}")
    return df_pra, df_cost

if __name__ == "__main__":
    run_analysis()
```

---

## Task 2.7 — Visualization (Updated)

*(Radar chart, heatmap, knowledge-level comparison, and model comparison are unchanged from v1. Adding one new chart:)*

### 2.7.5 — Cost per Model Bar Chart (NEW)

```python
def cost_per_model_bar(df_cost, save_path: str):
    """
    Simple bar chart: total USD cost per model for Exp1.
    Uses LiteLLM's per-call cost tracking.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    models = df_cost["model"].values
    costs = df_cost["total_usd"].values
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(range(len(models)), costs, color=colors[:len(models)])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split(":")[0] for m in models], rotation=15)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Experiment 1 — API Cost per Model (via LiteLLM)")

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"${cost:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

**Done:** `results/exp1/figures/cost_per_model.png` exists.

---

## Phase 2 — Final Checklist (Updated v2)

| Check | Command / Criterion |
|---|---|
| LiteLLM installed | `python -c "import litellm; print(litellm.__version__)"` |
| LLMClient uses LiteLLM | `grep "litellm.completion" src/utils/llm_client.py` — present |
| No direct `openai.` or `anthropic.` calls in experiment code | `grep -rn "openai\.\|anthropic\." src/agents/ experiments/` — only in `llm_client.py` internals |
| 80 agents created per model | `SELECT COUNT(*) FROM agents WHERE llm_model='gpt-4o';` → 80 |
| Knowledge level distribution | beginner:16, intermediate:16, advanced:16, NULL:32 |
| Raw responses include `litellm_model` | `jq .litellm_model results/exp1/raw_responses/*.json \| head` |
| Raw responses include `cost_usd` | `jq .total_cost_usd results/exp1/raw_responses/*.json \| head` |
| PRA summary CSV | `head results/exp1/metrics/pra_das_summary.csv` |
| Cost summary CSV | `cat results/exp1/metrics/cost_summary.csv` |
| PRA ≥ 0.70 overall | Check CSV |
| PRA general ≈ PRA leveled | Δ < 0.10 between general and leveled groups |
| All figures generated | `ls results/exp1/figures/` includes radar, heatmap, knowledge_level, cost |

---

## Prompt Refinement Loop (Unchanged from v1)

If PRA < 0.70 or knowledge-leveled agents differ by > 0.10 from general:

1. Knowledge contamination check
2. Increase FSLSM behavioral specificity
3. Add ILS-aligned few-shot examples
4. Lower temperature to 0.1
5. Re-run on 10-agent sample

---

## Recommended Execution Order (Updated v2)

| Day | Tasks |
|---|---|
| Day 1 | Pre-2.A: Install LiteLLM |
| Day 1 | Pre-2.B: Replace LLMClient with LiteLLM wrapper, run sanity test |
| Day 1 | Pre-2.C: DB migration (knowledge_level column) |
| Day 1 | Pre-2.D: Verify `.env` keys work with LiteLLM |
| Day 1 | Task 2.1–2.2: Prompt builder + agent factory |
| Day 2 | Task 2.3–2.4: ILS prompt + evaluator |
| Day 2 | Sanity test: 5 agents, 1 trial, inspect JSON (check `litellm_model` and `cost_usd` fields) |
| Day 3 | Task 2.5: Run GPT-4o (or gpt-4o-mini) — full 80 × 3 trials |
| Day 4 | Task 2.5: Run Claude Sonnet |
| Day 4 | Task 2.6: Metrics + cost CSV export |
| Day 5 | Task 2.5: Ollama / Llama 3 run |
| Day 5 | Task 2.7: All visualizations (including cost chart) |
| Day 6 | Buffer: prompt refinement, final analysis |

---

## Thesis Write-Up Notes for LiteLLM

When writing the methodology chapter, include a paragraph similar to:

> **Section 3.3 (System Architecture) or 3.5 (Experimental Setup):**
> To ensure fair cross-model comparison, all LLM interactions across experiments are mediated through LiteLLM (BerriAI, 2024), an open-source, MIT-licensed Python SDK that provides a unified completion interface for heterogeneous LLM providers. LiteLLM internally normalizes prompt formatting, token counting, and response parsing, eliminating potential confounds introduced by provider-specific API adapters. This design ensures that observed differences in Profile Recovery Accuracy (PRA) and Dimension Alignment Score (DAS) are attributable to model capabilities rather than implementation artifacts. Cost and token usage are tracked per-call using LiteLLM's built-in accounting, enabling the cost-efficiency analysis reported in Section X.X.

**BibTeX entry:**
```bibtex
@software{litellm2024,
  author       = {{BerriAI}},
  title        = {{LiteLLM}: Call 100+ {LLM} {API}s in {OpenAI} Format},
  year         = {2024},
  url          = {https://github.com/BerriAI/litellm},
  note         = {MIT License. Accessed: 2026-03-13},
}
```

---

## What This Enables for Experiments 2 & 3

- **Experiment 2:** The tutor agent and judge LLM both go through LiteLLM — same unified interface, consistent token tracking, easy model swapping for ablations.
- **Experiment 3:** Prompt Token Savings (PTS) metric compares `response.usage.prompt_tokens` across S0/S1a/S1b conditions. With LiteLLM, these counts come from the same normalization layer regardless of which model is used — critical for a valid PTS comparison.
- **Adding models later:** If a reviewer asks "what about Gemini?" or "try Mistral?", you add one line to `MODEL_REGISTRY` and one entry in `config.yaml`. No code changes to evaluators, metrics, or analysis.
