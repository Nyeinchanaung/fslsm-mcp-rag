"""
S0 (Prompt Bloat Baseline): Real LLM tool selection with all 15 schemas.
Injects all tool descriptions + structured 5-step matching into context.
"""
from __future__ import annotations

import os
import time
from openai import OpenAI

from experiments.exp3_revised.tools.tool_registry import TOOL_REGISTRY, s0_prompt_tokens

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


_POLE_DESCRIPTIONS = {
    "Active":      "learns by doing; prefers hands-on exercises, interactive tasks, active problem-solving",
    "Reflective":  "learns by thinking; prefers reflecting before answering, quiet review, synthesis tasks",
    "Sensing":     "prefers concrete facts, worked examples, real-world applications, standard procedures",
    "Intuitive":   "prefers abstract concepts, analogies, theoretical connections, cross-domain patterns",
    "Visual":      "prefers diagrams, charts, graphs, computation graphs, spatial representations",
    "Verbal":      "prefers text-based explanations, written definitions, narrative descriptions",
    "Sequential":  "prefers step-by-step explanations with clear numbered stages and intermediate steps",
    "Global":      "prefers overviews, big-picture summaries, holistic understanding before details",
}


def get_profile_description(profile_label: str) -> str:
    """Convert 'Active+Sensing+Visual+Sequential' label to a prose description."""
    poles = [p.strip() for p in profile_label.replace("+", " ").split()]
    parts = [_POLE_DESCRIPTIONS[p] for p in poles if p in _POLE_DESCRIPTIONS]
    return "; ".join(parts) if parts else "general learner with no specific style preference"


S0_SYSTEM_PROMPT = """You are an AI tutor tool selector for machine learning education.
Select the SINGLE most appropriate tool for the student's question.

Student learning profile: {profile_description}

Available tools:
{tool_schemas}

Selection process — follow these 5 steps:
1. Identify the question's primary intent: explain concept, compare/contrast, trace evolution,
   synthesize workflow, practice/exercise, adapt to style, search for information, or summarize.
2. Note the student's dominant learning preferences from the profile above.
3. Map intent to tool category:
     explain/trace/synthesize → tools 1-7 (content tools)
     style adaptation         → tools 8-9 (PersonaRAG Adapter, FSLSM Styler)
     practice/quiz            → tools 10-12 (Think-Pair-Share, Interactive Exercise, Quiz Generator)
     summarize                → tool 13 (Summarizer)
     retrieve from textbook   → tool 14 (Content Retriever)
     search recent research   → tool 15 (Web Search Tool)
4. Within the matched category, narrow by learning style:
     Visual → prefer tools 4, 7   |  Sequential → prefer tools 2, 3
     Active → prefer tools 10, 11 |  Reflective → prefer tools 8, 9
     Global → prefer tools 7, 13  |  Intuitive  → prefer tools 5, 7
     Verbal → prefer tools 1, 6   |  Sensing    → prefer tools 2, 3
5. Select the single best matching tool_id.

Respond with ONLY a single integer (1-15). Nothing else."""


def _build_tool_schemas() -> str:
    lines = []
    for t in TOOL_REGISTRY:
        dims = ", ".join(t.fslsm_dims[:3])
        lines.append(
            f"Tool {t.tool_id}: {t.name}\n"
            f"  Category: {t.category}\n"
            f"  FSLSM: {dims}\n"
            f"  Description: {t.description[:200]}\n"
        )
    return "\n".join(lines)


TOOL_SCHEMAS_TEXT = _build_tool_schemas()


def _s0_call_with_retry(call_fn, query_snippet: str) -> object | None:
    delays = [2, 4, 8]
    for attempt, delay in enumerate(delays, 1):
        try:
            return call_fn()
        except Exception as e:
            print(f"[s0_retry] attempt {attempt} failed for '{query_snippet[:40]}' ({e}), retrying in {delay}s...")
            time.sleep(delay)
    try:
        return call_fn()
    except Exception as e:
        print(f"[s0_retry] FAILED '{query_snippet[:40]}' after {len(delays)+1} attempts: {e}")
        return None


def s0_select_tool(query: str, profile_label: str = "") -> int:
    """Real LLM tool selection with all 15 schemas + 5-step structured matching.

    Returns selected tool_id (1–15). Falls back to 1 on parse error.
    """
    profile_desc = get_profile_description(profile_label) if profile_label else "general learner"
    prompt = S0_SYSTEM_PROMPT.format(
        profile_description=profile_desc,
        tool_schemas=TOOL_SCHEMAS_TEXT,
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    def _call():
        return _get_client().chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=5,
            temperature=0,
            timeout=60,
        )

    response = _s0_call_with_retry(_call, query)
    if response is None:
        return 1
    try:
        raw = response.choices[0].message.content.strip()
        tool_id = int(raw)
        if 1 <= tool_id <= 15:
            return tool_id
    except (ValueError, IndexError):
        pass
    return 1  # fallback: Concept Explainer


def s0_input_tokens() -> int:
    """Token count for S0 (all 15 schemas injected)."""
    return s0_prompt_tokens()
