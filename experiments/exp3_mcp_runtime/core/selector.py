from __future__ import annotations

import os
import re
from typing import Iterable

from openai import OpenAI

from experiments.exp3_mcp_runtime.core.profile_decoder import decode_profile
from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex
from experiments.exp3_mcp_runtime.tools.tool_registry import MCPTool, TOOL_REGISTRY

_SELECTOR_CLIENT: OpenAI | None = None
_SELECTOR_UNAVAILABLE = False

INTENT_RULES: list[tuple[int, tuple[str, ...], float]] = [
    (15, ("latest", "recent", "current", "outside d2l", "beyond d2l", "industry", "benchmark trend", "new benchmark"), 0.35),
    (14, ("find the d2l section", "retrieve", "what does d2l say", "textbook passage", "d2l passage", "locate the d2l"), 0.35),
    (12, ("quiz me", "test my understanding", "check what i know", "make a quiz", "create a quiz"), 0.35),
    (13, ("summarize", "summary", "key points", "tl;dr", "concise summary"), 0.35),
    (11, ("exercise", "practice task", "hands-on", "implement", "coding exercise", "let me practice"), 0.35),
    (4, ("draw", "diagram", "visualize", "illustrate", "ascii", "flowchart"), 0.35),
    (6, ("compare", "contrast", "versus", "difference between", "different from"), 0.35),
    (2, ("derive", "step by step", "walk through", "trace", "step through"), 0.35),
    (3, ("worked example", "show an example", "solved example", "calculate", "compute"), 0.30),
    (8, ("adapt this", "rewrite this", "re-explain", "learning style", "student style"), 0.30),
    (9, ("explain this differently", "convert this", "make this more visual", "make it more structured"), 0.30),
    (10, ("think-pair-share", "reflect on", "pause and discuss", "reflection prompt"), 0.30),
    (7, ("concept map", "big picture", "map the relationship", "how do these connect"), 0.30),
    (5, ("analogy", "intuition", "metaphor", "why does this matter"), 0.30),
]

PROFILE_TOOL_BOOSTS: dict[str, dict[int, float]] = {
    "Active": {11: 0.05, 12: 0.03, 2: 0.02},
    "Reflective": {10: 0.05, 13: 0.03},
    "Sensing": {3: 0.04, 1: 0.03, 6: 0.02},
    "Intuitive": {5: 0.04, 15: 0.02},
    "Visual": {4: 0.05, 7: 0.02},
    "Verbal": {1: 0.04, 6: 0.02},
    "Sequential": {2: 0.05, 3: 0.03},
    "Global": {7: 0.05, 13: 0.02},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _intent_scores(query: str) -> dict[int, float]:
    q = _normalize(query)
    scores: dict[int, float] = {}
    for tool_id, patterns, boost in INTENT_RULES:
        if any(pattern in q for pattern in patterns):
            scores[tool_id] = max(scores.get(tool_id, 0.0), boost)
    return scores


def _get_selector_client() -> OpenAI | None:
    global _SELECTOR_CLIENT
    if _SELECTOR_UNAVAILABLE:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    if _SELECTOR_CLIENT is None:
        _SELECTOR_CLIENT = OpenAI(api_key=api_key)
    return _SELECTOR_CLIENT


def build_s0_selector_prompt(tools: Iterable[MCPTool]) -> str:
    tool_blocks = []
    for tool in tools:
        tool_blocks.append(
            f"Tool {tool.tool_id}: {tool.name}\n"
            f"Category: {tool.category}\n"
            f"Description: {tool.description}\n"
        )
    return (
        "You are an instructional tool selector for machine learning tutoring.\n"
        "Choose the single best tool for the student's question.\n"
        "Do not assume any specific FSLSM learning profile.\n"
        "Return ONLY the numeric tool id.\n\n"
        f"{''.join(tool_blocks)}"
    )


def _heuristic_select(query: str) -> int:
    scores = _intent_scores(query)
    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]
    return 1


def select_s0_tool(query: str, tools: Iterable[MCPTool] = TOOL_REGISTRY) -> int:
    global _SELECTOR_UNAVAILABLE
    client = _get_selector_client()
    if client is None:
        return _heuristic_select(query)

    prompt = build_s0_selector_prompt(tools)
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
            temperature=0,
            timeout=60,
        )
        return int(response.choices[0].message.content.strip())
    except Exception:
        _SELECTOR_UNAVAILABLE = True
        return _heuristic_select(query)


def _candidate_ids_with_intent(index: ToolIndex, query: str, k: int) -> set[int]:
    candidate_ids = {tool.tool_id for tool, _ in index.retrieve(query, k=k)}
    candidate_ids.update(_intent_scores(query))
    return candidate_ids


def select_retrieved_tool(
    *,
    index: ToolIndex,
    query: str,
    profile: dict | None = None,
    use_profile: bool = False,
    k: int = 5,
) -> int:
    raw_hits = index.retrieve(query, k=k)
    candidate_ids = _candidate_ids_with_intent(index, query, k=k)
    by_id = {tool.tool_id: score for tool, score in raw_hits}
    intent_scores = _intent_scores(query)

    profile_scores: dict[int, float] = {}
    if use_profile and profile:
        for dim in decode_profile(profile):
            for tool_id, boost in PROFILE_TOOL_BOOSTS.get(dim, {}).items():
                profile_scores[tool_id] = profile_scores.get(tool_id, 0.0) + boost

    ranked = []
    for tool_id in candidate_ids:
        score = by_id.get(tool_id, 0.0)
        score += intent_scores.get(tool_id, 0.0)
        score += profile_scores.get(tool_id, 0.0)
        ranked.append((score, tool_id))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return ranked[0][1] if ranked else _heuristic_select(query)
