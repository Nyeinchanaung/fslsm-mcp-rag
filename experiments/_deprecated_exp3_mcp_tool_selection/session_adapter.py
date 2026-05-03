"""Bridge from Exp2 raw session records to (session_id, profile, query) tuples.

The Exp2 JSONL records have shape
    {"agent_id": "...", "question_id": "...", "mode": "R0|R1", ...}
while the Exp3 ablation runner needs
    {"session_id": "...", "student_profile": {...}, "query": "..."}.

Profile lookup → `data/agents/validated_agents.json` (agent_uid → fslsm_vector).
Query lookup   → `data/exp2/filtered_questions.json` (question_id → question).

We dedupe on (agent_id, question_id) across modes since Exp3 is mode-agnostic
— each unique (profile, query) pair should be counted once per condition.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from experiments.exp3_mcp_tool_selection.config import (
    AGENTS_PATH,
    QUESTIONS_PATH,
    RAW_R0_PATH,
    RAW_R1_PATH,
)


def _load_lookup(path: Path, key: str, value_key: str | None = None) -> dict[str, Any]:
    items = json.loads(path.read_text())
    if value_key is None:
        return {item[key]: item for item in items}
    return {item[key]: item[value_key] for item in items}


def load_agents() -> dict[str, dict]:
    """agent_uid → full agent record (incl. fslsm_vector)."""
    return _load_lookup(AGENTS_PATH, "agent_uid")


def load_questions() -> dict[str, str]:
    """question_id → question text."""
    return _load_lookup(QUESTIONS_PATH, "question_id", "question")


def iter_sessions(
    *,
    include_r0: bool = True,
    include_r1: bool = True,
    dedupe: bool = True,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield normalised session records `{session_id, student_profile, query}`.

    Args:
        include_r0: Read from `raw_sessions_r0.jsonl`.
        include_r1: Read from `raw_sessions_r1.jsonl`.
        dedupe: If True, yield each unique (agent_id, question_id) only once
                (R1 takes priority if both present).
        limit: Stop after yielding this many records (post-dedupe).
    """
    agents = load_agents()
    questions = load_questions()

    seen: set[tuple[str, str]] = set()
    sources: list[Path] = []
    if include_r1:
        sources.append(RAW_R1_PATH)
    if include_r0:
        sources.append(RAW_R0_PATH)

    yielded = 0
    for path in sources:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                agent_id = rec["agent_id"]
                question_id = rec["question_id"]
                key = (agent_id, question_id)
                if dedupe and key in seen:
                    continue
                seen.add(key)

                agent = agents.get(agent_id)
                question_text = questions.get(question_id)
                if agent is None or question_text is None:
                    continue  # skip orphans

                yield {
                    "session_id": f"{agent_id}__{question_id}",
                    "student_profile": agent["fslsm_vector"],
                    "query": question_text,
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
