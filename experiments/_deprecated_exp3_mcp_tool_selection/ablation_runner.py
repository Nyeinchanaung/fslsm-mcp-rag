"""S0/S1a/S1b runner + SQLite results logger.

For each session and each condition the runner records:
    - selected_tool_id : tool the condition picked
    - optimal_tool_id  : ground-truth optimum from `tool_registry.get_optimal_tool_id`
    - tsa_hit          : selected == optimal
    - input_tokens     : total tool-schema tokens carried in the prompt
    - pts_delta        : token savings vs S0 baseline, in percent

Results are appended to a single SQLite database; the dry-run and full-run
scripts use the same logger and just differ in input volume.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

from experiments.exp3_mcp_tool_selection.fslsm_query_augmentor import FSLSMQueryAugmentor
from experiments.exp3_mcp_tool_selection.tool_index import ToolIndex
from experiments.exp3_mcp_tool_selection.tool_registry import (
    get_optimal_tool_id,
    s0_prompt_tokens,
    s0_select_tool_id,
    s1_prompt_tokens,
)


@dataclass
class SessionResult:
    condition: str
    session_id: str
    student_profile: dict
    query: str
    selected_tool_id: int
    optimal_tool_id: int
    tsa_hit: bool
    input_tokens: int
    pts_delta: float


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #

class AblationRunner:
    def __init__(self, tool_index: ToolIndex):
        self.tool_index = tool_index
        self.augmentor = FSLSMQueryAugmentor()
        self._s0_tokens = s0_prompt_tokens()

    def run_session(self, *, profile: dict, query: str, session_id: str) -> dict[str, SessionResult]:
        optimal = get_optimal_tool_id(query, profile)

        # S0 — keyword-overlap selection over all 15 tools, full prompt cost.
        s0_id = s0_select_tool_id(query)
        s0_tokens = self._s0_tokens
        s0 = SessionResult(
            condition="S0",
            session_id=session_id,
            student_profile=profile,
            query=query,
            selected_tool_id=s0_id,
            optimal_tool_id=optimal,
            tsa_hit=(s0_id == optimal),
            input_tokens=s0_tokens,
            pts_delta=0.0,  # baseline
        )

        # S1a — FAISS top-1 over raw query, single-tool prompt cost.
        s1a_id = self.tool_index.retrieve(query, k=1)[0][0].tool_id
        s1a_tokens = s1_prompt_tokens(s1a_id)
        s1a = SessionResult(
            condition="S1a",
            session_id=session_id,
            student_profile=profile,
            query=query,
            selected_tool_id=s1a_id,
            optimal_tool_id=optimal,
            tsa_hit=(s1a_id == optimal),
            input_tokens=s1a_tokens,
            pts_delta=100.0 * (1 - s1a_tokens / self._s0_tokens),
        )

        # S1b — FAISS top-1 over FSLSM-augmented query.
        augmented = self.augmentor.augment(query, profile)
        s1b_id = self.tool_index.retrieve(augmented, k=1)[0][0].tool_id
        s1b_tokens = s1_prompt_tokens(s1b_id)
        s1b = SessionResult(
            condition="S1b",
            session_id=session_id,
            student_profile=profile,
            query=query,
            selected_tool_id=s1b_id,
            optimal_tool_id=optimal,
            tsa_hit=(s1b_id == optimal),
            input_tokens=s1b_tokens,
            pts_delta=100.0 * (1 - s1b_tokens / self._s0_tokens),
        )

        return {"S0": s0, "S1a": s1a, "S1b": s1b}


# --------------------------------------------------------------------------- #
# Logger                                                                      #
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exp3_session_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    condition          TEXT NOT NULL,
    session_id         TEXT NOT NULL,
    student_profile    TEXT NOT NULL,
    query              TEXT NOT NULL,
    selected_tool_id   INTEGER NOT NULL,
    optimal_tool_id    INTEGER NOT NULL,
    tsa_hit            INTEGER NOT NULL,
    input_tokens       INTEGER NOT NULL,
    pts_delta          REAL NOT NULL,
    created_at         TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_results_condition ON exp3_session_results(condition);
CREATE INDEX IF NOT EXISTS idx_results_session   ON exp3_session_results(session_id);
"""


class ExperimentLogger:
    def __init__(self, db_path: Path | str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def log(self, result: SessionResult) -> None:
        d = asdict(result)
        self.conn.execute(
            """INSERT INTO exp3_session_results
                 (condition, session_id, student_profile, query,
                  selected_tool_id, optimal_tool_id, tsa_hit,
                  input_tokens, pts_delta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d["condition"], d["session_id"], json.dumps(d["student_profile"]),
                d["query"], d["selected_tool_id"], d["optimal_tool_id"],
                int(d["tsa_hit"]), d["input_tokens"], d["pts_delta"],
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- live aggregates (used by dry-run gate) --------------------------- #

    def tsa_by_condition(self) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT condition, AVG(tsa_hit) FROM exp3_session_results GROUP BY condition"
        ).fetchall()
        return {c: float(v) for c, v in rows}

    def pts_by_condition(self) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT condition, AVG(pts_delta) FROM exp3_session_results GROUP BY condition"
        ).fetchall()
        return {c: float(v) for c, v in rows}

    def reset(self) -> None:
        """Truncate the results table — used between dry-run iterations."""
        self.conn.execute("DELETE FROM exp3_session_results")
        self.conn.commit()
