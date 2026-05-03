"""
R2 Session Runner with dual logging for Exp3 (primary) + Exp2 (passive).

Runs one session through the full MCP pipeline:
  1. FSLSM-augmented FAISS retrieval (S1b tool selection)
  2. Tool-specific response generation
  3. Log Exp3 metrics (TSA, PTS)
  4. Log Exp2 passive data (tool selection + response for later evaluation)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from experiments.exp3_revised.tools.tool_registry import s0_prompt_tokens
from experiments.exp3_revised.tools.tool_index import ToolIndex
from experiments.exp3_revised.tools.tool_prompts import get_tool_prompt
from experiments.exp3_revised.core.profile_decoder import decode_profile, profile_to_label
from experiments.exp3_revised.core.fslsm_augmentor import augment_query
from experiments.exp3_revised.core.ground_truth import get_optimal_tool_id

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS exp3_session_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    condition       TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    question_id     TEXT    NOT NULL,
    question_type   TEXT    NOT NULL,
    student_profile TEXT    NOT NULL,
    query           TEXT    NOT NULL,
    selected_tool   INTEGER NOT NULL,
    optimal_tool    INTEGER NOT NULL,
    tsa_hit         INTEGER NOT NULL,
    input_tokens    INTEGER NOT NULL,
    pts_delta       REAL    NOT NULL,
    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_condition   ON exp3_session_results(condition);
CREATE INDEX IF NOT EXISTS idx_session_id  ON exp3_session_results(session_id);
CREATE INDEX IF NOT EXISTS idx_qtype       ON exp3_session_results(question_type);
"""

_llm_client: OpenAI | None = None


def _get_llm() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _llm_client


class SessionRunner:
    def __init__(
        self,
        tool_index: ToolIndex,
        db_path: str | Path,
        exp2_log_path: str | Path,
    ) -> None:
        self.tool_index   = tool_index
        self.s0_tokens    = s0_prompt_tokens()
        self.db_path      = Path(db_path)
        self.exp2_log     = Path(exp2_log_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.exp2_log.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(_DB_SCHEMA)
        conn.commit()
        conn.close()

    def _call_with_retry(self, call_fn, session_id: str):
        delays = [2, 4, 8]
        for attempt, delay in enumerate(delays, 1):
            try:
                return call_fn()
            except Exception as e:
                print(f"[retry] attempt {attempt} failed for {session_id} ({e}), retrying in {delay}s...")
                time.sleep(delay)
        try:
            return call_fn()
        except Exception as e:
            print(f"[retry] FAILED session {session_id} after {len(delays)+1} attempts: {e}")
            return None

    def run_s1b_session(
        self,
        question_id: str,
        question_type: str,
        query: str,
        profile: dict,
        rag_chunks: str = "",
        session_id: Optional[str] = None,
        generate_response: bool = True,
    ) -> dict:
        """Run one session through S1b (FSLSM-conditioned RAG-MCP).

        Logs Exp3 data. Optionally generates response and logs Exp2 passive data.
        """
        sid = session_id or str(uuid.uuid4())
        profile_lbl = profile_to_label(profile)

        # Step 1: S1b tool selection (FSLSM-augmented FAISS)
        aug_query = augment_query(query, profile)
        hits = self.tool_index.retrieve(aug_query, k=1)
        selected_tool, cosine_score = hits[0]
        s1b_tokens = selected_tool.token_cost

        # Step 2: Ground truth
        optimal_id = get_optimal_tool_id(question_type, profile)
        tsa_hit    = (selected_tool.tool_id == optimal_id)
        pts_delta  = (self.s0_tokens - s1b_tokens) / self.s0_tokens * 100

        # Step 3: Log Exp3 S1b result
        self._log_exp3(
            "S1b", sid, question_id, question_type, profile,
            query, selected_tool.tool_id, optimal_id,
            tsa_hit, s1b_tokens, pts_delta,
        )

        result = {
            "session_id":        sid,
            "question_id":       question_id,
            "question_type":     question_type,
            "profile_label":     profile_lbl,
            "selected_tool_id":  selected_tool.tool_id,
            "selected_tool_name": selected_tool.name,
            "optimal_tool_id":   optimal_id,
            "tsa_hit":           tsa_hit,
            "cosine_score":      cosine_score,
            "input_tokens":      s1b_tokens,
            "pts_delta":         pts_delta,
        }

        # Step 4: Generate response (optional — for Exp2 passive log)
        if generate_response:
            response = self._generate_response(
                query, profile, selected_tool.tool_id, rag_chunks, sid
            )
            result["response"] = response
            self._log_exp2(
                sid, question_id, question_type, profile_lbl,
                profile, query, selected_tool.tool_id, selected_tool.name, response,
            )

        return result

    def _generate_response(
        self, query: str, profile: dict, tool_id: int,
        rag_chunks: str, session_id: str = "",
    ) -> str:
        # Tool 14: Content Retriever — return RAG chunks directly
        if tool_id == 14:
            return f"Retrieved content:\n{rag_chunks}" if rag_chunks else "No content found."

        # Tool 15: Web Search — use Tavily (lazy import so tavily package is optional)
        if tool_id == 15:
            from experiments.exp3_revised.tools.tavily_search import web_search  # noqa: PLC0415
            return web_search(query)

        # Tools 1–13: LLM with tool-specific system prompt
        dims = decode_profile(profile)
        style_desc = ", ".join(sorted(dims))
        prompt = get_tool_prompt(tool_id, style_description=style_desc, target_style=style_desc)
        if not prompt:
            prompt = "You are a helpful ML tutor."

        context = (
            f"Context from D2L:\n{rag_chunks[:1500]}\n\nQuestion: {query}"
            if rag_chunks else f"Question: {query}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": context},
        ]

        def _call():
            return _get_llm().chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                max_tokens=800,
                timeout=60,
            )

        resp = self._call_with_retry(_call, session_id)
        if resp is None:
            return "[generation failed after retries]"
        return resp.choices[0].message.content

    def _log_exp3(
        self, condition: str, sid: str, qid: str, qtype: str,
        profile: dict, query: str, selected: int, optimal: int,
        tsa: bool, tokens: int, pts: float,
    ) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO exp3_session_results
               (condition, session_id, question_id, question_type,
                student_profile, query, selected_tool, optimal_tool,
                tsa_hit, input_tokens, pts_delta, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (condition, sid, qid, qtype, json.dumps(profile), query,
             selected, optimal, int(tsa), tokens, pts,
             datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def _log_exp2(
        self, sid: str, qid: str, qtype: str, profile_label: str,
        profile: dict, query: str, tool_id: int, tool_name: str, response: str,
    ) -> None:
        entry = {
            "session_id":        sid,
            "question_id":       qid,
            "question_type":     qtype,
            "mode":              "R2",
            "profile_label":     profile_label,
            "fslsm_vector":      profile,
            "query":             query,
            "selected_tool_id":  tool_id,
            "selected_tool_name": tool_name,
            "response":          response,
            "timestamp":         datetime.now().isoformat(),
        }
        with open(self.exp2_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
