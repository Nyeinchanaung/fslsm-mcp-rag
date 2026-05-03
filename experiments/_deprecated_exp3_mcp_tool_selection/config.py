"""Experiment 3 constants — paths + thresholds.

The experiment computes TSA (tool selection accuracy) and PTS (prompt token
savings) across three conditions:
    S0  — prompt-bloat baseline (all 15 tool schemas in the prompt)
    S1a — unconditioned RAG-MCP (FAISS retrieval over query alone)
    S1b — FSLSM-conditioned RAG-MCP (FAISS over augmented query)

Reads OPENAI_API_KEY from the project-wide `config.settings` (pydantic-settings,
.env-backed). All paths are absolute to the experiment directory so scripts work
regardless of CWD.
"""
from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent

# Embedding model (matches text-embedding-3-small dim)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Source data (Exp2 reuse)
AGENTS_PATH = REPO_ROOT / "data" / "agents" / "validated_agents.json"
QUESTIONS_PATH = REPO_ROOT / "data" / "exp2" / "filtered_questions.json"
RAW_R0_PATH = REPO_ROOT / "experiments" / "exp2_tutor_personalization" / "results" / "raw_sessions_r0.jsonl"
RAW_R1_PATH = REPO_ROOT / "experiments" / "exp2_tutor_personalization" / "results" / "raw_sessions_r1.jsonl"

# Outputs
TOOL_INDEX_PATH = EXP_DIR / "tool_index.faiss"
TOOL_META_PATH = EXP_DIR / "tool_index_meta.json"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DB_PATH = RESULTS_DIR / "exp3_results.db"
METRICS_JSON_PATH = RESULTS_DIR / "exp3_metrics.json"
TABLE_MD_PATH = RESULTS_DIR / "exp3_table_3_4.md"
FIGURES_DIR = RESULTS_DIR / "figures"

# Experiment parameters
DRY_RUN_N = 50
TOP_K = 1

# Go/No-Go thresholds (from workplan §3.1)
MIN_TSA_S1B_OVER_S0 = 0.15
MIN_TSA_S1B_OVER_S1A = 0.05
MIN_PTS_S1B = 80.0
