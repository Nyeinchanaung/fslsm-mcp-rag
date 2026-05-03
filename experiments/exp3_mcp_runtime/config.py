from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent

DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"

AGENTS_PATH = REPO_ROOT / "data" / "agents" / "validated_agents.json"
CANONICAL_PROFILES_PATH = DATA_DIR / "canonical_profiles.json"
R2A_QUESTIONS_PATH = REPO_ROOT / "data" / "exp2" / "filtered_questions.json"
TRANSFER_QUESTIONS_PATH = R2A_QUESTIONS_PATH
R2B_QUESTIONS_PATH = DATA_DIR / "coverage_questions.json"
CORE_QUESTIONS_PATH = DATA_DIR / "exp3_core_questions.json"
CORE_ANSWER_KEY_PATH = DATA_DIR / "exp3_core_answer_key.json"
CORE_TOOL_SPECS_PATH = DATA_DIR / "exp3_core_tool_specs.json"
ALL_QUESTIONS_PATH = DATA_DIR / "all_questions.json"

CHUNKS_PATH = REPO_ROOT / "d2l" / "output" / "d2l_corpus_chunks.json"
FAISS_INDEX_PATH = REPO_ROOT / "data" / "processed" / "chunks" / "faiss.index"
FAISS_UIDS_PATH = REPO_ROOT / "data" / "processed" / "chunks" / "faiss.uids.txt"

TOOL_INDEX_PATH = DATA_DIR / "tool_index.faiss"
TOOL_META_PATH = DATA_DIR / "tool_index_meta.json"

RESULTS_DB_PATH = RESULTS_DIR / "exp3_runtime_results.db"
PASSIVE_R2_LOG_PATH = RESULTS_DIR / "exp2_r2_passive_log.jsonl"
METRICS_JSON_PATH = RESULTS_DIR / "exp3_runtime_metrics.json"
TABLE_MD_PATH = RESULTS_DIR / "exp3_runtime_table.md"

RESULTS_DB_FILENAME = "exp3_runtime_results.db"
PASSIVE_R2_LOG_FILENAME = "exp2_r2_passive_log.jsonl"
METRICS_JSON_FILENAME = "exp3_runtime_metrics.json"
TABLE_MD_FILENAME = "exp3_runtime_table.md"

DEMO_REPLAY_PATH = PASSIVE_R2_LOG_PATH
LEGACY_REPLAY_PATH = REPO_ROOT / "experiments" / "exp3_revised" / "results" / "exp2_passive_log.jsonl"

CORE_BENCHMARK_NAME = "exp3_core"
TRANSFER_BENCHMARK_NAME = "r2a_transfer"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

TOP_K_TOOL = 1
TOP_K_CHUNKS = 5
DRY_RUN_N = 12

S1B_MIN_LIFT_OVER_S1A = 0.05
MIN_PTS = 80.0
