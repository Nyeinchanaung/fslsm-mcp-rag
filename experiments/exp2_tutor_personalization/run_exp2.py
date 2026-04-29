"""
Experiment 2 — A/B Runner (Phase 4)
====================================
Orchestrates tutoring sessions across all agents and questions in both
R0 (generic RAG) and R1 (FSLSM-personalized RAG) conditions.

Usage:
  python experiments/exp2_tutor_personalization/run_exp2.py --mode both
  python experiments/exp2_tutor_personalization/run_exp2.py --mode R0 --workers 3
  python experiments/exp2_tutor_personalization/run_exp2.py --dry-run --n 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.tutor.profile_agent import ProfileAgent
from src.tutor.retrieval_agent import RetrievalAgent
from src.tutor.tutor_agent import TutorAgent
from src.utils.llm_client import LLMClient

# DB imports (optional — gracefully skip if DB unavailable)
try:
    from db import get_session
    from db.models import Agent, EvalQuestion, ExperimentRun, InteractionLog
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Paths
AGENTS_PATH = PROJECT_ROOT / "data" / "agents" / "validated_agents.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "exp2" / "filtered_questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
R0_OUTPUT = RESULTS_DIR / "raw_sessions_r0.jsonl"
R1_OUTPUT = RESULTS_DIR / "raw_sessions_r1.jsonl"
LOG_FILE = RESULTS_DIR / "run_exp2.log"


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_agents(path: Path = AGENTS_PATH) -> list[dict]:
    with open(path) as f:
        agents = json.load(f)
    logger.info("Loaded %d agents from %s", len(agents), path)
    return agents


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path) as f:
        questions = json.load(f)
    logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


# -------------------------------------------------------------------
# Session list building
# -------------------------------------------------------------------

def build_session_list(
    agents: list[dict],
    questions: list[dict],
    modes: list[str],
    seed: int = 42,
) -> list[dict]:
    """
    Build the full list of sessions: agents x questions x modes.

    Each session dict is ready to pass to TutorAgent.run_session().
    """
    sessions = []
    for agent in agents:
        for question in questions:
            for mode in modes:
                sessions.append({
                    "agent_id": agent["agent_uid"],
                    "profile_label": agent["profile_label"],
                    "fslsm_vector": agent["fslsm_vector"],
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "gold_chunk_ids": question.get("gold_chunk_ids", []),
                    "essential_chunk_ids": question.get("essential_chunk_ids", []),
                    "gold_answer": question.get("gold_answer", ""),
                    "mode": mode,
                })
    random.seed(seed)
    random.shuffle(sessions)
    return sessions


# -------------------------------------------------------------------
# Checkpoint / Resume
# -------------------------------------------------------------------

def load_completed_keys(output_path: Path) -> set[tuple]:
    """Load (agent_id, question_id, mode) tuples from existing JSONL."""
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["agent_id"], rec["question_id"], rec["mode"])
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def filter_remaining(
    sessions: list[dict],
    completed: set[tuple],
) -> list[dict]:
    """Remove already-completed sessions."""
    return [
        s for s in sessions
        if (s["agent_id"], s["question_id"], s["mode"]) not in completed
    ]


# -------------------------------------------------------------------
# Session runner
# -------------------------------------------------------------------

def run_single_session(
    session: dict,
    tutor_agent: TutorAgent,
) -> dict:
    """Run a single session and return the result with metadata."""
    result = tutor_agent.run_session(session)
    # Attach fields not in TutorAgent output
    result["profile_label"] = session.get("profile_label", "")
    result["fslsm_vector"] = session["fslsm_vector"]
    result["question"] = session["question"]
    result["gold_chunk_ids"] = session.get("gold_chunk_ids", [])
    result["essential_chunk_ids"] = session.get("essential_chunk_ids", [])
    result["gold_answer"] = session.get("gold_answer", "")
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def append_result(result: dict, output_path: Path) -> None:
    """Append a single result as one JSONL line."""
    with open(output_path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------
# DB persistence
# -------------------------------------------------------------------

def create_experiment_run() -> int | None:
    """Create an ExperimentRun row and return its id. Returns None if DB unavailable."""
    if not DB_AVAILABLE:
        return None
    try:
        with get_session() as session:
            run = ExperimentRun(experiment="exp2", config={"mode": "R0_R1_AB"})
            session.add(run)
            session.flush()
            return run.id
    except Exception as e:
        logger.warning("DB: Could not create experiment run: %s", e)
        return None


def _resolve_agent_id(session, agent_uid: str) -> int | None:
    """Look up the DB agent id. Try model-agnostic first, then gpt-4.1-mini prefixed."""
    agent = session.query(Agent).filter(Agent.agent_uid == agent_uid).first()
    if agent:
        return agent.id
    # Try gpt-4.1-mini prefixed version (Exp1 agents)
    prefixed = f"gpt41mini_{agent_uid}"
    agent = session.query(Agent).filter(Agent.agent_uid == prefixed).first()
    return agent.id if agent else None


def _resolve_question_id(session, question_id: str) -> int | None:
    """Look up the DB eval_question id."""
    q = session.query(EvalQuestion).filter(
        EvalQuestion.question_id == question_id
    ).first()
    return q.id if q else None


def save_result_to_db(result: dict, run_id: int | None) -> None:
    """Persist a session result to the InteractionLog table."""
    if not DB_AVAILABLE or run_id is None:
        return
    try:
        with get_session() as session:
            agent_db_id = _resolve_agent_id(session, result["agent_id"])
            question_db_id = _resolve_question_id(session, result["question_id"])

            log = InteractionLog(
                run_id=run_id,
                agent_id=agent_db_id,
                question_id=question_db_id,
                system_variant=result["mode"],
                prompt_tokens=result.get("token_count"),
                latency_ms=result.get("latency_ms"),
                retrieved_chunk_ids=result.get("retrieved_chunk_ids"),
                tutor_response=result.get("response"),
                engagement=result.get("engagement_score"),
                chunk_recall_5=_compute_cr5(
                    result.get("retrieved_chunk_ids", []),
                    result.get("gold_chunk_ids", []),
                ),
                extra={
                    "agent_uid": result["agent_id"],
                    "question_uid": result["question_id"],
                    "system_prompt_used": result.get("system_prompt_used", ""),
                    "reformulated_query": result.get("reformulated_query", ""),
                    "tutor_cost": result.get("tutor_cost"),
                    "fslsm_vector": result.get("fslsm_vector"),
                    "profile_label": result.get("profile_label"),
                },
            )
            session.add(log)
    except Exception as e:
        logger.warning("DB: Could not save result: %s", e)


def _compute_cr5(retrieved: list, gold: list) -> float:
    """Chunk Recall@5 — top-5 only, matching metrics.py definition."""
    if not gold:
        return 0.0
    return len(set(retrieved[:5]) & set(gold)) / len(set(gold))


# -------------------------------------------------------------------
# Main orchestrator
# -------------------------------------------------------------------

def run_experiment(
    modes: list[str],
    max_workers: int = 5,
    dry_run: bool = False,
    n_limit: int | None = None,
):
    """Run the full A/B experiment."""
    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    agents = load_agents()
    questions = load_questions()

    # Build session list
    all_sessions = build_session_list(agents, questions, modes)
    logger.info("Total sessions planned: %d", len(all_sessions))

    # Load checkpoints
    completed = set()
    if not dry_run:
        for mode, path in [("R0", R0_OUTPUT), ("R1", R1_OUTPUT)]:
            if mode in modes:
                completed |= load_completed_keys(path)
        if completed:
            logger.info("Found %d already-completed sessions", len(completed))

    remaining = filter_remaining(all_sessions, completed)

    if n_limit is not None:
        # Sample matched pairs: pick ceil(n_limit/2) unique (agent, question)
        # combos and include both R0 and R1 for each → balanced matched pairs.
        import itertools
        pair_keys = list({(s["agent_id"], s["question_id"]) for s in remaining})
        random.seed(42)
        random.shuffle(pair_keys)
        n_pairs = (n_limit + 1) // 2  # ceil division
        selected_pairs = set(pair_keys[:n_pairs])
        remaining = [
            s for s in remaining
            if (s["agent_id"], s["question_id"]) in selected_pairs
        ]

    if not remaining:
        print("All sessions already completed. Nothing to do.")
        return

    print(f"Sessions: {len(remaining)} remaining "
          f"(of {len(all_sessions)} total, {len(completed)} completed)")

    if dry_run:
        print(f"[DRY RUN] Would run {len(remaining)} sessions. Exiting.")
        return

    # Create DB experiment run
    run_id = create_experiment_run()
    if run_id:
        logger.info("DB experiment run created: id=%d", run_id)
    else:
        logger.info("DB unavailable — results will be saved to JSONL only")

    # Initialize pipeline components
    profile_agent = ProfileAgent(
        profiles_path=PROJECT_ROOT / "data" / "fslsm" / "profiles.json"
    )
    decompose_client = LLMClient("gpt-4.1-mini", temperature=0.0)
    retrieval_agent = RetrievalAgent(decompose_client=decompose_client)
    tutor_client = LLMClient("gpt-4.1-mini", temperature=0.3)
    student_client = LLMClient("gpt-4.1-mini", temperature=0.0)

    tutor_agent = TutorAgent(
        tutor_client=tutor_client,
        student_client=student_client,
        profile_agent=profile_agent,
        retrieval_agent=retrieval_agent,
    )

    # Run sessions
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for session in remaining:
            future = executor.submit(run_single_session, session, tutor_agent)
            futures[future] = session

        with tqdm(total=len(remaining), desc="Exp2 Sessions") as pbar:
            for future in as_completed(futures):
                session = futures[future]
                try:
                    result = future.result()
                    output_path = R0_OUTPUT if result["mode"] == "R0" else R1_OUTPUT
                    append_result(result, output_path)
                    save_result_to_db(result, run_id)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    error_record = {
                        "agent_id": session["agent_id"],
                        "question_id": session["question_id"],
                        "mode": session["mode"],
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    output_path = R0_OUTPUT if session["mode"] == "R0" else R1_OUTPUT
                    # Log error to a separate file
                    error_path = RESULTS_DIR / "errors.jsonl"
                    with open(error_path, "a") as f:
                        f.write(json.dumps(error_record) + "\n")
                    logger.error(
                        "Session failed [%s / %s / %s]: %s",
                        session["agent_id"], session["question_id"],
                        session["mode"], e,
                    )
                pbar.update(1)

    print(f"\nComplete: {success_count} success, {error_count} errors, "
          f"{len(completed)} previously completed")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2 — A/B Runner (R0 vs R1 PersonaRAG)"
    )
    parser.add_argument(
        "--mode", choices=["R0", "R1", "both"], default="both",
        help="Which condition(s) to run (default: both)",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Max concurrent workers (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print session count and exit without running",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit to N sessions (for testing)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )

    modes = ["R0", "R1"] if args.mode == "both" else [args.mode]

    run_experiment(
        modes=modes,
        max_workers=args.workers,
        dry_run=args.dry_run,
        n_limit=args.n,
    )


if __name__ == "__main__":
    main()
