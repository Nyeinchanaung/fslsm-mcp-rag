"""
Import the validated multi-hop QA dataset into the eval_questions table.

Usage:
    python scripts/import_qa.py [--source PATH]

Default source: data/processed/ground_truth/d2l_multihop_ground_truth.json
Fallback:       d2l/multi_hop/output_validated/d2l_multihop_ground_truth.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.logging_config import logger
from db import get_session
from db.models import EvalQuestion

CANONICAL = ROOT / "data" / "processed" / "ground_truth" / "d2l_multihop_ground_truth.json"
FALLBACK  = ROOT / "d2l" / "multi_hop" / "output_validated" / "d2l_multihop_ground_truth.json"


def find_source(override: str | None = None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Specified source not found: {p}")
        return p
    if CANONICAL.exists():
        return CANONICAL
    if FALLBACK.exists():
        logger.info("Using fallback QA path: %s", FALLBACK)
        return FALLBACK
    raise FileNotFoundError(
        "QA JSON not found. Run the multi-hop generator first, then either:\n"
        f"  cp <output> {CANONICAL}\nor specify --source PATH"
    )


def import_qa(source: str | None = None, tiers: set[str] | None = None) -> int:
    """
    Import QA questions into DB.

    Args:
        source: Path override for the QA JSON.
        tiers:  Set of quality tiers to include. Default: {"gold", "silver"}.
                Pass {"gold", "silver", "reject"} to import all.

    Returns:
        Number of newly inserted rows.
    """
    if tiers is None:
        tiers = {"gold", "silver"}

    path = find_source(source)
    logger.info("Loading QA dataset from %s …", path)
    data = json.loads(path.read_text())

    questions: list[dict] = data.get("questions", [])
    logger.info("Total questions in file: %d", len(questions))

    # Filter by tier
    filtered = [q for q in questions if q.get("quality_tier", "silver") in tiers]
    logger.info("After tier filter (%s): %d questions", tiers, len(filtered))

    inserted = 0
    skipped  = 0
    with get_session() as session:
        existing_ids: set[str] = {
            r[0] for r in session.query(EvalQuestion.question_id).all()
        }

        for q in filtered:
            qid = q["question_id"]
            if qid in existing_ids:
                skipped += 1
                continue

            session.add(EvalQuestion(
                question_id   = qid,
                question      = q["question"],
                gold_answer   = q["gold_answer"],
                answer_type   = q.get("answer_type"),
                difficulty    = q.get("difficulty"),
                quality_tier  = q.get("quality_tier"),
                gold_chunk_ids= q["gold_chunk_ids"],
                essential_ids = q.get("essential_chunk_ids") or q.get("essential_ids"),
                supporting_ids= q.get("supporting_chunk_ids") or q.get("supporting_ids"),
            ))
            inserted += 1

    logger.info("Inserted %d questions, skipped %d (already existed).", inserted, skipped)
    print(f"\nImport complete: {inserted} new questions added ({skipped} already existed).")
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import multi-hop QA into eval_questions table.")
    parser.add_argument("--source", default=None, help="Path to QA JSON (overrides defaults)")
    parser.add_argument(
        "--include-rejects", action="store_true",
        help="Also import reject-tier questions (default: Gold+Silver only)"
    )
    args = parser.parse_args()

    tiers = {"gold", "silver", "reject"} if args.include_rejects else {"gold", "silver"}
    import_qa(source=args.source, tiers=tiers)
