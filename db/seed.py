"""Seed script — called by ingest and import scripts. Not run directly."""
from db.models import FslsmProfile
from db import get_session
import json
from pathlib import Path


def seed_profiles(profiles_json: str = "data/fslsm/profiles.json") -> int:
    """Insert 16 FSLSM profiles into DB. Skips existing profiles (idempotent)."""
    path = Path(profiles_json)
    if not path.exists():
        raise FileNotFoundError(f"Profiles JSON not found: {path}. Create it in Phase 1.")

    profiles = json.loads(path.read_text())

    inserted = 0
    with get_session() as session:
        existing_codes = {r[0] for r in session.query(FslsmProfile.profile_code).all()}
        for p in profiles:
            if p["profile_code"] in existing_codes:
                continue
            dims = p["dimensions"]
            session.add(FslsmProfile(
                profile_code=p["profile_code"],
                act_ref=dims["act_ref"],
                sen_int=dims["sen_int"],
                vis_ver=dims["vis_ver"],
                seq_glo=dims["seq_glo"],
                label=p.get("label"),
                description=p.get("style_descriptor_graf"),
                style_descriptor=p.get("style_descriptor_graf"),
                behavioral_instructions=p.get("behavioral_instructions"),
            ))
            inserted += 1

    print(f"Seeded {inserted} new profiles ({len(profiles) - inserted} already existed).")
    return inserted


def seed_eval_questions(questions_json: str = "data/exp2/sampled_questions.json") -> int:
    """Insert multi-hop eval questions into DB. Skips existing (idempotent)."""
    from db.models import EvalQuestion

    path = Path(questions_json)
    if not path.exists():
        raise FileNotFoundError(f"Questions JSON not found: {path}")

    questions = json.loads(path.read_text())

    inserted = 0
    with get_session() as session:
        existing_ids = {
            r[0] for r in session.query(EvalQuestion.question_id).all()
        }
        for q in questions:
            if q["question_id"] in existing_ids:
                continue
            session.add(EvalQuestion(
                question_id=q["question_id"],
                question=q["question"],
                gold_answer=q["gold_answer"],
                answer_type=q.get("question_type"),
                difficulty=q.get("difficulty"),
                quality_tier=q.get("quality_tier"),
                gold_chunk_ids=q["gold_chunk_ids"],
                essential_ids=q.get("essential_chunk_ids"),
            ))
            inserted += 1

    print(f"Seeded {inserted} new eval questions ({len(questions) - inserted} already existed).")
    return inserted
