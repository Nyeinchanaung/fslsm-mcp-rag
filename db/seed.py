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
