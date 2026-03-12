"""
Seed the 16 FSLSM profiles into the fslsm_profiles table.

Usage:
    python scripts/create_profiles.py

Reads:  data/fslsm/profiles.json
Writes: fslsm_profiles table (idempotent — skips existing profiles)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.logging_config import logger
from db.seed import seed_profiles

if __name__ == "__main__":
    profiles_json = str(ROOT / "data" / "fslsm" / "profiles.json")
    logger.info("Seeding FSLSM profiles from %s …", profiles_json)
    inserted = seed_profiles(profiles_json)
    print(f"Done. {inserted} profile(s) inserted.")
