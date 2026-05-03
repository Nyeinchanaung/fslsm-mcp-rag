from __future__ import annotations

import json
from functools import lru_cache

from experiments.exp3_mcp_runtime.config import AGENTS_PATH, CANONICAL_PROFILES_PATH


@lru_cache(maxsize=1)
def load_canonical_profiles() -> list[dict]:
    if CANONICAL_PROFILES_PATH.exists():
        return json.loads(CANONICAL_PROFILES_PATH.read_text())

    agents = json.loads(AGENTS_PATH.read_text())
    dedup: dict[str, dict] = {}
    for agent in agents:
        dedup.setdefault(
            agent["profile_label"],
            {
                "profile_label": agent["profile_label"],
                "profile_code": agent["profile_code"],
                "fslsm_vector": agent["fslsm_vector"],
            },
        )
    profiles = [dedup[label] for label in sorted(dedup)]
    CANONICAL_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANONICAL_PROFILES_PATH.write_text(json.dumps(profiles, indent=2))
    return profiles
