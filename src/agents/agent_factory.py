"""Factory for creating 80 virtual student agents (16 profiles × 5 instances)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from config.constants import BASELINE_PROFILE_CODE, KNOWLEDGE_LEVEL_MAP
from config.logging_config import logger
from db import get_session
from db.models import Agent, FslsmProfile
from src.agents.prompts.student_system import (
    build_baseline_system_prompt,
    build_student_system_prompt,
)


def _model_tag(model_name: str) -> str:
    """Create a short filesystem-safe tag from a model name.

    e.g. 'gpt-4.1-mini' → 'gpt41mini', 'claude-sonnet-4-5-20251001' → 'claudes45'
    """
    tag = re.sub(r"[^a-z0-9]", "", model_name.lower())
    return tag[:12]  # keep it short


def create_agents(
    llm_model: str,
    profiles_json: str = "data/fslsm/profiles.json",
) -> int:
    """
    Create 80 agent records for a given LLM model.

    Instances 1-3 get knowledge levels (beginner/intermediate/advanced),
    instances 4-5 are general (no level). Idempotent: skips existing agent_uids.

    Returns the number of newly created agents.
    """
    profiles = json.loads(Path(profiles_json).read_text())
    mtag = _model_tag(llm_model)

    with get_session() as session:
        # Map profile_code → DB id
        db_profiles = {
            p.profile_code: p.id
            for p in session.query(FslsmProfile).all()
        }

        # Check existing agents for this model
        existing_uids = set(
            uid for (uid,) in session.query(Agent.agent_uid)
            .filter(Agent.llm_model == llm_model)
            .all()
        )

        created = 0
        for profile in profiles:
            code = profile["profile_code"]
            if code == BASELINE_PROFILE_CODE:
                continue  # baseline agents handled by create_baseline_agents()
            profile_id = db_profiles.get(code)
            if profile_id is None:
                logger.warning("Profile %s not found in DB, skipping", code)
                continue

            for instance in range(1, 6):
                knowledge_level = KNOWLEDGE_LEVEL_MAP[instance]
                level_abbrev = (
                    knowledge_level[:3] if knowledge_level else "gen"
                )
                agent_uid = f"{mtag}_{code}_I{instance:02d}_{level_abbrev}"

                if agent_uid in existing_uids:
                    continue

                system_prompt = build_student_system_prompt(
                    profile, knowledge_level=knowledge_level,
                )

                session.add(Agent(
                    agent_uid=agent_uid,
                    profile_id=profile_id,
                    instance_num=instance,
                    llm_model=llm_model,
                    knowledge_level=knowledge_level,
                    system_prompt=system_prompt,
                ))
                created += 1

        session.commit()

    logger.info(
        "Created %d agents for %s (skipped %d existing)",
        created, llm_model, 80 - created,
    )
    print(f"Created {created} agents for {llm_model} (48 leveled + 32 general)")
    return created


def create_baseline_agents(llm_model: str) -> int:
    """
    Create 5 Non-Personalized Baseline agents for a given LLM model.

    Same knowledge_level mapping as FSLSM agents (beg/int/adv/gen/gen).
    Profile FK points to P00_Baseline (all dimensions = 0).
    Idempotent: skips existing agent_uids.

    Returns the number of newly created agents.
    """
    mtag = _model_tag(llm_model)

    with get_session() as session:
        baseline_profile = (
            session.query(FslsmProfile)
            .filter_by(profile_code=BASELINE_PROFILE_CODE)
            .one()
        )

        existing_uids = set(
            uid for (uid,) in session.query(Agent.agent_uid)
            .filter(Agent.llm_model == llm_model)
            .all()
        )

        created = 0
        for instance in range(1, 6):
            knowledge_level = KNOWLEDGE_LEVEL_MAP[instance]
            level_abbrev = knowledge_level[:3] if knowledge_level else "gen"
            agent_uid = f"{mtag}_Baseline_I{instance:02d}_{level_abbrev}"

            if agent_uid in existing_uids:
                continue

            system_prompt = build_baseline_system_prompt(
                knowledge_level=knowledge_level,
            )

            session.add(Agent(
                agent_uid=agent_uid,
                profile_id=baseline_profile.id,
                instance_num=instance,
                llm_model=llm_model,
                knowledge_level=knowledge_level,
                system_prompt=system_prompt,
            ))
            created += 1

        session.commit()

    logger.info(
        "Created %d baseline agents for %s (skipped %d existing)",
        created, llm_model, 5 - created,
    )
    print(f"Created {created} baseline agents for {llm_model}")
    return created
