"""ILS evaluator — runs the 44-question questionnaire for virtual student agents."""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from config.constants import BASELINE_PROFILE_CODE, FSLSM_DIMENSIONS
from config.logging_config import logger
from db import get_session
from db.models import Agent, FslsmProfile
from src.agents.prompts.ils_answering import build_ils_question_prompt
from src.utils.helpers import extract_ab_choice
from src.utils.llm_client import LLMClient


def run_ils_for_agent(
    agent_info: dict,
    questions: list[dict],
    client: LLMClient,
    trial: int,
) -> tuple[dict[str, int], float]:
    """
    Run 44 ILS questions for a single agent.

    Args:
        agent_info: Dict with keys 'agent_uid', 'system_prompt',
                    'knowledge_level' (pre-extracted from ORM to avoid
                    detached session issues).

    Returns:
        (dim_scores, total_cost) where dim_scores maps each dimension
        to a score in [-11, +11].
    """
    agent_uid = agent_info["agent_uid"]
    system_prompt = agent_info["system_prompt"]
    knowledge_level = agent_info["knowledge_level"]

    dim_scores: dict[str, int] = {d: 0 for d in FSLSM_DIMENSIONS}
    raw: list[dict] = []
    total_cost = 0.0

    for q in questions:
        response = client.chat(
            system=system_prompt,
            user=build_ils_question_prompt(q),
            max_tokens=10,
        )
        total_cost += response.cost

        answer = extract_ab_choice(response.content)
        if answer:
            pole = q[f"option_{answer}"]["pole"]
            dim_scores[q["dimension"]] += pole
        else:
            logger.warning(
                "Could not parse a/b from agent %s q%d: %r",
                agent_uid, q["q_num"], response.content,
            )

        raw.append({
            "q_num": q["q_num"],
            "answer": answer,
            "raw_text": response.content.strip(),
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost,
        })

    # Save raw response file
    out_path = Path(f"results/exp1/raw_responses/{agent_uid}_trial{trial}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "agent_uid": agent_uid,
        "knowledge_level": knowledge_level,
        "model": client.model_name,
        "litellm_model": client.litellm_model,
        "trial": trial,
        "dim_scores": dim_scores,
        "total_cost_usd": total_cost,
        "raw": raw,
    }, indent=2))

    return dim_scores, total_cost


def run_experiment1(
    llm_model: str,
    num_trials: int = 3,
    temperature: float = 0.3,
) -> list[dict]:
    """
    Run ILS questionnaire across all 80 agents for a given model.

    No provider argument needed — LiteLLM resolves the provider
    from the model string via MODEL_REGISTRY in llm_client.py.
    """
    questions = json.loads(
        Path("data/fslsm/ils_questionnaire.json").read_text()
    )
    client = LLMClient(llm_model, temperature=temperature)

    # Load agents and their assigned profiles from DB (eagerly extract all fields)
    with get_session() as session:
        agents = (
            session.query(Agent)
            .join(FslsmProfile)
            .filter(
                Agent.llm_model == llm_model,
                FslsmProfile.profile_code != BASELINE_PROFILE_CODE,
            )
            .all()
        )
        agent_data = []
        for a in agents:
            agent_data.append({
                "agent_uid": a.agent_uid,
                "system_prompt": a.system_prompt,
                "instance_num": a.instance_num,
                "knowledge_level": a.knowledge_level,
                "profile_code": a.profile.profile_code,
                "assigned": {
                    "act_ref": a.profile.act_ref,
                    "sen_int": a.profile.sen_int,
                    "vis_ver": a.profile.vis_ver,
                    "seq_glo": a.profile.seq_glo,
                },
            })

    results: list[dict] = []
    cumulative_cost = 0.0
    skipped = 0

    for ad in tqdm(agent_data, desc=f"Exp1 [{llm_model}]"):
        for trial in range(1, num_trials + 1):
            # Resume support: skip if raw response already exists
            raw_path = Path(
                f"results/exp1/raw_responses/{ad['agent_uid']}_trial{trial}.json"
            )
            if raw_path.exists():
                cached = json.loads(raw_path.read_text())
                dim_scores = cached["dim_scores"]
                call_cost = cached["total_cost_usd"]
                skipped += 1
            else:
                dim_scores, call_cost = run_ils_for_agent(
                    ad, questions, client, trial,
                )
            cumulative_cost += call_cost

            # Binarize detection; ties (score == 0) → 0 (counted as mismatch)
            detected = {
                d: (1 if dim_scores[d] > 0 else (-1 if dim_scores[d] < 0 else 0))
                for d in FSLSM_DIMENSIONS
            }

            results.append({
                "agent_uid": ad["agent_uid"],
                "knowledge_level": ad["knowledge_level"],
                "profile_code": ad["profile_code"],
                "instance_num": ad["instance_num"],
                "trial": trial,
                "assigned": ad["assigned"],
                "detected": detected,
                "raw_scores": dim_scores,
                "cost_usd": call_cost,
            })

    if skipped:
        print(f"\n  Resumed: {skipped} agent-trials loaded from cache")
    print(f"\n  Total cost for {llm_model}: ${cumulative_cost:.4f}")
    return results


def run_baseline_experiment(
    llm_model: str,
    num_trials: int = 3,
    temperature: float = 0.3,
) -> list[dict]:
    """
    Run ILS questionnaire on 5 baseline agents for a given model.

    Baseline agents have no assigned FSLSM profile — their detected poles
    reveal the LLM's natural style tendencies. Results are used for
    PRA-vs-all-profiles analysis downstream.
    """
    questions = json.loads(
        Path("data/fslsm/ils_questionnaire.json").read_text()
    )
    client = LLMClient(llm_model, temperature=temperature)

    with get_session() as session:
        agents = (
            session.query(Agent)
            .join(FslsmProfile)
            .filter(
                Agent.llm_model == llm_model,
                FslsmProfile.profile_code == BASELINE_PROFILE_CODE,
            )
            .all()
        )
        agent_data = []
        for a in agents:
            agent_data.append({
                "agent_uid": a.agent_uid,
                "system_prompt": a.system_prompt,
                "instance_num": a.instance_num,
                "knowledge_level": a.knowledge_level,
                "profile_code": BASELINE_PROFILE_CODE,
            })

    results: list[dict] = []
    cumulative_cost = 0.0
    skipped = 0

    for ad in tqdm(agent_data, desc=f"Baseline [{llm_model}]"):
        for trial in range(1, num_trials + 1):
            raw_path = Path(
                f"results/exp1/raw_responses/{ad['agent_uid']}_trial{trial}.json"
            )
            if raw_path.exists():
                cached = json.loads(raw_path.read_text())
                dim_scores = cached["dim_scores"]
                call_cost = cached["total_cost_usd"]
                skipped += 1
            else:
                dim_scores, call_cost = run_ils_for_agent(
                    ad, questions, client, trial,
                )
            cumulative_cost += call_cost

            detected = {
                d: (1 if dim_scores[d] > 0 else (-1 if dim_scores[d] < 0 else 0))
                for d in FSLSM_DIMENSIONS
            }

            results.append({
                "agent_uid": ad["agent_uid"],
                "knowledge_level": ad["knowledge_level"],
                "profile_code": BASELINE_PROFILE_CODE,
                "instance_num": ad["instance_num"],
                "trial": trial,
                "detected": detected,
                "raw_scores": dim_scores,
                "cost_usd": call_cost,
            })

    if skipped:
        print(f"\n  Resumed: {skipped} baseline agent-trials loaded from cache")
    print(f"\n  Baseline cost for {llm_model}: ${cumulative_cost:.4f}")
    return results
