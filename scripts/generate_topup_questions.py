"""
Step 9C — Targeted multi-hop question generation.

Generates new questions from D2L corpus chunks, requiring ≥3 gold_chunk_ids
in the LLM's own output so no separate re-annotation pass is needed.

Target: produce enough valid questions to bring the ≥3-gold-chunk pool to 72.

Usage:
    python scripts/generate_topup_questions.py
    python scripts/generate_topup_questions.py --target 40 --model gpt-4o-mini

Output: data/exp2/filtered_questions.json  (32 existing + new ones)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_PATH = PROJECT_ROOT / "d2l" / "output" / "d2l_corpus_chunks.json"
SAMPLED_PATH = PROJECT_ROOT / "data" / "exp2" / "sampled_questions.json"
FILTERED_PATH = PROJECT_ROOT / "data" / "exp2" / "filtered_questions.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "exp2" / "topup_checkpoint.json"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are generating evaluation questions for a multi-hop QA benchmark "
    "from the Dive into Deep Learning (D2L) textbook. "
    "Your output MUST be valid JSON. Do not include any text outside the JSON object."
)

GENERATION_PROMPT = """You are generating evaluation questions for a multi-hop QA benchmark from the D2L textbook.

Given these {n} related corpus chunks:

{chunk_texts}

Generate ONE question that:
1. REQUIRES synthesizing information from AT LEAST 3 of the {n} chunks to answer
2. Cannot be answered from any single chunk alone
3. Is of type: {question_type}
4. Has a gold answer that explicitly references content from each contributing chunk

Output a single JSON object with these exact fields:
{{
  "question": "...",
  "gold_answer": "...",
  "gold_chunk_ids": ["id1", "id2", "id3"],
  "essential_chunk_ids": ["id1"],
  "question_type": "{question_type}",
  "strategy": "intra_cluster"
}}

Rules:
- gold_chunk_ids MUST contain at least 3 IDs from the provided chunks
- essential_chunk_ids should contain 1-2 IDs without which the answer is impossible
- The question must be at least 50 characters long
- Do not add any fields beyond those listed above
"""

QUESTION_TYPES = [
    "compare",
    "trace_evolution",
    "explain_relationship",
    "synthesize_workflow",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_existing_qualified(path: Path) -> list[dict]:
    """Return questions from sampled_questions.json with ≥3 gold chunks."""
    with open(path) as f:
        qs = json.load(f)
    qualified = [q for q in qs if len(q.get("gold_chunk_ids", [])) >= 3]
    logger.info("Loaded %d qualified questions (≥3 gold chunks) from %s", len(qualified), path)
    return qualified


def cluster_chunks(chunks: list[dict], n_clusters: int = 25) -> list[int]:
    texts = [c.get("text", "") for c in chunks]
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(X).tolist()


def form_intra_cluster_groups(
    chunks: list[dict],
    labels: list[int],
    group_size: int = 5,
    seed: int = 42,
) -> list[list[dict]]:
    """Form groups of `group_size` chunks from the same cluster."""
    rng = random.Random(seed)
    from collections import defaultdict
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for chunk, label in zip(chunks, labels):
        by_cluster[label].append(chunk)

    groups = []
    for cluster_chunks_list in by_cluster.values():
        if len(cluster_chunks_list) < group_size:
            continue
        rng.shuffle(cluster_chunks_list)
        # Slide over cluster to form multiple non-overlapping groups
        for i in range(0, len(cluster_chunks_list) - group_size + 1, group_size):
            groups.append(cluster_chunks_list[i : i + group_size])

    rng.shuffle(groups)
    return groups


def parse_json_response(text: str) -> dict | None:
    """Extract the first JSON object from LLM output."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def is_valid(record: dict, allowed_chunk_ids: set[str]) -> bool:
    gold = record.get("gold_chunk_ids", [])
    essential = record.get("essential_chunk_ids", [])
    question = record.get("question", "")
    if len(gold) < 3:
        return False
    if len(essential) == 0:
        return False
    if len(question) < 50:
        return False
    # All declared chunk IDs must exist in the corpus
    if not all(cid in allowed_chunk_ids for cid in gold):
        return False
    if not all(cid in allowed_chunk_ids for cid in essential):
        return False
    return True


def load_checkpoint() -> list[dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        logger.info("Resumed checkpoint: %d generated so far", len(data))
        return data
    return []


def save_checkpoint(records: list[dict]) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=40, help="Number of new valid questions to generate")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Load data ---
    corpus = load_corpus(CORPUS_PATH)
    chunk_id_set = {c["chunk_id"] for c in corpus}
    existing_qualified = load_existing_qualified(SAMPLED_PATH)

    # --- How many new questions do we need? ---
    already_have = len(existing_qualified)
    need = max(0, 72 - already_have)
    target = max(args.target, need)
    logger.info("Have %d qualified questions; need %d more to reach 72 (target=%d)", already_have, need, target)

    # --- Resume checkpoint ---
    generated: list[dict] = load_checkpoint()
    existing_ids = {q["question_id"] for q in existing_qualified}

    # Assign starting ID
    all_ids = {q["question_id"] for q in existing_qualified}
    max_num = max((int(qid.replace("MH_", "")) for qid in all_ids if qid.startswith("MH_")), default=72)
    next_id = max_num + 1 + len(generated)

    if len(generated) >= target:
        logger.info("Checkpoint already has %d records — skipping generation.", len(generated))
    else:
        # --- Cluster corpus ---
        logger.info("Clustering %d chunks into %d clusters...", len(corpus), args.n_clusters)
        labels = cluster_chunks(corpus, n_clusters=args.n_clusters)
        groups = form_intra_cluster_groups(corpus, labels, group_size=5, seed=args.seed)
        logger.info("Formed %d candidate groups", len(groups))

        client = LLMClient(model=args.model, temperature=args.temperature)
        qt_cycle = QUESTION_TYPES * (len(groups) // len(QUESTION_TYPES) + 1)

        for i, group in enumerate(groups):
            if len(generated) >= target:
                break

            qt = qt_cycle[i]
            chunk_texts = "\n\n".join(
                f"[Chunk ID: {c['chunk_id']}]\nHeading: {c.get('heading', 'N/A')}\n{c['text'][:800]}"
                for c in group
            )
            prompt = GENERATION_PROMPT.format(
                n=len(group),
                chunk_texts=chunk_texts,
                question_type=qt,
            )

            try:
                resp = client.chat(system=SYSTEM_PROMPT, user=prompt, max_tokens=800)
                record = parse_json_response(resp.content)
            except Exception as e:
                logger.warning("Group %d: LLM call failed: %s", i, e)
                continue

            if record is None:
                logger.warning("Group %d: could not parse JSON response", i)
                continue

            if not is_valid(record, chunk_id_set):
                gold_count = len(record.get("gold_chunk_ids", []))
                logger.info(
                    "Group %d: rejected (gold=%d, q_len=%d)",
                    i, gold_count, len(record.get("question", "")),
                )
                continue

            qid = f"MH_{next_id:03d}"
            next_id += 1
            record["question_id"] = qid
            record["quality_tier"] = "silver"
            generated.append(record)
            save_checkpoint(generated)
            logger.info(
                "  [%d/%d] Accepted %s (%d gold chunks, type=%s)",
                len(generated), target, qid,
                len(record["gold_chunk_ids"]), record.get("question_type"),
            )

    # --- Combine and save ---
    final = existing_qualified + generated[:target]
    logger.info("Final dataset: %d questions (%d existing + %d new)", len(final), already_have, len(generated[:target]))

    # Verify
    assert len(final) >= 72, f"Need ≥72 questions, got {len(final)}"
    assert all(len(q["gold_chunk_ids"]) >= 3 for q in final), "Some questions have <3 gold chunks"
    assert all(len(q.get("essential_chunk_ids", [])) >= 1 for q in final), "Some questions missing essential chunks"

    with open(FILTERED_PATH, "w") as f:
        json.dump(final, f, indent=2)
    logger.info("✓ Saved %d questions to %s", len(final), FILTERED_PATH)

    # Distribution summary
    from collections import Counter
    qt_dist = Counter(q.get("question_type", "unknown") for q in final)
    gold_dist = Counter(len(q["gold_chunk_ids"]) for q in final)
    print(f"\n=== Final dataset: {len(final)} questions ===")
    print("Question type distribution:")
    for qt, n in qt_dist.most_common():
        print(f"  {qt}: {n}")
    print("Gold chunk distribution:")
    for k in sorted(gold_dist):
        print(f"  {k} chunks: {gold_dist[k]}")


if __name__ == "__main__":
    main()
