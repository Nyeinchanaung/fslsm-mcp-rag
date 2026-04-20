"""
Re-annotate gold chunks using LLM relevance judgement + embedding similarity.

For each question, scores each gold chunk on relevance (1-5 via GPT-4o)
and embedding similarity. Keeps only chunks that are genuinely relevant
and retrievable, reducing |gold| from 5 to ~2-3 per question.

Usage:
    python scripts/reannotate_gold_chunks.py
    python scripts/reannotate_gold_chunks.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.llm_client import LLMClient
from src.utils.embedding_client import EmbeddingClient

QUESTIONS_PATH = ROOT / "data" / "exp2" / "sampled_questions.json"
CORPUS_PATH = ROOT / "d2l" / "output" / "d2l_corpus_chunks.json"
OUTPUT_PATH = ROOT / "data" / "exp2" / "sampled_questions.json"
BACKUP_PATH = ROOT / "data" / "exp2" / "sampled_questions_pre_reannotate.json"
CHECKPOINT_PATH = ROOT / "data" / "exp2" / "reannotate_checkpoint.json"

RELEVANCE_PROMPT = """\
You are evaluating whether a textbook chunk is relevant to answering a student question.

## Question
{question}

## Chunk (ID: {chunk_id})
{chunk_text}

## Task
Rate how relevant this chunk is for answering the question above.

1 = Completely irrelevant — the chunk discusses unrelated topics
2 = Tangentially related — shares a keyword but doesn't help answer the question
3 = Partially relevant — covers a supporting concept needed for the answer
4 = Directly relevant — contains key information needed to answer the question
5 = Essential — the question cannot be properly answered without this chunk

Respond with ONLY a single integer (1-5)."""


def load_corpus(path: Path) -> dict[str, dict]:
    with open(path) as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def judge_relevance(
    question: str,
    chunk_id: str,
    chunk_text: str,
    judge: LLMClient,
) -> int:
    """Get GPT-4o relevance score (1-5) for a chunk."""
    import re
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = judge.chat(
                system="You are an expert relevance judge for educational content.",
                user=RELEVANCE_PROMPT.format(
                    question=question,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text[:2000],  # truncate very long chunks
                ),
                max_tokens=5,
                temperature=0.0,
            )
            text = resp.content.strip()
            if text in ("1", "2", "3", "4", "5"):
                return int(text)
            match = re.search(r"[1-5]", text)
            if match:
                return int(match.group())
            return 3  # default
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Warning: judge failed for {chunk_id}: {e}")
                return 3
    return 3


def compute_similarity(
    question: str,
    chunk_text: str,
    embedder: EmbeddingClient,
) -> float:
    """Compute cosine similarity between question and chunk."""
    q_emb = embedder.embed_one(question)
    c_emb = embedder.embed_one(chunk_text[:1000])  # truncate for embedding
    return float(np.dot(q_emb, c_emb))


def reannotate(dry_run: bool = False):
    questions = load_questions(QUESTIONS_PATH)
    corpus = load_corpus(CORPUS_PATH)
    judge = LLMClient("gpt-4o", temperature=0.0)
    embedder = EmbeddingClient()

    # Load checkpoint if exists
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text())
        print(f"Loaded {len(checkpoint)} cached judgements from checkpoint")

    # Backup original
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
        print(f"Backed up original to {BACKUP_PATH}")

    stats = {"total_gold": 0, "kept_gold": 0, "total_essential": 0, "kept_essential": 0}
    updated_questions = []

    for q in tqdm(questions, desc="Re-annotating"):
        qid = q["question_id"]
        question_text = q["question"]
        original_gold = q["gold_chunk_ids"]
        original_essential = q.get("essential_chunk_ids", [])

        scored_chunks = []
        for chunk_id in original_gold:
            chunk = corpus.get(chunk_id)
            if not chunk:
                print(f"  Warning: chunk {chunk_id} not found in corpus, skipping")
                continue

            cache_key = f"{qid}_{chunk_id}"
            if cache_key in checkpoint:
                relevance = checkpoint[cache_key]["relevance"]
                sim = checkpoint[cache_key]["similarity"]
            else:
                if dry_run:
                    relevance = 0
                    sim = 0.0
                else:
                    relevance = judge_relevance(
                        question_text, chunk_id, chunk.get("text", ""), judge
                    )
                    sim = compute_similarity(
                        question_text, chunk.get("text", ""), embedder
                    )
                    checkpoint[cache_key] = {
                        "relevance": relevance,
                        "similarity": round(sim, 4),
                    }
                    # Checkpoint every 10 judgements
                    if len(checkpoint) % 10 == 0:
                        CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))

            scored_chunks.append({
                "chunk_id": chunk_id,
                "relevance": relevance,
                "similarity": sim,
                "is_essential": chunk_id in original_essential,
            })

        # Filter: keep chunks with relevance >= 3 OR similarity >= 0.35
        new_gold = [
            sc["chunk_id"] for sc in scored_chunks
            if sc["relevance"] >= 3 or sc["similarity"] >= 0.35
        ]
        # Essential: keep from new_gold those with relevance >= 4
        new_essential = [
            sc["chunk_id"] for sc in scored_chunks
            if sc["chunk_id"] in new_gold and (sc["relevance"] >= 4 or sc["is_essential"])
        ]

        # Ensure at least 1 gold chunk (keep the highest-scored)
        if not new_gold and scored_chunks:
            best = max(scored_chunks, key=lambda x: (x["relevance"], x["similarity"]))
            new_gold = [best["chunk_id"]]
            new_essential = [best["chunk_id"]]

        # Ensure at least 1 essential chunk
        if not new_essential and new_gold:
            new_essential = [new_gold[0]]

        stats["total_gold"] += len(original_gold)
        stats["kept_gold"] += len(new_gold)
        stats["total_essential"] += len(original_essential)
        stats["kept_essential"] += len(new_essential)

        # Log changes
        if len(new_gold) != len(original_gold):
            dropped = set(original_gold) - set(new_gold)
            if not dry_run:
                details = {sc["chunk_id"]: f"rel={sc['relevance']} sim={sc['similarity']:.3f}"
                           for sc in scored_chunks}
                tqdm.write(
                    f"  {qid}: {len(original_gold)}→{len(new_gold)} gold "
                    f"(dropped: {', '.join(f'{c} ({details[c]})' for c in dropped)})"
                )

        updated_q = dict(q)
        updated_q["gold_chunk_ids"] = new_gold
        updated_q["essential_chunk_ids"] = new_essential
        updated_questions.append(updated_q)

    # Final checkpoint
    if not dry_run:
        CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RE-ANNOTATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Gold chunks:      {stats['total_gold']} → {stats['kept_gold']} "
          f"({stats['kept_gold']/stats['total_gold']*100:.0f}% kept)")
    print(f"Essential chunks:  {stats['total_essential']} → {stats['kept_essential']} "
          f"({stats['kept_essential']/stats['total_essential']*100:.0f}% kept)")

    # Distribution of new gold counts
    gold_counts = [len(q["gold_chunk_ids"]) for q in updated_questions]
    from collections import Counter
    dist = Counter(gold_counts)
    print(f"\nGold chunks per question distribution:")
    for k in sorted(dist.keys()):
        print(f"  {k} chunks: {dist[k]} questions")

    avg_gold = np.mean(gold_counts)
    print(f"\nMean gold chunks per question: {avg_gold:.1f} (was 5.0)")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write updated questions
    OUTPUT_PATH.write_text(json.dumps(updated_questions, indent=2, ensure_ascii=False))
    print(f"\nUpdated questions saved to {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Re-annotate gold chunks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reannotate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
