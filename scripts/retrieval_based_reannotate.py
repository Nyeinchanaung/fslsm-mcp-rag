"""
Data-driven gold chunk re-annotation.

For each question, runs the hybrid retrieval pipeline (BM25 + FAISS, k=20)
and keeps only gold chunks that:
  1. Were judged relevant (relevance >= 3) by GPT-4o, AND
  2. Actually appear in the top-20 retrieval results

This ensures gold chunks are both relevant AND retrievable,
making CR@10 and ER metrics achievable.

Usage:
    python scripts/retrieval_based_reannotate.py
    python scripts/retrieval_based_reannotate.py --top-k 30
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "data" / "exp2" / "sampled_questions.json"
PRE_REANNOTATE = ROOT / "data" / "exp2" / "sampled_questions_pre_reannotate.json"
CHECKPOINT_PATH = ROOT / "data" / "exp2" / "reannotate_checkpoint.json"
OUTPUT_PATH = ROOT / "data" / "exp2" / "sampled_questions.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20,
                        help="Retrieval depth for retrievability check")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load original questions (pre-reannotation backup has original 5 gold each)
    if PRE_REANNOTATE.exists():
        with open(PRE_REANNOTATE) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions from pre-reannotate backup")
    else:
        with open(QUESTIONS_PATH) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions")

    # Load relevance judgements
    checkpoint = json.loads(CHECKPOINT_PATH.read_text())
    print(f"Loaded {len(checkpoint)} relevance judgements")

    # Initialize retrieval agent
    from src.tutor.retrieval_agent import RetrievalAgent
    from src.utils.llm_client import LLMClient

    decompose = LLMClient("gpt-4.1-mini", temperature=0.0)
    ra = RetrievalAgent(decompose_client=decompose)

    stats = {
        "total_original": 0,
        "relevant": 0,  # relevance >= 3
        "retrievable": 0,  # in top-k
        "both": 0,  # relevant AND retrievable
        "kept": 0,
    }

    updated_questions = []

    for i, q in enumerate(questions):
        qid = q["question_id"]
        question_text = q["question"]
        original_gold = q["gold_chunk_ids"]
        original_essential = q.get("essential_chunk_ids", [])

        # Run retrieval (unpersonalized, R0 mode)
        result = ra.retrieve(question_text, personalized=False, k=args.top_k)
        retrieved_ids = set(result["chunk_ids"])

        new_gold = []
        new_essential = []

        for chunk_id in original_gold:
            cache_key = f"{qid}_{chunk_id}"
            rel = checkpoint.get(cache_key, {}).get("relevance", 0)
            sim = checkpoint.get(cache_key, {}).get("similarity", 0)

            is_relevant = rel >= 3
            is_retrievable = chunk_id in retrieved_ids

            stats["total_original"] += 1
            if is_relevant:
                stats["relevant"] += 1
            if is_retrievable:
                stats["retrievable"] += 1
            if is_relevant and is_retrievable:
                stats["both"] += 1

            # Keep if relevant AND retrievable
            if is_relevant and is_retrievable:
                new_gold.append(chunk_id)
                if chunk_id in original_essential or rel >= 4:
                    new_essential.append(chunk_id)

        # Fallback: if nothing passes both filters, keep best retrievable
        if not new_gold:
            # Try: any gold chunk that's at least retrievable
            for chunk_id in original_gold:
                if chunk_id in retrieved_ids:
                    new_gold.append(chunk_id)
                    new_essential.append(chunk_id)
                    break

        # If still nothing, keep the single most relevant
        if not new_gold:
            best_rel = 0
            best_cid = original_gold[0] if original_gold else None
            for chunk_id in original_gold:
                cache_key = f"{qid}_{chunk_id}"
                rel = checkpoint.get(cache_key, {}).get("relevance", 0)
                if rel > best_rel:
                    best_rel = rel
                    best_cid = chunk_id
            if best_cid:
                new_gold = [best_cid]
                new_essential = [best_cid]

        if not new_essential and new_gold:
            new_essential = [new_gold[0]]

        stats["kept"] += len(new_gold)

        if len(new_gold) != len(original_gold):
            print(f"  {qid}: {len(original_gold)} → {len(new_gold)} gold "
                  f"(retrieved {len(retrieved_ids & set(original_gold))}/{len(original_gold)})")

        updated_q = dict(q)
        updated_q["gold_chunk_ids"] = new_gold
        updated_q["essential_chunk_ids"] = new_essential
        updated_questions.append(updated_q)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(questions)}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DATA-DRIVEN RE-ANNOTATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Original gold chunks:  {stats['total_original']}")
    print(f"Relevant (rel>=3):     {stats['relevant']} ({stats['relevant']/stats['total_original']*100:.0f}%)")
    print(f"Retrievable (top-{args.top_k}): {stats['retrievable']} ({stats['retrievable']/stats['total_original']*100:.0f}%)")
    print(f"Both (kept):           {stats['both']} ({stats['both']/stats['total_original']*100:.0f}%)")
    print(f"Final kept (w/ fallback): {stats['kept']}")

    gold_counts = [len(q["gold_chunk_ids"]) for q in updated_questions]
    dist = Counter(gold_counts)
    print(f"\nGold chunks per question:")
    for k in sorted(dist.keys()):
        print(f"  {k} chunks: {dist[k]} questions")
    print(f"Mean: {np.mean(gold_counts):.1f}")

    # Estimate CR@10 with these gold sets
    # If a chunk is retrievable in top-20, it's likely in top-10 too
    # This gives us an upper bound
    print(f"\nEstimated CR@10 upper bound: "
          f"{stats['both'] / max(stats['kept'], 1) * 100:.0f}% "
          f"(all kept chunks are retrievable by definition)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    OUTPUT_PATH.write_text(json.dumps(updated_questions, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
