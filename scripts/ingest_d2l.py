"""
Ingest the D2L corpus into PostgreSQL (chunks table) and build a FAISS index.

Usage:
    python scripts/ingest_d2l.py

Reads:  data/raw/d2l/d2l_corpus_chunks.json  (or d2l/output/d2l_corpus_chunks.json)
Writes: chunks → PostgreSQL, data/processed/chunks/faiss.index + faiss.uids.txt
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# Project root on sys.path (so imports work from any cwd)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.logging_config import logger
from config.settings import settings
from db import get_session
from db.models import Chunk, Textbook
from src.utils.embedding_client import EmbeddingClient

# ── Source paths ────────────────────────────────────────────────────────────
CANONICAL = ROOT / "data" / "raw" / "d2l" / "d2l_corpus_chunks.json"
FALLBACK   = ROOT / "d2l" / "output" / "d2l_corpus_chunks.json"

# ── Output paths ─────────────────────────────────────────────────────────────
INDEX_DIR  = ROOT / "data" / "processed" / "chunks"
INDEX_PATH = INDEX_DIR / "faiss.index"
UIDS_PATH  = INDEX_DIR / "faiss.uids.txt"

TEXTBOOK_NAME    = "Dive into Deep Learning"
TEXTBOOK_VERSION = "2.0"


def find_source() -> Path:
    if CANONICAL.exists():
        return CANONICAL
    if FALLBACK.exists():
        logger.info("Using fallback corpus path: %s", FALLBACK)
        return FALLBACK
    raise FileNotFoundError(
        "Corpus JSON not found. Expected at:\n"
        f"  {CANONICAL}\nor\n  {FALLBACK}"
    )


def get_or_create_textbook(session, name: str, version: str, source_path: str) -> Textbook:
    tb = session.query(Textbook).filter_by(name=name).first()
    if tb:
        logger.info("Textbook already exists (id=%d), skipping insert.", tb.id)
        return tb
    tb = Textbook(name=name, version=version, source_path=source_path)
    session.add(tb)
    session.flush()  # assigns tb.id
    logger.info("Created textbook '%s' (id=%d).", name, tb.id)
    return tb


def ingest(batch_size: int = 128) -> None:
    source = find_source()
    logger.info("Loading corpus from %s …", source)
    raw_chunks: list[dict] = json.loads(source.read_text())
    logger.info("Loaded %d chunks (before dedup).", len(raw_chunks))

    # Deduplicate by chunk_id — source JSON can contain repeated IDs
    seen: dict[str, dict] = {}
    for c in raw_chunks:
        if c["chunk_id"] not in seen:
            seen[c["chunk_id"]] = c
    raw_chunks = list(seen.values())
    logger.info("After dedup: %d unique chunks.", len(raw_chunks))

    # ── Embed all texts ───────────────────────────────────────────────────────
    ec = EmbeddingClient()
    texts = [c["text"] for c in raw_chunks]
    logger.info("Embedding %d chunks (batch_size=%d) …", len(texts), batch_size)
    embeddings: np.ndarray = ec.embed(texts, batch_size=batch_size)  # (N, 384), float32

    # ── Insert into PostgreSQL ────────────────────────────────────────────────
    logger.info("Inserting chunks into PostgreSQL …")
    with get_session() as session:
        tb = get_or_create_textbook(
            session, TEXTBOOK_NAME, TEXTBOOK_VERSION, str(source)
        )

        # Fetch already-inserted chunk_uids to support re-runs
        existing_uids: set[str] = {
            r[0] for r in session.query(Chunk.chunk_uid).filter_by(textbook_id=tb.id).all()
        }
        logger.info("%d chunks already in DB, inserting new ones …", len(existing_uids))

        new_rows: list[Chunk] = []
        for chunk, emb in zip(raw_chunks, embeddings):
            uid = chunk["chunk_id"]
            if uid in existing_uids:
                continue
            new_rows.append(Chunk(
                chunk_uid=uid,
                textbook_id=tb.id,
                chapter=chunk.get("chapter"),
                section=chunk.get("heading"),
                content=chunk["text"],
                embedding=emb.tolist(),
                token_count=len(chunk["text"].split()),
            ))

        if new_rows:
            # Batch insert
            for i in tqdm(range(0, len(new_rows), batch_size), desc="DB insert"):
                session.bulk_save_objects(new_rows[i : i + batch_size])
                session.flush()
            logger.info("Inserted %d new chunks into DB.", len(new_rows))
        else:
            logger.info("All chunks already present — skipping DB insert.")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Building FAISS IndexFlatIP over %d vectors …", len(embeddings))
    index = faiss.IndexFlatIP(settings.embed_dim)   # inner product on L2-normed = cosine
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(INDEX_PATH))
    logger.info("FAISS index saved → %s  (%d vectors)", INDEX_PATH, index.ntotal)

    # Save UID list for chunk_uid ↔ FAISS row mapping
    UIDS_PATH.write_text("\n".join(c["chunk_id"] for c in raw_chunks))
    logger.info("FAISS UID list saved → %s", UIDS_PATH)

    print(f"\nIngestion complete: {len(raw_chunks)} chunks in DB + FAISS index.")


if __name__ == "__main__":
    ingest()
