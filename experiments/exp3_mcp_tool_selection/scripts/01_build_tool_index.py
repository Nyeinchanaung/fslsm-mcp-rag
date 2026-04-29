"""Embed the 15 tool descriptions and save the FAISS index.

One-time, ~5 s, ~$0.0001 in OpenAI credits. Re-run only if tool descriptions
change.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from config.settings import settings  # noqa: E402
import os                              # noqa: E402

# Surface OpenAI key from pydantic settings into the env for the OpenAI client.
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

from experiments.exp3_mcp_tool_selection.config import (  # noqa: E402
    EMBED_MODEL,
    TOOL_INDEX_PATH,
    TOOL_META_PATH,
)
from experiments.exp3_mcp_tool_selection.tool_index import ToolIndex  # noqa: E402


def main() -> None:
    idx = ToolIndex(embed_model=EMBED_MODEL)
    print(f"[build_tool_index] Embedding 15 tool descriptions with {EMBED_MODEL}...")
    idx.build()
    idx.save(TOOL_INDEX_PATH, TOOL_META_PATH)
    print(f"[build_tool_index] Saved index → {TOOL_INDEX_PATH}")
    print(f"[build_tool_index] Saved meta  → {TOOL_META_PATH}")


if __name__ == "__main__":
    main()
