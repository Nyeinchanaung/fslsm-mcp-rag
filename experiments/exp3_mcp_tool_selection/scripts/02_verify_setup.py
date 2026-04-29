"""Smoke test before running the dry run.

Checks:
  1. FAISS index loads and returns a sensible top-1 for a known query
  2. Augmentor produces a longer string than its input
  3. S0 token cost is non-trivial (> 1000 tokens)
  4. Exp2 session files exist and yield at least one session record
  5. Ground-truth selector returns a valid tool_id for a sample (query, profile)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from config.settings import settings  # noqa: E402
import os                              # noqa: E402

if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

from experiments.exp3_mcp_tool_selection.config import (  # noqa: E402
    RAW_R0_PATH,
    RAW_R1_PATH,
    TOOL_INDEX_PATH,
    TOOL_META_PATH,
)
from experiments.exp3_mcp_tool_selection.fslsm_query_augmentor import (  # noqa: E402
    FSLSMQueryAugmentor,
)
from experiments.exp3_mcp_tool_selection.session_adapter import iter_sessions  # noqa: E402
from experiments.exp3_mcp_tool_selection.tool_index import ToolIndex  # noqa: E402
from experiments.exp3_mcp_tool_selection.tool_registry import (  # noqa: E402
    get_optimal_tool_id,
    s0_prompt_tokens,
)


def check(label: str, ok: bool) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    return ok


def main() -> None:
    print("\n── Experiment 3 Setup Verification ──\n")
    results: list[bool] = []

    # 1. FAISS index
    try:
        idx = ToolIndex()
        idx.load(TOOL_INDEX_PATH, TOOL_META_PATH)
        hits = idx.retrieve("explain backpropagation visually with diagrams", k=1)
        tool, score = hits[0]
        results.append(check(
            f"FAISS top-1 'explain backprop visually' → "
            f"[{tool.tool_id}] {tool.name}  (cosine={score:.3f})",
            score > 0.2,
        ))
    except Exception as exc:
        results.append(check(f"FAISS index load: {exc}", False))

    # 2. Augmentor
    try:
        aug = FSLSMQueryAugmentor()
        profile = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
        out = aug.augment("Explain backpropagation", profile)
        results.append(check(
            f"Augmentor output length {len(out)} > input length 25",
            len(out) > len("Explain backpropagation"),
        ))
    except Exception as exc:
        results.append(check(f"Augmentor: {exc}", False))

    # 3. S0 token cost
    try:
        tokens = s0_prompt_tokens()
        results.append(check(
            f"S0 baseline token cost = {tokens} (target > 1000)",
            tokens > 1000,
        ))
    except Exception as exc:
        results.append(check(f"s0_prompt_tokens(): {exc}", False))

    # 4. Sessions readable
    try:
        first = next(iter_sessions(limit=1), None)
        ok = first is not None and "student_profile" in first and "query" in first
        results.append(check(
            f"Session adapter yields records — first session_id: "
            f"{first['session_id'] if first else 'NONE'}",
            ok,
        ))
    except Exception as exc:
        results.append(check(f"Session adapter: {exc}", False))

    # 5. Ground truth
    try:
        profile = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
        gt = get_optimal_tool_id("Explain backpropagation visually", profile)
        results.append(check(
            f"Ground truth selector returns valid tool_id={gt}",
            1 <= gt <= 15,
        ))
    except Exception as exc:
        results.append(check(f"Ground truth: {exc}", False))

    # 6. Source files (independent of adapter)
    results.append(check(
        f"raw_sessions_r0 exists at {RAW_R0_PATH.name}",
        RAW_R0_PATH.exists(),
    ))
    results.append(check(
        f"raw_sessions_r1 exists at {RAW_R1_PATH.name}",
        RAW_R1_PATH.exists(),
    ))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed.")
    if passed < total:
        print("Fix failures before running the dry run.\n")
        sys.exit(1)
    print("All checks passed. Ready for dry run (scripts/03_dry_run.py).\n")


if __name__ == "__main__":
    main()
