"""Dry run: 50 sessions × 3 conditions, with go/no-go gate.

Truncates the results DB before each run so the gate reflects only this run.
Use scripts/04_full_run.py if you want to keep results across runs.
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

from tqdm import tqdm  # noqa: E402

from experiments.exp3_mcp_tool_selection.ablation_runner import (  # noqa: E402
    AblationRunner,
    ExperimentLogger,
)
from experiments.exp3_mcp_tool_selection.config import (  # noqa: E402
    DRY_RUN_N,
    MIN_PTS_S1B,
    MIN_TSA_S1B_OVER_S0,
    MIN_TSA_S1B_OVER_S1A,
    RESULTS_DB_PATH,
    TOOL_INDEX_PATH,
    TOOL_META_PATH,
)
from experiments.exp3_mcp_tool_selection.session_adapter import iter_sessions  # noqa: E402
from experiments.exp3_mcp_tool_selection.tool_index import ToolIndex  # noqa: E402


def main() -> None:
    idx = ToolIndex()
    idx.load(TOOL_INDEX_PATH, TOOL_META_PATH)
    runner = AblationRunner(tool_index=idx)
    logger = ExperimentLogger(RESULTS_DB_PATH)
    logger.reset()  # dry-run is idempotent

    sessions = list(iter_sessions(limit=DRY_RUN_N))
    print(f"[dry_run] {len(sessions)} sessions × 3 conditions = "
          f"{len(sessions) * 3} log rows\n")

    for sess in tqdm(sessions, desc="Dry run"):
        results = runner.run_session(
            profile=sess["student_profile"],
            query=sess["query"],
            session_id=sess["session_id"],
        )
        for r in results.values():
            logger.log(r)
    logger.commit()

    tsa = logger.tsa_by_condition()
    pts = logger.pts_by_condition()

    print("\n── Dry Run Results ──")
    for cond in ("S0", "S1a", "S1b"):
        t = tsa.get(cond, 0.0)
        p = pts.get(cond, 0.0)
        print(f"  {cond:<4}  TSA={t:.3f} ({t*100:.1f}%)  PTS={p:.1f}%")

    s1b_over_s0  = tsa.get("S1b", 0) - tsa.get("S0", 0)
    s1b_over_s1a = tsa.get("S1b", 0) - tsa.get("S1a", 0)
    pts_s1b      = pts.get("S1b", 0)

    print("\n── Go / No-Go Gate ──")
    go = True

    def gate(label: str, value: float, threshold: float, fmt: str = ".3f") -> None:
        nonlocal go
        ok = value >= threshold
        if not ok:
            go = False
        symbol = "GO" if ok else "NO-GO"
        print(f"  [{symbol}] {label}: {value:{fmt}} (threshold ≥ {threshold:{fmt}})")

    gate("S1b TSA − S0 TSA ", s1b_over_s0,  MIN_TSA_S1B_OVER_S0)
    gate("S1b TSA − S1a TSA", s1b_over_s1a, MIN_TSA_S1B_OVER_S1A)
    gate("PTS S1b (%)      ", pts_s1b,      MIN_PTS_S1B, ".1f")

    print()
    if go:
        print("ALL GATES PASSED — proceed to scripts/04_full_run.py")
    else:
        print("GATE FAILURE — diagnose before full run:")
        print("  - S1b TSA close to S0 TSA → broaden GROUND_TRUTH_MAP intent keywords")
        print("  - S1b TSA close to S1a TSA → strengthen DIM_DIRECTIVES vocabulary")
        print("  - PTS S1b low → check token_cost calculation")

    logger.close()
    sys.exit(0 if go else 2)


if __name__ == "__main__":
    main()
