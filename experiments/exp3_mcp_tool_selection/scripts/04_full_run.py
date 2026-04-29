"""Full ablation: all available sessions × 3 conditions.

Truncates the DB at start so each run produces a clean result set. Re-run when
Exp2 expands its session count — no code change needed.
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
    logger.reset()

    sessions = list(iter_sessions())
    print(f"[full_run] {len(sessions)} sessions × 3 conditions = "
          f"{len(sessions) * 3} log rows\n")

    for sess in tqdm(sessions, desc="Full run"):
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
    print("\n── Full Run Summary ──")
    for cond in ("S0", "S1a", "S1b"):
        print(f"  {cond:<4}  TSA={tsa.get(cond,0)*100:.1f}%  PTS={pts.get(cond,0):.1f}%")
    print(f"\nResults saved → {RESULTS_DB_PATH}")
    logger.close()


if __name__ == "__main__":
    main()
