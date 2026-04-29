"""
RR-Only Resume — bypasses SCS phase, computes only RR judge scores.

Resumes from rr_scores_checkpoint.json. Does not touch SCS, CR, ER, or final summary.
Use evaluate_exp2.py for the full pipeline once RR is complete.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp2_tutor_personalization.evaluate_exp2 import (
    R0_JSONL, R1_JSONL, compute_all_rr, load_sessions
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sessions_r0 = load_sessions(R0_JSONL)
    sessions_r1 = load_sessions(R1_JSONL)
    print(f"Sessions: R0={len(sessions_r0)}, R1={len(sessions_r1)}")

    print("\n--- Computing RR (R0) ---")
    compute_all_rr(sessions_r0, max_workers=1)

    print("\n--- Computing RR (R1) ---")
    compute_all_rr(sessions_r1, max_workers=1)

    print("\nRR phase complete. Run evaluate_exp2.py to assemble the full summary.")


if __name__ == "__main__":
    main()
