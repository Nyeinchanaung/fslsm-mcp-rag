from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiments.exp3_mcp_runtime.config import (
    METRICS_JSON_FILENAME,
    PASSIVE_R2_LOG_FILENAME,
    RESULTS_DB_FILENAME,
    RUNS_DIR,
    TABLE_MD_FILENAME,
)


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    db_path: Path
    passive_log_path: Path
    metrics_path: Path
    table_path: Path


def make_run_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def get_run_artifacts(run_id: str | None = None, *, prefix: str = "run") -> RunArtifacts:
    resolved_run_id = run_id or make_run_id(prefix)
    run_dir = RUNS_DIR / resolved_run_id
    return RunArtifacts(
        run_id=resolved_run_id,
        run_dir=run_dir,
        db_path=run_dir / RESULTS_DB_FILENAME,
        passive_log_path=run_dir / PASSIVE_R2_LOG_FILENAME,
        metrics_path=run_dir / METRICS_JSON_FILENAME,
        table_path=run_dir / TABLE_MD_FILENAME,
    )
