from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.config import METRICS_JSON_PATH, TABLE_MD_PATH
from experiments.exp3_mcp_runtime.core.run_artifacts import get_run_artifacts


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Markdown Exp3 metrics report.")
    parser.add_argument("--run-id", default=None, help="Run folder id under results/runs/.")
    parser.add_argument("--metrics-path", default=None, help="Explicit metrics JSON path. Overrides --run-id.")
    parser.add_argument("--table-path", default=None, help="Explicit Markdown report output path.")
    args = parser.parse_args()

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        table_path = Path(args.table_path) if args.table_path else TABLE_MD_PATH
    elif args.run_id:
        artifacts = get_run_artifacts(args.run_id)
        metrics_path = artifacts.metrics_path
        table_path = Path(args.table_path) if args.table_path else artifacts.table_path
    else:
        metrics_path = METRICS_JSON_PATH
        table_path = Path(args.table_path) if args.table_path else TABLE_MD_PATH

    metrics = json.loads(metrics_path.read_text())
    lines = ["# Exp3 MCP Runtime Report", ""]
    lines.append(f"Source DB: `{metrics.get('source_db', 'unknown')}`")
    lines.append("")
    for benchmark, payload in metrics["benchmarks"].items():
        lines.extend(
            [
                f"## {benchmark}",
                "",
                "| Condition | n | Task-TSA | Profile-TSA All | Profile-TSA Eligible | Profile Eligible n | PTS | Exec Success | Latency ms | Grounded Output |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, values in payload["conditions"].items():
            lines.append(
                f"| {condition} | {values['n']} | {_fmt(values.get('task_tsa'))} | "
                f"{_fmt(values.get('profile_tsa_all'))} | {_fmt(values.get('profile_tsa_eligible'))} | "
                f"{values.get('profile_eligible_n', 0)} | {_fmt(values.get('pts'), 1)} | "
                f"{_fmt(values.get('execution_success_rate'))} | {_fmt(values.get('latency_ms'), 1)} | "
                f"{_fmt(values.get('grounded_tool_output_rate'))} |"
            )
        if payload["paired"]:
            lines.extend(
                [
                    "",
                    "| Paired Comparison | Task-TSA Delta | Profile-TSA All Delta | Profile-TSA Eligible Delta | PTS Delta | Latency Delta ms | N Pairs | Profile Eligible Pairs |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for label, values in payload["paired"].items():
                lines.append(
                    f"| {label} | {_fmt(values.get('task_tsa_delta'))} | "
                    f"{_fmt(values.get('profile_tsa_all_delta'))} | "
                    f"{_fmt(values.get('profile_tsa_eligible_delta'))} | "
                    f"{_fmt(values.get('pts_delta'), 1)} | {_fmt(values.get('latency_ms_delta'), 1)} | "
                    f"{values.get('n_pairs', 0)} | {values.get('profile_eligible_n_pairs', 0)} |"
                )
        lines.append("")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines))
    print(table_path.read_text())
