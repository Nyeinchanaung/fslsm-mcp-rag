"""
Analyze which tools are being assigned as optimal and how often.
Uses get_optimal_tool_id (intent+FSLSM scoring) — there is no flat GROUND_TRUTH_MAP.
Output: diagnostics/ground_truth_coverage.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import sqlite3
import json
from experiments.exp3_mcp_tool_selection.config import RESULTS_DB_PATH, EXP_DIR
from experiments.exp3_mcp_tool_selection.tool_registry import (
    TOOL_REGISTRY, INTENT_KEYWORDS, INTENT_CATEGORY_PRIORITY, get_optimal_tool_id
)

DIAGNOSTICS_DIR = EXP_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(exist_ok=True)

_tool_by_id = {t.tool_id: t for t in TOOL_REGISTRY}

conn = sqlite3.connect(RESULTS_DB_PATH)

# Distribution of optimal_tool_id across S1b sessions
rows = conn.execute("""
    SELECT optimal_tool_id, COUNT(*) as n
    FROM exp3_session_results
    WHERE condition = 'S1b'
    GROUP BY optimal_tool_id
    ORDER BY n DESC
""").fetchall()

total_sessions = sum(r[1] for r in rows)

tool_distribution = []
for tool_id, count in rows:
    tool = _tool_by_id[tool_id]
    tool_distribution.append({
        "tool_id": tool_id,
        "tool_name": tool.name,
        "fslsm_dims": list(tool.fslsm_dims),
        "count": count,
        "percentage": round(count / total_sessions * 100, 2)
    })

# Tools with zero assignment
assigned_ids = {r[0] for r in rows}
unassigned = [
    {"tool_id": t.tool_id, "tool_name": t.name, "fslsm_dims": list(t.fslsm_dims)}
    for t in TOOL_REGISTRY if t.tool_id not in assigned_ids
]

# Intent classification coverage: how often each intent fires (sample 200 queries)
sample_rows = conn.execute("""
    SELECT query, student_profile
    FROM exp3_session_results
    WHERE condition = 'S1b'
    ORDER BY RANDOM()
    LIMIT 200
""").fetchall()

intent_counts: dict[str, int] = {}
for query, profile_json in sample_rows:
    q = query.lower()
    matched = "explain"  # default
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in q for k in kws):
            matched = intent
            break
    intent_counts[matched] = intent_counts.get(matched, 0) + 1

conn.close()

output_path = DIAGNOSTICS_DIR / "ground_truth_coverage.json"
with open(output_path, "w") as f:
    json.dump({
        "summary": {
            "total_sessions": total_sessions,
            "unique_tools_assigned": len(rows),
            "total_tools": len(TOOL_REGISTRY),
            "unassigned_tools": len(unassigned)
        },
        "tool_distribution": tool_distribution,
        "unassigned_tools": unassigned,
        "intent_distribution_sample200": intent_counts,
        "intent_keywords": INTENT_KEYWORDS,
        "intent_category_priority": INTENT_CATEGORY_PRIORITY
    }, f, indent=2)

print(f"✓ Exported ground truth coverage to {output_path}")
print(f"\n  Summary:")
print(f"    Total sessions: {total_sessions}")
print(f"    Unique tools assigned: {len(rows)}/{len(TOOL_REGISTRY)}")
print(f"    Unassigned tools: {len(unassigned)}")

print(f"\n  Top 5 most-assigned tools:")
for item in tool_distribution[:5]:
    flag = " ⚠️  DOMINANT" if item["percentage"] > 30 else ""
    print(f"    [{item['tool_id']:2d}] {item['tool_name']:<35} {item['percentage']:5.1f}%{flag}")

if tool_distribution[0]["percentage"] > 50:
    print(f"\n  ⚠️  WARNING: Tool {tool_distribution[0]['tool_id']} dominates with {tool_distribution[0]['percentage']:.1f}%")
    print(f"      This suggests intent classification falls back to 'explain' for most queries.")

print(f"\n  Intent distribution (sample 200):")
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"    {intent:<12} {count:3d}  ({count/2:.0f}%)")
