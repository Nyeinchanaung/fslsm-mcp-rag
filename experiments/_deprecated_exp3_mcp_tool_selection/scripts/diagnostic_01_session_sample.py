"""
Export 50 random S1b sessions with all relevant fields.
Output: diagnostics/exp3_sample_50_sessions.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import sqlite3
import json
from experiments.exp3_mcp_tool_selection.config import RESULTS_DB_PATH, EXP_DIR
from experiments.exp3_mcp_tool_selection.tool_registry import TOOL_REGISTRY

DIAGNOSTICS_DIR = EXP_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(exist_ok=True)

_tool_by_id = {t.tool_id: t for t in TOOL_REGISTRY}

def get_tool_by_id(tool_id):
    return _tool_by_id[tool_id]

conn = sqlite3.connect(RESULTS_DB_PATH)
cur = conn.cursor()

rows = cur.execute("""
    SELECT
        session_id,
        query,
        student_profile,
        selected_tool_id,
        optimal_tool_id,
        tsa_hit,
        input_tokens
    FROM exp3_session_results
    WHERE condition = 'S1b'
    ORDER BY RANDOM()
    LIMIT 50
""").fetchall()

sessions = []
for row in rows:
    sid, query, profile_json, selected_id, optimal_id, tsa_hit, tokens = row

    selected_tool = get_tool_by_id(selected_id)
    optimal_tool = get_tool_by_id(optimal_id)

    sessions.append({
        "session_id": sid,
        "query": query,
        "student_profile": json.loads(profile_json),
        "selected_tool": {
            "id": selected_id,
            "name": selected_tool.name,
            "fslsm_dims": list(selected_tool.fslsm_dims)
        },
        "optimal_tool": {
            "id": optimal_id,
            "name": optimal_tool.name,
            "fslsm_dims": list(optimal_tool.fslsm_dims),
            "description": optimal_tool.description
        },
        "tsa_hit": bool(tsa_hit),
        "input_tokens": tokens
    })

output_path = DIAGNOSTICS_DIR / "exp3_sample_50_sessions.json"
with open(output_path, "w") as f:
    json.dump(sessions, f, indent=2)

conn.close()

hits = sum(s['tsa_hit'] for s in sessions)
print(f"✓ Exported 50 sessions to {output_path}")
print(f"  TSA hits: {hits}/50 ({hits * 2}%)")
