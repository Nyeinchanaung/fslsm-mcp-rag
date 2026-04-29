"""
For 30 random S1a sessions, show:
- What FAISS retrieved (top-5 with scores)
- What the optimal tool was
- Whether optimal tool appears in top-5

Output: diagnostics/faiss_retrieval_diagnostics.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import os
import sqlite3
import json
from config.settings import settings
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

from experiments.exp3_mcp_tool_selection.config import RESULTS_DB_PATH, TOOL_INDEX_PATH, TOOL_META_PATH, EXP_DIR
from experiments.exp3_mcp_tool_selection.tool_index import ToolIndex
from experiments.exp3_mcp_tool_selection.tool_registry import TOOL_REGISTRY

DIAGNOSTICS_DIR = EXP_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(exist_ok=True)

_tool_by_id = {t.tool_id: t for t in TOOL_REGISTRY}

def get_tool_by_id(tool_id):
    return _tool_by_id[tool_id]

# Load FAISS index
idx = ToolIndex()
idx.load(TOOL_INDEX_PATH, TOOL_META_PATH)

conn = sqlite3.connect(RESULTS_DB_PATH)
rows = conn.execute("""
    SELECT query, optimal_tool_id, selected_tool_id
    FROM exp3_session_results
    WHERE condition = 'S1a'
    ORDER BY RANDOM()
    LIMIT 30
""").fetchall()

results = []
optimal_in_top5 = 0
optimal_in_rank1 = 0

for query, optimal_id, selected_id in rows:
    hits = idx.retrieve(query, k=5)

    optimal_rank = None
    for rank, (tool, score) in enumerate(hits, start=1):
        if tool.tool_id == optimal_id:
            optimal_rank = rank
            optimal_in_top5 += 1
            if rank == 1:
                optimal_in_rank1 += 1
            break

    entry = {
        "query": query,
        "optimal_tool": {
            "id": optimal_id,
            "name": get_tool_by_id(optimal_id).name,
            "rank_in_faiss": optimal_rank
        },
        "s1a_selected": {
            "id": selected_id,
            "name": get_tool_by_id(selected_id).name
        },
        "faiss_top5": [
            {
                "rank": rank,
                "tool_id": tool.tool_id,
                "tool_name": tool.name,
                "cosine_score": round(float(score), 4),
                "is_optimal": tool.tool_id == optimal_id
            }
            for rank, (tool, score) in enumerate(hits, start=1)
        ],
        "diagnosis": {
            "optimal_in_top5": optimal_rank is not None,
            "score_spread": round(float(hits[0][1] - hits[4][1]), 4) if len(hits) == 5 else None
        }
    }
    results.append(entry)

output_path = DIAGNOSTICS_DIR / "faiss_retrieval_diagnostics.json"
with open(output_path, "w") as f:
    json.dump({
        "summary": {
            "total_queries": len(results),
            "optimal_in_rank1": optimal_in_rank1,
            "optimal_in_top5": optimal_in_top5,
            "optimal_not_in_top5": len(results) - optimal_in_top5,
            "rank1_accuracy_pct": round(optimal_in_rank1 / len(results) * 100, 1),
            "top5_recall_pct": round(optimal_in_top5 / len(results) * 100, 1)
        },
        "queries": results
    }, f, indent=2)

conn.close()

print(f"✓ Exported FAISS diagnostics to {output_path}")
print(f"\n  Summary:")
print(f"    Optimal tool in rank 1:   {optimal_in_rank1}/30 ({optimal_in_rank1/30*100:.1f}%)")
print(f"    Optimal tool in top-5:    {optimal_in_top5}/30 ({optimal_in_top5/30*100:.1f}%)")
print(f"    Optimal NOT in top-5:     {30-optimal_in_top5}/30 ({(30-optimal_in_top5)/30*100:.1f}%)")
