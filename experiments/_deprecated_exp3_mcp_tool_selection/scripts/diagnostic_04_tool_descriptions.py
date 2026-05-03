"""
Export all 15 tool descriptions with metadata and overlap analysis.
Output: diagnostics/tool_descriptions.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import json
from experiments.exp3_mcp_tool_selection.tool_registry import TOOL_REGISTRY
from experiments.exp3_mcp_tool_selection.config import EXP_DIR

DIAGNOSTICS_DIR = EXP_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(exist_ok=True)

tools_export = [
    {
        "tool_id": t.tool_id,
        "name": t.name,
        "category": t.category,
        "fslsm_dims": list(t.fslsm_dims),
        "description": t.description,
        "description_word_count": len(t.description.split()),
        "token_cost": t.token_cost
    }
    for t in TOOL_REGISTRY
]

# Pairwise word-overlap analysis
word_overlap = {}
for i, t1 in enumerate(TOOL_REGISTRY):
    for j, t2 in enumerate(TOOL_REGISTRY):
        if i >= j:
            continue
        words1 = set(t1.description.lower().split())
        words2 = set(t2.description.lower().split())
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        if overlap > 0.5:
            pair_key = f"{t1.tool_id}-{t2.tool_id}"
            word_overlap[pair_key] = {
                "tool1": t1.name,
                "tool2": t2.name,
                "overlap_pct": round(overlap * 100, 1)
            }

output_path = DIAGNOSTICS_DIR / "tool_descriptions.json"
with open(output_path, "w") as f:
    json.dump({
        "tools": tools_export,
        "analysis": {
            "total_tools": len(tools_export),
            "avg_description_length": round(
                sum(t["description_word_count"] for t in tools_export) / len(tools_export), 1
            ),
            "high_overlap_pairs": word_overlap
        }
    }, f, indent=2)

print(f"✓ Exported tool descriptions to {output_path}")
print(f"\n  Analysis:")
print(f"    Total tools: {len(tools_export)}")
print(f"    Avg description length: {sum(t['description_word_count'] for t in tools_export) / len(tools_export):.1f} words")
print(f"    Tool pairs with >50% word overlap: {len(word_overlap)}")
if word_overlap:
    print(f"\n  High-overlap pairs (these may confuse FAISS):")
    for pair, info in sorted(word_overlap.items(), key=lambda x: -x[1]["overlap_pct"])[:5]:
        print(f"    {info['tool1']} ↔ {info['tool2']}: {info['overlap_pct']}%")
