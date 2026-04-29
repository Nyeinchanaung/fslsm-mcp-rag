"""
Export 100 random sessions from Experiment 2 data.
Exp2 JSONL fields: agent_id, question, question_id, mode, fslsm_vector, profile_label, etc.

Output: diagnostics/exp2_sample_100_sessions.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import json
import random
from experiments.exp3_mcp_tool_selection.config import RAW_R0_PATH, RAW_R1_PATH, EXP_DIR

DIAGNOSTICS_DIR = EXP_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(exist_ok=True)

# Load from both runs and combine
all_sessions = []
for path, run_label in [(RAW_R0_PATH, "r0"), (RAW_R1_PATH, "r1")]:
    try:
        with open(path, 'r') as f:
            for line in f:
                try:
                    sess = json.loads(line.strip())
                    if not sess:
                        continue
                    all_sessions.append({
                        "run": run_label,
                        "agent_id": sess.get("agent_id"),
                        "question_id": sess.get("question_id"),
                        "query": sess.get("question"),   # Exp2 uses 'question' field
                        "mode": sess.get("mode"),
                        "profile_label": sess.get("profile_label"),
                        "fslsm_vector": sess.get("fslsm_vector"),
                    })
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"WARNING: {path} not found — skipping")

if len(all_sessions) == 0:
    print("ERROR: No sessions loaded. Check RAW_R0_PATH / RAW_R1_PATH in config.py")
    sys.exit(1)

random.shuffle(all_sessions)
sessions = all_sessions[:100]

output_path = DIAGNOSTICS_DIR / "exp2_sample_100_sessions.json"
with open(output_path, "w") as f:
    json.dump(sessions, f, indent=2)

print(f"✓ Exported {len(sessions)} Exp2 sessions to {output_path}")

query_lengths = [len(s["query"].split()) for s in sessions if s.get("query")]
if query_lengths:
    print(f"\n  Query stats:")
    print(f"    Avg length: {sum(query_lengths)/len(query_lengths):.1f} words")
    print(f"    Min length: {min(query_lengths)} words")
    print(f"    Max length: {max(query_lengths)} words")

modes = {}
for s in sessions:
    m = s.get("mode", "unknown")
    modes[m] = modes.get(m, 0) + 1
print(f"\n  Mode distribution: {modes}")
