"""
Merge 72 original questions + 16 coverage questions → data/all_questions.json.
Run after manually reviewing coverage_questions.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_revised.config import QUESTIONS_PATH, COVERAGE_Q_PATH, ALL_Q_PATH

with open(QUESTIONS_PATH) as f:
    original = json.load(f)
with open(COVERAGE_Q_PATH) as f:
    coverage = json.load(f)

orig_ids = {q["question_id"] for q in original}
cov_ids  = {q["question_id"] for q in coverage}
assert not (orig_ids & cov_ids), "Duplicate question IDs found!"

unreviewed = [q for q in coverage if q.get("needs_review", True)]
if unreviewed:
    print(f"ERROR: {len(unreviewed)} coverage questions still need manual review!")
    print("  Edit data/coverage_questions.json and set needs_review=false for each.")
    sys.exit(1)

merged = original + coverage
ALL_Q_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(ALL_Q_PATH, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged: {len(original)} original + {len(coverage)} coverage = {len(merged)} total")
print(f"Saved → {ALL_Q_PATH}")
