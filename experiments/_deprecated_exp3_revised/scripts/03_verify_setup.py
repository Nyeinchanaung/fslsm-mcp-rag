"""
Phase 3: Smoke tests for all components.
No API calls — verifies imports, data files, and ground truth coverage.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_revised.config import (
    AGENTS_PATH, QUESTIONS_PATH, TOOL_INDEX_PATH, TOOL_META_PATH,
)
from experiments.exp3_revised.tools.tool_registry import TOOL_REGISTRY, s0_prompt_tokens, registry_summary
from experiments.exp3_revised.core.profile_decoder import decode_profile, profile_to_label, profile_to_tuple
from experiments.exp3_revised.core.fslsm_augmentor import augment_query
from experiments.exp3_revised.core.ground_truth import verify_coverage, get_optimal_tool_id

PASS = "✓"
FAIL = "✗"

results = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    sym = PASS if condition else FAIL
    print(f"  [{sym}] {label}" + (f": {detail}" if detail else ""))
    results.append(condition)
    return condition


print("=== Exp3 Revised Setup Verification ===\n")

# 1. Tool registry
print("── Tool Registry ──")
check("15 tools registered", len(TOOL_REGISTRY) == 15, str(len(TOOL_REGISTRY)))
check("All tool_ids 1–15", all(1 <= t.tool_id <= 15 for t in TOOL_REGISTRY))
check("No duplicate tool_ids", len({t.tool_id for t in TOOL_REGISTRY}) == 15)
s0_tok = s0_prompt_tokens()
check("S0 token count ~1,200–1,600", 1000 < s0_tok < 2000, str(s0_tok))

# 2. Profile decoder
print("\n── Profile Decoder ──")
profile_bipolar = {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
dims = decode_profile(profile_bipolar)
check("Bipolar decode: Active-Sensing-Visual-Sequential",
      dims == {"Active", "Sensing", "Visual", "Sequential"}, str(dims))
label = profile_to_label(profile_bipolar)
check("profile_to_label correct", label == "Active-Sensing-Visual-Sequential", label)
tup = profile_to_tuple(profile_bipolar)
check("profile_to_tuple correct", tup == ("Active", "Sensing", "Visual", "Sequential"), str(tup))

profile_positive = {"act_ref": 1, "sen_int": 1, "vis_ver": 1, "seq_glo": 1}
dims2 = decode_profile(profile_positive)
check("Bipolar +1 decode: Reflective-Intuitive-Verbal-Global",
      dims2 == {"Reflective", "Intuitive", "Verbal", "Global"}, str(dims2))

# 3. Query augmentor
print("\n── Query Augmentor ──")
aug = augment_query("How does backpropagation work?", profile_bipolar)
check("Augmented query contains 'hands-on'", "hands-on" in aug)
check("Augmented query contains original query",
      "How does backpropagation work?" in aug)

# 4. Ground truth coverage
print("\n── Ground Truth ──")
ok = verify_coverage()
check("All 15 tools reachable", ok)
optimal = get_optimal_tool_id("explain_relationship", profile_bipolar)
check("explain_relationship + ActSenVisSeq → tool 4",
      optimal == 4, str(optimal))
optimal2 = get_optimal_tool_id("practice", profile_bipolar)
check("practice + ActSenVisSeq → tool 11", optimal2 == 11, str(optimal2))

# 5. Data files
print("\n── Data Files ──")
check("agents file exists",    AGENTS_PATH.exists(),    str(AGENTS_PATH))
check("questions file exists", QUESTIONS_PATH.exists(), str(QUESTIONS_PATH))
check("FAISS index exists",    TOOL_INDEX_PATH.exists(), str(TOOL_INDEX_PATH))
check("FAISS meta exists",     TOOL_META_PATH.exists(),  str(TOOL_META_PATH))

if AGENTS_PATH.exists():
    agents = json.loads(AGENTS_PATH.read_text())
    check("80 agents loaded", len(agents) == 80, str(len(agents)))

if QUESTIONS_PATH.exists():
    qs = json.loads(QUESTIONS_PATH.read_text())
    qtypes = {q["question_type"] for q in qs}
    check("72 questions loaded", len(qs) == 72, str(len(qs)))
    expected_qtypes = {"compare", "explain_relationship", "synthesize_workflow", "trace_evolution"}
    check("4 expected question_types present", qtypes == expected_qtypes, str(qtypes))

# Summary
print(f"\n{'═'*45}")
passed = sum(results)
total  = len(results)
if passed == total:
    print(f"✅ ALL {total} checks passed — ready to proceed")
else:
    print(f"❌ {total - passed}/{total} checks FAILED — fix before proceeding")
    sys.exit(1)
