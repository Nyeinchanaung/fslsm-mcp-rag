from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.config import (
    CANONICAL_PROFILES_PATH,
    CORE_ANSWER_KEY_PATH,
    CORE_QUESTIONS_PATH,
    CORE_TOOL_SPECS_PATH,
)
from experiments.exp3_mcp_runtime.core.core_dataset import (
    build_core_answer_key,
    build_core_questions,
    build_tool_specs_payload,
)
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles


if __name__ == "__main__":
    CORE_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORE_QUESTIONS_PATH.write_text(json.dumps(build_core_questions(), indent=2))
    CORE_ANSWER_KEY_PATH.write_text(json.dumps(build_core_answer_key(), indent=2))
    CORE_TOOL_SPECS_PATH.write_text(json.dumps(build_tool_specs_payload(), indent=2))
    CANONICAL_PROFILES_PATH.write_text(json.dumps(load_canonical_profiles(), indent=2))
    print(f"Wrote {CORE_QUESTIONS_PATH}")
    print(f"Wrote {CORE_ANSWER_KEY_PATH}")
    print(f"Wrote {CORE_TOOL_SPECS_PATH}")
    print(f"Wrote {CANONICAL_PROFILES_PATH}")
