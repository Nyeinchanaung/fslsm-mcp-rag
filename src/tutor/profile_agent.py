"""
Phase 1 — Profile Agent for Experiment 2 (FSLSM-Based Tutor Personalization).

Translates a binary FSLSM vector [-1|+1, -1|+1, -1|+1, -1|+1] into a
natural-language reasoning plan that downstream agents (Retrieval, Tutor)
use to shape retrieval and generation.

Source: Graf, Viola, Leo & Kinshuk (2007) dimension descriptors.
"""

import json
from pathlib import Path

REQUIRED_DIMS = {"act_ref", "sen_int", "vis_ver", "seq_glo"}
VALID_VALUES = {-1, 1}

# ---------------------------------------------------------------------------
# Dimension-level directives (combined per-profile in FSLSM_STYLE_MAP)
# ---------------------------------------------------------------------------

_DIM_RETRIEVAL = {
    "act_ref": {
        -1: "interactive exercises, hands-on worked examples, collaborative activities",
        1: "analytical summaries, reflective case studies, observation-based explanations",
    },
    "sen_int": {
        -1: "concrete facts, specific numerical examples, established procedures",
        1: "abstract concepts, theoretical frameworks, general principles",
    },
    "vis_ver": {
        -1: "diagrams, charts, flow diagrams, visual walkthroughs, spatial representations",
        1: "written explanations, narrative prose, verbal definitions, textual analogies",
    },
    "seq_glo": {
        -1: "step-by-step procedures, linear walkthroughs, sequentially structured content",
        1: "conceptual overviews, big-picture summaries, holistic framework introductions",
    },
}

_DIM_GENERATION = {
    "act_ref": {
        -1: (
            "ACTIVE LEARNING STYLE:\n"
            "- Provide the complete factual explanation first.\n"
            "- At the END of your response, ADDITIONALLY add a short hands-on exercise\n"
            "  or 'Try it yourself' prompt (2–4 sentences maximum).\n"
            "- Use phrases like 'Now try...', 'Your turn:', 'Experiment with...'.\n"
            "- The practice prompt is an appendix to the explanation, not a replacement."
        ),
        1: (
            "REFLECTIVE LEARNING STYLE — You MUST:\n"
            "- Pose reflective questions ('Why do you think...?', 'Consider what happens if...')\n"
            "- Include a 'Think About It' or 'Reflection' section\n"
            "- Encourage the student to analyze and compare before acting"
        ),
    },
    "sen_int": {
        -1: (
            "SENSING LEARNING STYLE — You MUST:\n"
            "- Include specific numerical examples with concrete values\n"
            "- Reference real-world applications or practical use cases\n"
            "- Show actual computations, not just formulas"
        ),
        1: (
            "INTUITIVE LEARNING STYLE — You MUST:\n"
            "- Lead with the underlying theory or principle before examples\n"
            "- Explain the 'why' behind each concept\n"
            "- Connect concepts to broader theoretical frameworks"
        ),
    },
    "vis_ver": {
        -1: (
            "VISUAL LEARNING STYLE:\n"
            "- First provide a complete written explanation covering all key facts.\n"
            "- Then ADDITIONALLY include at least one ASCII diagram, markdown table,\n"
            "  or structured visual that illustrates the relationships.\n"
            "- The diagram supplements the explanation — it does not replace written content.\n"
            "- Use visual metaphors and spatial language throughout."
        ),
        1: (
            "VERBAL LEARNING STYLE — You MUST:\n"
            "- Use rich narrative explanations and analogies\n"
            "- Explain concepts through storytelling or verbal walkthroughs\n"
            "- Avoid diagrams; prefer detailed written descriptions"
        ),
    },
    "seq_glo": {
        -1: (
            "SEQUENTIAL LEARNING STYLE:\n"
            "- Structure your response as numbered steps (Step 1, Step 2, Step 3...).\n"
            "- Each step must contain a FULL explanation of that concept — do not\n"
            "  abbreviate factual content to fit the step structure.\n"
            "- Use transitional phrases ('Now that we understand X, let's move to Y').\n"
            "- All key facts from the evidence must appear within the steps."
        ),
        1: (
            "GLOBAL LEARNING STYLE — You MUST:\n"
            "- Start with a big-picture overview paragraph before any details\n"
            "- Use a 'Summary First' structure: conclusion then supporting details\n"
            "- Connect each concept back to the overall framework"
        ),
    },
}

_DIM_RERANK_BIAS = {
    "act_ref": {-1: ["exercises", "interactive"], 1: ["reflective", "analytical"]},
    "sen_int": {-1: ["concrete_examples", "procedures"], 1: ["abstract_theory", "principles"]},
    "vis_ver": {-1: ["visual_content", "diagrams"], 1: ["verbal_text", "definitions"]},
    "seq_glo": {-1: ["step_by_step", "sequential"], 1: ["conceptual_overview", "holistic"]},
}

_DIM_DEPRIORITIZE = {
    "act_ref": {-1: ["passive_reading"], 1: ["exercises"]},
    "sen_int": {-1: ["abstract_theory"], 1: ["concrete_examples"]},
    "vis_ver": {-1: ["verbal_text"], 1: ["visual_content"]},
    "seq_glo": {-1: ["conceptual_overview"], 1: ["step_by_step"]},
}

# ---------------------------------------------------------------------------
# FSLSM_STYLE_MAP — all 16 binary profiles
# Built programmatically from per-dimension tables above, then enriched with
# profile metadata from data/fslsm/profiles.json at load time.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Factual anchor injected before style directives in every R1 system prompt.
# Guarantees the judge finds core claims regardless of structural formatting.
# ---------------------------------------------------------------------------
_CONTENT_ANCHOR = (
    "## STEP 0 — FACTUAL FOUNDATION (always do this first)\n"
    "Before applying any style formatting, write 2–3 sentences that directly "
    "answer the question using the key facts from the retrieved evidence. "
    "This factual core MUST appear at the start of your response. "
    "Style formatting is built around this foundation, not instead of it.\n"
)

_LABEL_PARTS = {
    "act_ref": {-1: "Active", 1: "Reflective"},
    "sen_int": {-1: "Sensing", 1: "Intuitive"},
    "vis_ver": {-1: "Visual", 1: "Verbal"},
    "seq_glo": {-1: "Sequential", 1: "Global"},
}

_CODE_PARTS = {
    "act_ref": {-1: "Act", 1: "Ref"},
    "sen_int": {-1: "Sen", 1: "Int"},
    "vis_ver": {-1: "Vis", 1: "Ver"},
    "seq_glo": {-1: "Seq", 1: "Glo"},
}


def _build_style_map() -> dict:
    """Build the 16-entry FSLSM_STYLE_MAP from per-dimension tables."""
    style_map = {}
    idx = 0
    for ar in (-1, 1):
        for si in (-1, 1):
            for vv in (-1, 1):
                for sg in (-1, 1):
                    idx += 1
                    vec = {"act_ref": ar, "sen_int": si, "vis_ver": vv, "seq_glo": sg}
                    key = (ar, si, vv, sg)

                    label = "-".join(
                        _LABEL_PARTS[d][vec[d]] for d in ("act_ref", "sen_int", "vis_ver", "seq_glo")
                    )
                    code = "".join(
                        _CODE_PARTS[d][vec[d]] for d in ("act_ref", "sen_int", "vis_ver", "seq_glo")
                    )
                    profile_code = f"P{idx:02d}_{code}"

                    retrieval_directive = "Retrieve chunks containing " + "; ".join(
                        _DIM_RETRIEVAL[d][vec[d]] for d in ("act_ref", "sen_int", "vis_ver", "seq_glo")
                    ) + "."

                    generation_directive = "\n\n".join(
                        _DIM_GENERATION[d][vec[d]] for d in ("act_ref", "sen_int", "vis_ver", "seq_glo")
                    )

                    reranking_bias = []
                    for d in ("act_ref", "sen_int", "vis_ver", "seq_glo"):
                        reranking_bias.extend(_DIM_RERANK_BIAS[d][vec[d]])

                    deprioritize = []
                    for d in ("act_ref", "sen_int", "vis_ver", "seq_glo"):
                        deprioritize.extend(_DIM_DEPRIORITIZE[d][vec[d]])

                    style_map[key] = {
                        "profile_code": profile_code,
                        "style_label": label,
                        "retrieval_directive": retrieval_directive,
                        "generation_directive": generation_directive,
                        "reranking_bias": reranking_bias,
                        "deprioritize": deprioritize,
                    }
    return style_map


FSLSM_STYLE_MAP: dict = _build_style_map()


# ---------------------------------------------------------------------------
# ProfileAgent
# ---------------------------------------------------------------------------

class ProfileAgent:
    """Translates an FSLSM binary vector into a reasoning plan for downstream agents."""

    def __init__(self, profiles_path: str | Path | None = None):
        """
        Args:
            profiles_path: Path to data/fslsm/profiles.json.
                           If None, style_descriptor_graf will not be attached.
        """
        self._graf_descriptors: dict[tuple, str] = {}
        if profiles_path is not None:
            self._load_graf_descriptors(Path(profiles_path))

    def _load_graf_descriptors(self, path: Path) -> None:
        with open(path) as f:
            profiles = json.load(f)
        for p in profiles:
            dims = p.get("dimensions", {})
            if dims.get("act_ref") == 0:
                continue  # skip baseline P00
            key = (dims["act_ref"], dims["sen_int"], dims["vis_ver"], dims["seq_glo"])
            self._graf_descriptors[key] = p.get("style_descriptor_graf", "")

    def generate_reasoning_plan(self, fslsm_vector: dict) -> dict:
        """
        Convert an FSLSM binary vector to a reasoning plan.

        Args:
            fslsm_vector: Dict with keys act_ref, sen_int, vis_ver, seq_glo,
                          each valued -1 or +1.

        Returns:
            Reasoning plan dict with profile_code, style_label,
            retrieval_directive, generation_directive, reranking_bias,
            deprioritize, and optionally style_descriptor_graf.

        Raises:
            ValueError: If the vector has invalid keys or values.
        """
        self._validate_vector(fslsm_vector)
        key = (
            fslsm_vector["act_ref"],
            fslsm_vector["sen_int"],
            fslsm_vector["vis_ver"],
            fslsm_vector["seq_glo"],
        )
        plan = dict(FSLSM_STYLE_MAP[key])  # shallow copy
        if key in self._graf_descriptors:
            plan["style_descriptor_graf"] = self._graf_descriptors[key]
        return plan

    def generate_system_prompt(self, reasoning_plan: dict) -> str:
        """
        Build a tutor system prompt from a reasoning plan.

        Uses prescriptive MUST-language with concrete structural requirements
        so the LLM produces visibly style-conformant responses.
        """
        label = reasoning_plan["style_label"]
        gen_dir = reasoning_plan["generation_directive"]
        graf = reasoning_plan.get("style_descriptor_graf", "")

        prompt = (
            f"You are an expert AI Tutor for Introductory Machine Learning, "
            f"using the Dive into Deep Learning (D2L) textbook as your knowledge source.\n\n"
            f"The student is a {label} learner. Your response must achieve two goals "
            f"in strict priority order:\n\n"
            f"{_CONTENT_ANCHOR}\n"
            f"## STYLE REQUIREMENTS (applied after Step 0)\n\n"
            f"{gen_dir}\n"
        )
        if graf:
            prompt += f"\n## PEDAGOGICAL GUIDANCE\n{graf}\n"
        prompt += (
            f"\n## PRIORITY ORDER\n"
            f"1. FACTUAL COMPLETENESS (non-negotiable): Your response MUST accurately "
            f"cover all key concepts from the retrieved evidence. An incomplete or "
            f"inaccurate answer is a failure regardless of style.\n"
            f"2. STYLE ADAPTATION (required, applied on top): Once factual content is "
            f"secured, adapt the structure, formatting, and language to match the "
            f"{label} learning style using the requirements above.\n\n"
            f"A response that is stylistically perfect but factually incomplete "
            f"is a failure. A response that is factually complete but ignores style "
            f"is also a failure. You must achieve both."
        )
        return prompt

    @staticmethod
    def _validate_vector(fslsm_vector: dict) -> None:
        if not isinstance(fslsm_vector, dict):
            raise ValueError(f"fslsm_vector must be a dict, got {type(fslsm_vector).__name__}")

        keys = set(fslsm_vector.keys())
        if keys != REQUIRED_DIMS:
            missing = REQUIRED_DIMS - keys
            extra = keys - REQUIRED_DIMS
            parts = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra:
                parts.append(f"unexpected: {extra}")
            raise ValueError(f"Invalid FSLSM vector dimensions — {', '.join(parts)}")

        for dim, val in fslsm_vector.items():
            if val not in VALID_VALUES:
                raise ValueError(
                    f"Dimension '{dim}' must be -1 or +1, got {val!r}"
                )
