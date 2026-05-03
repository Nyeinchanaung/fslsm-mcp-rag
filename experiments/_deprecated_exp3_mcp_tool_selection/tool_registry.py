"""15 MCP tutoring tools + ground truth + S0 token-cost baseline.

Design:
- Tools are pedagogical actions that a tutor agent could invoke. Each is tagged
  with the FSLSM dimension poles it primarily serves (Felder & Silverman, 1988).
- All 8 poles are covered by 3–5 tools (some tools serve multiple poles).
- `get_optimal_tool_id` defines the *expert-optimal* tool for a (query, profile)
  pair via intent classification + PROFILE_TOOL_MAP. For the dominant "explain"
  intent, profile alone determines the tool, making the experiment a clean test
  of FSLSM conditioning. Three non-explain intents use profile-aware overrides.

FSLSM encoding (matches `data/agents/validated_agents.json`):
    act_ref: -1 = Active,    +1 = Reflective
    sen_int: -1 = Sensing,   +1 = Intuitive
    vis_ver: -1 = Visual,    +1 = Verbal
    seq_glo: -1 = Sequential,+1 = Global
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MCPTool:
    tool_id: int
    name: str
    category: str
    fslsm_dims: tuple[str, ...]
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def token_cost(self) -> int:
        """Approximate prompt-bloat cost: name + description + parameters JSON.
        Uses the GPT-style ~4 chars/token rule. Computed per access; cheap."""
        payload = f"{self.name}\n{self.description}\n{json.dumps(self.parameters)}"
        return max(1, len(payload) // 4)


# --------------------------------------------------------------------------- #
# 15 MCP tools                                                                #
# --------------------------------------------------------------------------- #

TOOL_REGISTRY: list[MCPTool] = [
    MCPTool(
        tool_id=1,
        name="diagram_renderer",
        category="visualization",
        fslsm_dims=("Visual", "Sequential"),
        description=(
            "Generate a labeled diagram, flowchart, or schematic figure that visually "
            "illustrates a concept, architecture, or network structure. Use when the "
            "query asks to draw or visualize an architecture, show a computation graph, "
            "depict layer layout, or render a training pipeline as a figure. Produces "
            "an annotated image with visual elements linked to terminology in a clear "
            "ordered structure. Best for Visual and Sequential learners who prefer "
            "pictures, charts, and graphical representations laid out step by step."
        ),
        parameters={"concept": "string", "style": "schematic|flowchart|architecture"},
    ),
    MCPTool(
        tool_id=2,
        name="interactive_simulation",
        category="visualization",
        fslsm_dims=("Visual", "Active"),
        description=(
            "Launch an interactive visual simulation with sliders, buttons, and "
            "manipulable parameters that the learner can experiment with hands-on. "
            "Use when the query asks to experiment with model behaviour, adjust "
            "hyperparameters and observe how outputs change, or explore how varying "
            "an architecture setting affects a training curve. Updates plots in real "
            "time. Best for Visual and Active learners who learn by doing and "
            "experimenting with interactive graphical demonstrations."
        ),
        parameters={"system": "string", "controls": "list"},
    ),
    MCPTool(
        tool_id=3,
        name="stepwise_walkthrough",
        category="procedural",
        fslsm_dims=("Sequential", "Sensing"),
        description=(
            "Decompose an algorithm, training procedure, or architectural sequence "
            "into an explicit numbered walkthrough — each step shows one transformation, "
            "computation, or configuration change with a concrete intermediate result. "
            "Use when the query asks how an algorithm works step by step, traces "
            "execution of a training loop, describes the forward or backward pass, "
            "follows layer-by-layer construction of a network, or asks about "
            "initialization and update sequences. Best for Sequential and Sensing "
            "learners who need an ordered concrete progression through technical "
            "procedures with practical details at each stage."
        ),
        parameters={"procedure": "string", "depth": "shallow|standard|detailed"},
    ),
    MCPTool(
        tool_id=4,
        name="worked_example",
        category="procedural",
        fslsm_dims=("Sensing", "Sequential"),
        description=(
            "Present a fully solved example problem with every intermediate computation "
            "shown explicitly — concrete numerical inputs, standard methods, and "
            "annotated calculation steps rather than abstract formulas. Use when the "
            "query asks for a numerical example, a concrete calculation walkthrough, "
            "an explicit worked solution, or wants to see the numbers behind a "
            "formula or algorithm applied to a specific instance. Best for Sensing "
            "and Sequential learners who prefer concrete facts, established procedures, "
            "and detail-rich worked solutions with explicit values."
        ),
        parameters={"problem": "string"},
    ),
    MCPTool(
        tool_id=5,
        name="conceptual_overview",
        category="overview",
        fslsm_dims=("Global", "Intuitive"),
        description=(
            "Provide a high-level conceptual overview that frames the topic in its "
            "broader context — motivates why it matters, maps how it connects to "
            "neighbouring architectures or ideas, and explains the big picture before "
            "details. Use when the query asks for the big picture of a topic, the "
            "motivation behind an architectural design choice, how a technique fits "
            "into the broader field, an overview of a method, or why a certain "
            "approach exists. Best for Global and Intuitive learners who need "
            "holistic abstract framing and underlying meaning before diving into details."
        ),
        parameters={"topic": "string"},
    ),
    MCPTool(
        tool_id=6,
        name="abstract_derivation",
        category="theoretical",
        fslsm_dims=("Intuitive", "Verbal"),
        description=(
            "Derive a result from first principles using formal notation and "
            "mathematical reasoning — loss function derivation, gradient computation, "
            "proof of convergence, or symbolic manipulation of an objective. Use when "
            "the query asks to derive, prove, or formally show a mathematical "
            "relationship, asks how a loss function is constructed, how gradients "
            "are computed through a layer, or requests a theoretical justification "
            "using symbolic algebra. Best for Intuitive and Verbal learners who "
            "prefer abstract theory, formal derivations, and symbolic mathematical "
            "reasoning over concrete examples."
        ),
        parameters={"theorem": "string"},
    ),
    MCPTool(
        tool_id=7,
        name="analogy_explainer",
        category="theoretical",
        fslsm_dims=("Intuitive", "Verbal"),
        description=(
            "Explain a concept by mapping it to a familiar analogy or metaphor, "
            "drawing conceptual parallels in written prose to convey underlying "
            "meaning. Use when the query asks for the intuition behind a technique, "
            "what a concept is like in simpler terms, a metaphor for how an "
            "architectural mechanism works, or an inventive comparison between two "
            "ideas to build abstract understanding. Best for Intuitive and Verbal "
            "learners who grasp abstractions through verbal metaphors and "
            "creative theoretical comparisons."
        ),
        parameters={"concept": "string", "domain_hint": "string"},
    ),
    MCPTool(
        tool_id=8,
        name="socratic_dialogue",
        category="dialogic",
        fslsm_dims=("Active", "Reflective"),
        description=(
            "Engage the learner with a sequence of pointed Socratic questions that "
            "probe understanding, surface misconceptions, and require active discussion. "
            "Use when the query asks to discuss a topic, debate architectural choices, "
            "probe understanding of a design decision, or explore what the learner "
            "thinks about a concept through interactive questioning. Alternates "
            "between active answering and reflective pauses for thinking. Best for "
            "Active and Reflective learners who learn by discussing, questioning, "
            "and actively engaging with dialogue."
        ),
        parameters={"topic": "string", "depth": "shallow|standard|probing"},
    ),
    MCPTool(
        tool_id=9,
        name="practice_exercise",
        category="exercise",
        fslsm_dims=("Active", "Sensing"),
        description=(
            "Generate a hands-on practice problem set with concrete numerical inputs, "
            "standard procedures to apply, and immediate feedback on each attempt. "
            "Use when the query asks for practice problems, exercises to try, a "
            "problem set on a topic, or wants to test understanding by solving "
            "instances. Learner works through concrete factual problems with "
            "established methods. Best for Active and Sensing learners who learn "
            "by doing concrete practical exercises with direct feedback."
        ),
        parameters={"topic": "string", "difficulty": "easy|standard|hard"},
    ),
    MCPTool(
        tool_id=10,
        name="reflection_prompt",
        category="dialogic",
        fslsm_dims=("Reflective", "Verbal"),
        description=(
            "Issue a guided reflection prompt asking the learner to think quietly, "
            "write out their reasoning in their own words, and articulate what they "
            "understand versus what is still unclear. Use when the query asks to "
            "reflect on a concept, write out an explanation in personal terms, "
            "articulate understanding of a mechanism, or consolidate learning "
            "through written self-explanation. Best for Reflective and Verbal "
            "learners who process by thinking alone and writing prose."
        ),
        parameters={"topic": "string"},
    ),
    MCPTool(
        tool_id=11,
        name="summary_outline",
        category="overview",
        fslsm_dims=("Reflective", "Sequential"),
        description=(
            "Produce a structured outline that organises the topic into ordered "
            "sections, sub-sections, and bullet points for review. Use when the "
            "query asks for a summary, an outline of key points, a structured list "
            "of concepts, a review of what was covered, or a scannable reference "
            "sheet for a topic. Linear, ordered, bullet-point format. Best for "
            "Reflective and Sequential learners who consolidate by reading "
            "ordered structured outlines quietly after a lesson."
        ),
        parameters={"topic": "string", "depth": "1|2|3"},
    ),
    MCPTool(
        tool_id=12,
        name="prose_explainer",
        category="explanation",
        fslsm_dims=("Verbal", "Sequential"),
        description=(
            "Explain a concept in flowing connected written prose, building the "
            "explanation sentence by sentence in a clear linear order. Use when "
            "the query asks what X is, how X influences Y, how does X work, "
            "describe the relationship between two components, what is the effect "
            "of X on Y, or explain the mechanism behind an architectural or "
            "training concept. Heavy on written words, light on figures. Best for "
            "Verbal and Sequential learners who prefer reading ordered written "
            "narrative explanations."
        ),
        parameters={"topic": "string"},
    ),
    MCPTool(
        tool_id=13,
        name="code_sandbox",
        category="exercise",
        fslsm_dims=("Active", "Sensing"),
        description=(
            "Open a runnable code sandbox where the learner can edit, execute, and "
            "inspect outputs of working code that demonstrates the concept. Use when "
            "the query asks to implement, code, write a function, program a solution, "
            "run an algorithm, test an implementation, or experiment with concrete "
            "code that applies a technique. Best for Active and Sensing learners "
            "who prefer doing and trying concrete code over reading theory."
        ),
        parameters={"language": "python", "starter_code": "string"},
    ),
    MCPTool(
        tool_id=14,
        name="concept_map",
        category="visualization",
        fslsm_dims=("Global", "Visual"),
        description=(
            "Build a graphical concept map that visually shows how the topic "
            "connects to surrounding ideas as a network of nodes and labelled edges. "
            "Use when the query asks about the connections between ideas, how X "
            "relates to Y across the field, a map of how concepts fit together, "
            "or how an architecture sits within the broader landscape of techniques. "
            "Holistic big-picture diagram of relationships. Best for Global and "
            "Visual learners who need to see overall picture connections in a "
            "chart-like network layout."
        ),
        parameters={"central_concept": "string"},
    ),
    MCPTool(
        tool_id=15,
        name="case_study",
        category="overview",
        fslsm_dims=("Sensing", "Global"),
        description=(
            "Present an extended real-world case study that anchors the topic in a "
            "concrete practical application — showing how the broader picture and "
            "related ideas come together within a single facts-rich scenario. Use "
            "when the query asks for a real-world example of a technique, how X is "
            "used in practice, an application of a method in a deployed system, "
            "or wants to see the topic through a concrete scenario embedded in "
            "its wider context. Best for Sensing and Global learners who learn "
            "through concrete real applications placed in holistic context."
        ),
        parameters={"domain": "string"},
    ),
]

assert len({t.tool_id for t in TOOL_REGISTRY}) == 15, "tool_id collision"
assert all(1 <= t.tool_id <= 15 for t in TOOL_REGISTRY)


# --------------------------------------------------------------------------- #
# Ground truth — expert-optimal tool for a (query, profile) pair              #
# --------------------------------------------------------------------------- #

# Maps query intent (lowercase keyword) → category that best serves it.
INTENT_KEYWORDS: dict[str, list[str]] = {
    "visualize":   ["diagram", "visualize", "draw", "plot", "chart", "figure", "show me"],
    "solve":       ["solve", "compute", "calculate", "evaluate", "derive the value"],
    "implement":   ["implement", "code", "program", "write a function", "write code"],
    "practice":    ["practice", "exercise", "try", "test yourself"],
    "compare":     ["compare", "contrast", "difference between", "versus", " vs "],
    "summarize":   ["summarize", "overview", "outline", "summary", "tl;dr", "in short"],
    "derive":      ["derive", "prove", "show that", "demonstrate that"],
    "discuss":     ["discuss", "debate", "argue", "what do you think"],
    "explain":     ["explain", "describe", "what is", "how does", "how do",
                    "why does", "why is", "elaborate", "tell me about"],
}

# Intent → preferred categories (ordered). Used by non-PROFILE_TOOL_MAP paths.
INTENT_CATEGORY_PRIORITY: dict[str, list[str]] = {
    "visualize":   ["visualization", "explanation", "overview"],
    "solve":       ["procedural", "exercise", "explanation"],
    "implement":   ["exercise", "procedural"],
    "practice":    ["exercise", "dialogic", "procedural"],
    "compare":     ["overview", "explanation", "theoretical"],
    "summarize":   ["overview", "explanation"],
    "derive":      ["theoretical", "procedural", "explanation"],
    "discuss":     ["dialogic", "theoretical"],
    "explain":     ["explanation", "procedural", "theoretical", "visualization", "overview"],
}

# Expert-optimal tool for each of the 16 FSLSM profile combinations.
# Keys: (act_ref_pole, sen_int_pole, vis_ver_pole, seq_glo_pole)
# Used when query intent is "explain" (the default 70% case).
# For the non-explain majority, profile alone drives the choice so that
# S1b (FSLSM-conditioned) vs S1a (no profile) measures conditioning value cleanly.
PROFILE_TOOL_MAP: dict[tuple[str, str, str, str], int] = {
    ("Active",     "Sensing",    "Visual",  "Sequential"): 1,   # diagram_renderer
    ("Active",     "Sensing",    "Visual",  "Global"):     14,  # concept_map
    ("Active",     "Intuitive",  "Visual",  "Sequential"): 2,   # interactive_simulation
    ("Active",     "Intuitive",  "Visual",  "Global"):     2,   # interactive_simulation
    ("Reflective", "Sensing",    "Visual",  "Sequential"): 1,   # diagram_renderer
    ("Reflective", "Sensing",    "Visual",  "Global"):     14,  # concept_map
    ("Reflective", "Intuitive",  "Visual",  "Sequential"): 1,   # diagram_renderer
    ("Reflective", "Intuitive",  "Visual",  "Global"):     14,  # concept_map
    ("Active",     "Sensing",    "Verbal",  "Sequential"): 3,   # stepwise_walkthrough
    ("Active",     "Sensing",    "Verbal",  "Global"):     15,  # case_study
    ("Active",     "Intuitive",  "Verbal",  "Sequential"): 6,   # abstract_derivation
    ("Active",     "Intuitive",  "Verbal",  "Global"):     5,   # conceptual_overview
    ("Reflective", "Sensing",    "Verbal",  "Sequential"): 12,  # prose_explainer
    ("Reflective", "Sensing",    "Verbal",  "Global"):     15,  # case_study
    ("Reflective", "Intuitive",  "Verbal",  "Sequential"): 6,   # abstract_derivation
    ("Reflective", "Intuitive",  "Verbal",  "Global"):     7,   # analogy_explainer
}


def _classify_intent(query: str) -> str:
    q = query.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in q for k in kws):
            return intent
    return "explain"  # default


def _profile_to_poles(profile: dict) -> set[str]:
    """Convert {act_ref, sen_int, vis_ver, seq_glo} → set of pole names."""
    poles = set()
    poles.add("Reflective" if profile["act_ref"] > 0 else "Active")
    poles.add("Intuitive"  if profile["sen_int"] > 0 else "Sensing")
    poles.add("Verbal"     if profile["vis_ver"] > 0 else "Visual")
    poles.add("Global"     if profile["seq_glo"] > 0 else "Sequential")
    return poles


def get_optimal_tool_id(query: str, profile: dict) -> int:
    """Expert-defined optimal tool for (query, profile).

    For "explain" intent (≈70% of queries): profile alone determines the tool
    via PROFILE_TOOL_MAP — makes the experiment a clean test of FSLSM conditioning.
    For three non-explain intents with strong query-content signal, a profile-aware
    override is applied. All other intents (solve, derive, compare, summarize,
    discuss) also fall through to PROFILE_TOOL_MAP.
    """
    intent = _classify_intent(query)
    poles = _profile_to_poles(profile)

    # Strong intent overrides where query content determines the tool category
    if intent == "implement":
        return 13 if "Active" in poles else 4    # code_sandbox vs worked_example
    if intent == "practice":
        return 9  if "Active" in poles else 11   # practice_exercise vs summary_outline
    if intent == "visualize":
        return 1  if "Sequential" in poles else 14  # diagram_renderer vs concept_map

    # All remaining intents (explain, solve, derive, compare, summarize, discuss):
    # FSLSM profile is the sole determinant via PROFILE_TOOL_MAP
    act = "Active"     if "Active"     in poles else "Reflective"
    sns = "Sensing"    if "Sensing"    in poles else "Intuitive"
    vis = "Visual"     if "Visual"     in poles else "Verbal"
    seq = "Sequential" if "Sequential" in poles else "Global"
    return PROFILE_TOOL_MAP.get((act, sns, vis, seq), 12)  # fallback: prose_explainer


def s0_select_tool_id(query: str) -> int:
    """S0 baseline: keyword-overlap selection across all 15 tools (no profile).

    Simulates a 'naive LLM' that has all 15 schemas in context and picks the
    one whose description shares the most surface vocabulary with the query.
    Profile-blind by construction.
    """
    q_words = {w for w in query.lower().split() if len(w) > 3}
    if not q_words:
        return TOOL_REGISTRY[0].tool_id

    best_score = -1
    best_id = TOOL_REGISTRY[0].tool_id
    for tool in TOOL_REGISTRY:
        text = f"{tool.name.replace('_', ' ')} {tool.description}".lower()
        d_words = {w for w in text.split() if len(w) > 3}
        overlap = len(q_words & d_words)
        if overlap > best_score:
            best_score = overlap
            best_id = tool.tool_id
    return best_id


# --------------------------------------------------------------------------- #
# S0 prompt-bloat baseline                                                    #
# --------------------------------------------------------------------------- #

def s0_prompt_tokens() -> int:
    """Total tokens for the S0 baseline (all 15 schemas in the prompt)."""
    return sum(t.token_cost for t in TOOL_REGISTRY)


def s1_prompt_tokens(tool_id: int) -> int:
    """Tokens for an S1a/S1b prompt — schema for a single retrieved tool."""
    tool = next(t for t in TOOL_REGISTRY if t.tool_id == tool_id)
    return tool.token_cost
