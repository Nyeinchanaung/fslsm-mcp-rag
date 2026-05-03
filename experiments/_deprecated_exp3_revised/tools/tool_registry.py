"""
Experiment 3 Tool Registry — Aligned to Thesis Appendix B (Table B.1)
All 15 instructional MCP tools + 1 system logger.
Descriptions include domain-bridging vocabulary for FAISS retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class MCPTool:
    tool_id: int
    name: str
    category: str
    fslsm_dims: List[str]
    description: str
    parameters: Dict[str, Any]
    token_cost: int


# ── CONTENT DELIVERY (tools 1–7) ─────────────────────────────

TOOL_01 = MCPTool(
    tool_id=1,
    name="Concept Explainer",
    category="Content Delivery",
    fslsm_dims=["Sensing", "Verbal"],
    description=(
        "Concept Explainer: Provide a clear, concrete, fact-based textual "
        "explanation of a machine-learning concept using flowing connected "
        "written prose. Build the explanation sentence by sentence with "
        "real-world examples, standard definitions, and practical details. "
        "Use when the query asks 'what is', 'explain', 'describe', 'how does "
        "X work', or requests a definition of neural networks, loss functions, "
        "optimizers, regularization, embeddings, or any D2L topic. "
        "Best for Sensing and Verbal learners who prefer reading ordered "
        "written explanations with concrete factual grounding."
    ),
    parameters={
        "concept": {"type": "string", "description": "ML concept to explain"},
        "detail_level": {"type": "string", "enum": ["brief", "standard", "detailed"]},
        "include_example": {"type": "boolean", "default": True},
    },
    token_cost=95,
)

TOOL_02 = MCPTool(
    tool_id=2,
    name="Step-by-Step Derivator",
    category="Content Delivery",
    fslsm_dims=["Sequential", "Active"],
    description=(
        "Step-by-Step Derivator: Walk through a mathematical derivation, "
        "algorithm execution, or training procedure in numbered sequential "
        "steps. Each step shows one transformation, one computation, or one "
        "configuration change with explicit intermediate results. "
        "Use when the query asks 'derive', 'trace the steps', 'walk through', "
        "'how is X computed step by step', or involves backpropagation, "
        "gradient descent updates, forward pass computation, loss calculation, "
        "or layer-by-layer network construction. "
        "Best for Sequential and Active learners who need ordered procedural "
        "progression through technical operations."
    ),
    parameters={
        "topic": {"type": "string"},
        "show_intermediate": {"type": "boolean", "default": True},
        "notation": {"type": "string", "enum": ["latex", "plain"], "default": "latex"},
    },
    token_cost=102,
)

TOOL_03 = MCPTool(
    tool_id=3,
    name="Worked Example Generator",
    category="Content Delivery",
    fslsm_dims=["Sensing", "Sequential"],
    description=(
        "Worked Example Generator: Present a fully solved example problem "
        "with every intermediate computation shown explicitly using concrete "
        "numbers, real tensor shapes, and standard library functions. "
        "Each line is a calculation step annotated with its rule or formula. "
        "Use when the query asks 'show me an example', 'solve this', 'compute', "
        "'calculate', or involves numerical matrix multiplication, convolution "
        "output dimensions, loss computation, or gradient calculation. "
        "Best for Sensing and Sequential learners who learn through concrete "
        "worked solutions with practical numerical details."
    ),
    parameters={
        "concept": {"type": "string"},
        "domain": {"type": "string", "enum": ["numerical", "code", "hybrid"]},
        "difficulty": {"type": "string", "enum": ["introductory", "intermediate"]},
    },
    token_cost=98,
)

TOOL_04 = MCPTool(
    tool_id=4,
    name="Diagrammatic-Text Explainer",
    category="Content Delivery",
    fslsm_dims=["Visual"],
    description=(
        "Diagrammatic-Text Explainer: Generate a labeled diagram, flowchart, "
        "computation graph, or schematic figure using ASCII art, text-based "
        "spatial layouts, or structured visual scaffolding with annotations. "
        "Show architecture blocks, data flow arrows, and connection patterns. "
        "Use when the query asks 'visualize', 'draw', 'diagram', 'show the "
        "architecture', 'illustrate', or involves network architectures like "
        "ResNet, transformer encoder-decoder, attention mechanisms, CNN layers, "
        "or computation graphs for backpropagation. "
        "Best for Visual learners who prefer pictures, charts, and spatial "
        "representations of information."
    ),
    parameters={
        "concept": {"type": "string"},
        "format": {"type": "string", "enum": ["ascii_graph", "latex_diagram", "spatial_narrative"]},
        "annotate": {"type": "boolean", "default": True},
    },
    token_cost=110,
)

TOOL_05 = MCPTool(
    tool_id=5,
    name="Analogical Reasoner",
    category="Content Delivery",
    fslsm_dims=["Intuitive", "Global"],
    description=(
        "Analogical Reasoner: Explain ML concepts through analogies, metaphors, "
        "and big-picture connections to familiar domains. Map abstract ideas "
        "like attention mechanisms to everyday phenomena, link theoretical "
        "concepts across different areas of machine learning, and provide "
        "holistic intuitive understanding before technical details. "
        "Use when the query asks 'why does this matter', 'how does X relate to Y', "
        "'what is the intuition behind', or involves connecting abstract concepts "
        "like regularization to overfitting, or transformers to RNNs conceptually. "
        "Best for Intuitive and Global learners who grasp abstractions through "
        "metaphorical reasoning and cross-domain connections."
    ),
    parameters={
        "concept": {"type": "string"},
        "analogy_domain": {"type": "string"},
        "depth": {"type": "string", "enum": ["surface", "deep"], "default": "deep"},
    },
    token_cost=97,
)

TOOL_06 = MCPTool(
    tool_id=6,
    name="Comparative Explainer",
    category="Content Delivery",
    fslsm_dims=["Sensing", "Sequential"],
    description=(
        "Comparative Explainer: Generate a structured side-by-side comparison "
        "of two or more related concepts, algorithms, or architectures. "
        "Output an enumerated contrast list or comparison table showing "
        "key differences across specific axes like speed, accuracy, memory, "
        "number of parameters, or training strategy. "
        "Use when the query asks 'compare', 'contrast', 'difference between', "
        "'how does X differ from Y', 'X versus Y', or involves comparing "
        "architectures (ResNet vs VGG), optimizers (SGD vs Adam), or "
        "frameworks (PyTorch vs MXNet). "
        "Best for Sensing and Sequential learners who anchor understanding "
        "through systematic factual distinction."
    ),
    parameters={
        "concept_a": {"type": "string"},
        "concept_b": {"type": "string"},
        "comparison_axes": {"type": "array", "items": {"type": "string"}},
    },
    token_cost=105,
)

TOOL_07 = MCPTool(
    tool_id=7,
    name="Concept Map Generator",
    category="Content Delivery",
    fslsm_dims=["Global"],
    description=(
        "Concept Map Generator: Build a hierarchical or networked concept map "
        "showing how the topic connects to surrounding ideas as an indented "
        "text tree or adjacency description with labeled relationships. "
        "Show the big picture of how neural network components, training "
        "strategies, and evaluation metrics relate to each other. "
        "Use when the query asks 'overview', 'how do these topics connect', "
        "'map the relationships', 'big picture', or involves understanding "
        "how multiple D2L chapters relate to a single theme. "
        "Best for Global learners who need to see the whole picture and "
        "thematic connections before engaging with details."
    ),
    parameters={
        "root_concept": {"type": "string"},
        "depth": {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
        "include_descriptions": {"type": "boolean", "default": False},
    },
    token_cost=90,
)


# ── PERSONALISATION (tools 8–11) ─────────────────────────────

TOOL_08 = MCPTool(
    tool_id=8,
    name="PersonaRAG Adapter",
    category="Personalisation",
    fslsm_dims=["Active", "Reflective", "Sensing", "Intuitive",
                "Visual", "Verbal", "Sequential", "Global"],
    description=(
        "PersonaRAG Adapter: Rewrite retrieved D2L corpus chunks into the "
        "learner's FSLSM style before presentation. Apply style transfer to "
        "raw retrieval output — convert factual text into visual diagrams for "
        "Visual learners, step-by-step for Sequential, analogical for Intuitive, "
        "concrete examples for Sensing, and reflective prompts for Reflective. "
        "Use when the query asks 're-explain in my style', 'adapt this to how "
        "I learn', or when retrieved content needs pedagogical style matching. "
        "Applicable across all FSLSM dimension combinations. "
        "Best when content exists but needs style transformation."
    ),
    parameters={
        "raw_text": {"type": "string", "description": "Retrieved chunk to adapt"},
        "fslsm_profile": {"type": "object"},
    },
    token_cost=130,
)

TOOL_09 = MCPTool(
    tool_id=9,
    name="FSLSM Styler",
    category="Personalisation",
    fslsm_dims=["Active", "Reflective", "Sensing", "Intuitive",
                "Visual", "Verbal", "Sequential", "Global"],
    description=(
        "FSLSM Styler: Transform an existing explanation from one FSLSM "
        "learning style to another on demand. Convert verbal prose to visual "
        "layout, sequential steps to global overview, abstract theory to "
        "concrete worked example, or passive reading to active exercise. "
        "Use when the query asks 'explain this differently', 'transform this "
        "into a more visual/verbal/active form', or 'show me another way'. "
        "Real-time style transfer of already-generated content. "
        "Best when the learner has seen one presentation and wants another."
    ),
    parameters={
        "source_text": {"type": "string"},
        "target_style": {"type": "string"},
    },
    token_cost=118,
)

TOOL_10 = MCPTool(
    tool_id=10,
    name="Think-Pair-Share Generator",
    category="Personalisation",
    fslsm_dims=["Reflective"],
    description=(
        "Think-Pair-Share Generator: Generate structured reflection prompts "
        "with three phases — a silent thinking question, a peer discussion "
        "prompt, and a synthesis task. Encourage self-reflection before "
        "responding, articulation of reasoning in own words, and "
        "identification of what is understood versus unclear. "
        "Use when the query benefits from pause-and-think before answering, "
        "when the learner should reflect on gradient computation, model "
        "selection trade-offs, or architectural design decisions. "
        "Best for Reflective learners who process by thinking and writing "
        "alone before engaging in discussion."
    ),
    parameters={
        "topic": {"type": "string"},
        "think_time_minutes": {"type": "integer", "default": 2},
        "include_synthesis": {"type": "boolean", "default": True},
    },
    token_cost=92,
)

TOOL_11 = MCPTool(
    tool_id=11,
    name="Interactive Exercise Generator",
    category="Personalisation",
    fslsm_dims=["Active"],
    description=(
        "Interactive Exercise Generator: Create hands-on coding exercises, "
        "fill-in-the-blank derivations, or interactive problem-solving tasks "
        "that require the learner to do rather than observe. Include setup "
        "instructions, task description, expected output, and progressive hints. "
        "Use when the query asks 'give me an exercise', 'let me practice', "
        "'hands-on task', or when the learner should implement a neural network "
        "layer, write a training loop, or compute gradients manually. "
        "Best for Active learners who retain information through doing, "
        "experimenting, and trying things out."
    ),
    parameters={
        "concept": {"type": "string"},
        "exercise_type": {"type": "string", "enum": ["coding", "fill_blank", "problem_solve"]},
        "difficulty": {"type": "string", "enum": ["beginner", "intermediate"]},
    },
    token_cost=108,
)


# ── ASSESSMENT (tools 12–13) ─────────────────────────────────

TOOL_12 = MCPTool(
    tool_id=12,
    name="Quiz Generator",
    category="Assessment",
    fslsm_dims=["Active", "Sensing"],
    description=(
        "Quiz Generator: Generate multiple-choice, short-answer, or true/false "
        "quiz questions on a given ML concept with immediate feedback and "
        "brief explanations for each answer. Test factual knowledge of "
        "definitions, formulas, hyperparameters, and architectural choices. "
        "Use when the query asks 'quiz me', 'test my understanding', "
        "'check if I know', or when the learner wants concrete knowledge "
        "checks on backpropagation rules, optimizer parameters, or "
        "activation function properties. "
        "Best for Active and Sensing learners who prefer concrete "
        "knowledge verification over open-ended reflection."
    ),
    parameters={
        "topic": {"type": "string"},
        "n_questions": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        "format": {"type": "string", "enum": ["mcq", "short_answer", "true_false"]},
    },
    token_cost=96,
)

TOOL_13 = MCPTool(
    tool_id=13,
    name="Summarizer",
    category="Assessment",
    fslsm_dims=["Global", "Reflective"],
    description=(
        "Summarizer: Produce a concise synthesis of session content, a D2L "
        "chapter section, or a multi-concept discussion. Highlight key "
        "takeaways, thematic connections, and core definitions. Include "
        "a structured list of key terms with brief explanations. "
        "Use when the query asks 'summarize', 'overview', 'what are the "
        "key points', 'tl;dr', or when consolidating understanding after "
        "a long explanation of training procedures, architectures, or "
        "optimization strategies. "
        "Best for Global and Reflective learners who consolidate "
        "understanding through synthesis and review."
    ),
    parameters={
        "content": {"type": "string"},
        "length": {"type": "string", "enum": ["brief", "standard", "comprehensive"]},
        "include_key_terms": {"type": "boolean", "default": True},
    },
    token_cost=88,
)


# ── RETRIEVAL (tools 14–15) ──────────────────────────────────

TOOL_14 = MCPTool(
    tool_id=14,
    name="Content Retriever",
    category="Retrieval",
    fslsm_dims=["Active", "Reflective", "Sensing", "Intuitive",
                "Visual", "Verbal", "Sequential", "Global"],
    description=(
        "Content Retriever: Retrieve the top-k most semantically relevant "
        "chunks from the D2L corpus FAISS index for a given query. Return "
        "raw document passages for downstream processing by other tools. "
        "Use when the query needs factual grounding from the textbook, "
        "when the learner asks 'what does D2L say about', 'find the "
        "relevant section', or when retrieved context is needed before "
        "explanation, comparison, or exercise generation. "
        "Applicable across all learning styles — style adaptation is "
        "handled by downstream tools."
    ),
    parameters={
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        "rerank": {"type": "boolean", "default": False},
    },
    token_cost=85,
)

TOOL_15 = MCPTool(
    tool_id=15,
    name="Web Search Tool",
    category="Retrieval",
    fslsm_dims=["Intuitive"],
    description=(
        "Web Search Tool: Perform a real-time web search for supplementary "
        "information beyond the D2L corpus. Retrieve novel cross-domain "
        "connections, recent research papers, cutting-edge model comparisons, "
        "and innovative applications not covered in the textbook. "
        "Use when the query asks 'what is the latest', 'beyond D2L', "
        "'recent developments', 'how is X used in industry', or when the "
        "learner seeks abstract connections and inventive theoretical ideas "
        "outside the standard curriculum. "
        "Best for Intuitive learners who seek novel, cross-domain, and "
        "cutting-edge connections beyond established course material."
    ),
    parameters={
        "query": {"type": "string"},
        "max_results": {"type": "integer", "default": 3},
        "filter_academic": {"type": "boolean", "default": False},
    },
    token_cost=78,
)


# ── SYSTEM (tool 0) ──────────────────────────────────────────

TOOL_00 = MCPTool(
    tool_id=0,
    name="Evaluation Logger",
    category="System",
    fslsm_dims=[],
    description="System-level logger. Not a tool-selection participant.",
    parameters={},
    token_cost=0,
)


# ── REGISTRY ─────────────────────────────────────────────────

TOOL_REGISTRY: List[MCPTool] = [
    TOOL_01, TOOL_02, TOOL_03, TOOL_04, TOOL_05,
    TOOL_06, TOOL_07, TOOL_08, TOOL_09, TOOL_10,
    TOOL_11, TOOL_12, TOOL_13, TOOL_14, TOOL_15,
]

TOOL_BY_ID: Dict[int, MCPTool] = {t.tool_id: t for t in TOOL_REGISTRY}
TOOL_BY_ID[0] = TOOL_00


def get_tool_by_id(tool_id: int) -> MCPTool:
    return TOOL_BY_ID[tool_id]


def s0_prompt_tokens() -> int:
    """Total tokens for S0 (all 15 schemas in context)."""
    return sum(t.token_cost for t in TOOL_REGISTRY)


def registry_summary() -> None:
    print(f"\n{'#':>3}  {'Tool Name':<35} {'Category':<20} {'FSLSM Dims':<30} {'Tokens':>6}")
    print("─" * 100)
    for t in TOOL_REGISTRY:
        dims = ", ".join(t.fslsm_dims[:3]) + ("..." if len(t.fslsm_dims) > 3 else "")
        print(f"{t.tool_id:>3}  {t.name:<35} {t.category:<20} {dims:<30} {t.token_cost:>6}")
    print("─" * 100)
    print(f"     {'S0 total':<55} {s0_prompt_tokens():>6}")


if __name__ == "__main__":
    registry_summary()
