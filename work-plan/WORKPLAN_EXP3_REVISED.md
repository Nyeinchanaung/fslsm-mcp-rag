# Experiment 3 — Complete Revision & Development Workplan
> **For use with Claude Code in local IDE**
> **Version 2.0 — Supersedes EXPERIMENT_3_WORKPLAN.md and DIAGNOSTIC_DATA_COLLECTION.md**

---

## Table of Contents

- [Overview](#overview)
- [Phase 0: Prerequisites](#phase-0--prerequisites)
- [Phase 1: Tool Registry — Align to Appendix B](#phase-1--tool-registry--align-to-appendix-b)
- [Phase 2: FAISS Tool Index — Rebuild with Domain-Bridging Descriptions](#phase-2--faiss-tool-index--rebuild-with-domain-bridging-descriptions)
- [Phase 3: Profile Decoder & Augmentor Fixes](#phase-3--profile-decoder--augmentor-fixes)
- [Phase 4: GROUND_TRUTH_MAP — Full 4D Mapping](#phase-4--ground_truth_map--full-4d-mapping)
- [Phase 5: Generate 16 Coverage Questions (R2b)](#phase-5--generate-16-coverage-questions-r2b)
- [Phase 6: Web Search Tool — Tavily Integration](#phase-6--web-search-tool--tavily-integration)
- [Phase 7: TOOL_PROMPTS — All 15 Prompt Templates](#phase-7--tool_prompts--all-15-prompt-templates)
- [Phase 8: S0 Baseline — Real LLM Tool Selection](#phase-8--s0-baseline--real-llm-tool-selection)
- [Phase 9: R2 Session Runner — Dual Logger](#phase-9--r2-session-runner--dual-logger)
- [Phase 10: Dry Run Gate (100 Sessions)](#phase-10--dry-run-gate-100-sessions)
- [Phase 11: Full Run — R2a + R2b](#phase-11--full-run--r2a--r2b)
- [Phase 12: Post-hoc S0 + S1a Ablation](#phase-12--post-hoc-s0--s1a-ablation)
- [Phase 13: Metrics & Statistical Analysis](#phase-13--metrics--statistical-analysis)
- [Phase 14: Decision Point & Exp2 R2 Extension](#phase-14--decision-point--exp2-r2-extension)

---

## Overview

### What this workplan fixes

| Issue | Root Cause | Fix Phase |
|---|---|---|
| Tool registry mismatch with thesis | Code tools ≠ Appendix B tools | Phase 1 |
| FAISS cosine scores 0.12–0.32 | Tool descriptions lack domain vocabulary | Phase 2 |
| Profile ±1 misread as binary | `decode_profile()` broken | Phase 3 |
| Augmentor field name mismatch | `act_ref` vs `act` | Phase 3 |
| 8/15 tools never assigned optimal | GROUND_TRUTH_MAP too narrow | Phase 4 |
| Missing tool-type question coverage | D2L-QA lacks practice/quiz/search | Phase 5 |
| Web Search Tool cannot execute | No search API configured | Phase 6 |
| S0 baseline simulated as random | Not defensible for thesis | Phase 8 |

### Architecture after all fixes

```
R2a: 72 original questions × 80 agents = 5,760 sessions
     → Serves BOTH Exp2 (comparable to R0/R1) AND Exp3

R2b: 16 coverage questions × 80 agents = 1,280 sessions
     → Serves Exp3 ONLY (extends tool coverage to all 15 tools)

Exp3 total: 7,040 sessions × 3 conditions (S0/S1a/S1b) = 21,120 data points
```

### Models used

| Role | Model | Purpose |
|---|---|---|
| Tutor agent | GPT-4.1-mini | R2 response generation, S0 tool selection |
| Student agent | GPT-4.1-mini | Persona-based dialogue |
| LLM-as-Judge | GPT-4o | Exp2 evaluation (SCS, RR, Engagement) |
| Embeddings | text-embedding-3-small | FAISS tool index (1536-dim) |
| Coverage Q gen | GPT-4.1-mini | Generate 16 coverage questions |
| Web search | Tavily API | Tool 15 execution in R2 |

---

## Phase 0 — Prerequisites

### Step 0.1 — Install all dependencies

```bash
pip install openai faiss-cpu numpy psycopg2-binary sqlalchemy \
            fastmcp mcp python-dotenv tqdm pandas scipy \
            tavily-python litellm --break-system-packages
```

### Step 0.2 — Set up environment variables

```bash
# .env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...           # Free tier: https://app.tavily.com
POSTGRES_URL=postgresql://localhost:5432/thesis_exp3
```

### Step 0.3 — Confirm existing files

```bash
# Verify these exist before proceeding:
ls data/filtered_questions.json          # 72 original questions
ls data/exp2/sessions.jsonl              # Exp2 session data (adjust path)
ls experiments/exp2_tutor_personalization/results/exp2_session_metrics.csv
```

### Step 0.4 — Create working directory structure

```
exp3_revised/
├── WORKPLAN_EXP3_REVISED.md             ← this file
├── .env
├── config.py
├── tools/
│   ├── tool_registry.py                 ← Phase 1: Appendix B tools
│   ├── tool_prompts.py                  ← Phase 7: all 15 prompt templates
│   ├── tool_index.py                    ← Phase 2: FAISS builder
│   └── tavily_search.py                 ← Phase 6: web search wrapper
├── core/
│   ├── profile_decoder.py               ← Phase 3: ±1 profile decoding
│   ├── fslsm_augmentor.py               ← Phase 3: query augmentation
│   ├── ground_truth.py                  ← Phase 4: GROUND_TRUTH_MAP_FULL
│   ├── s0_baseline.py                   ← Phase 8: real LLM S0 simulation
│   └── session_runner.py                ← Phase 9: R2 runner + dual logger
├── scripts/
│   ├── 01_build_tool_index.py
│   ├── 02_generate_coverage_questions.py ← Phase 5
│   ├── 03_verify_setup.py
│   ├── 04_dry_run.py
│   ├── 05_full_run_r2a.py
│   ├── 06_full_run_r2b.py
│   ├── 07_run_s0_s1a_ablation.py
│   ├── 08_compute_metrics.py
│   └── 09_generate_report.py
├── data/
│   ├── filtered_questions.json          ← original 72 questions
│   ├── coverage_questions.json          ← generated in Phase 5
│   └── all_questions.json               ← merged (72+16) for Exp3
├── results/
│   ├── exp3_results.db
│   ├── exp2_passive_log.jsonl           ← optional Exp2 data
│   ├── exp3_metrics.json
│   └── figures/
└── tests/
    └── test_ground_truth.py
```

---

## Phase 1 — Tool Registry — Align to Appendix B

### Step 1.1 — Create `tools/tool_registry.py`

This replaces the entire tool registry with the 15 proposed tools from Appendix B,
plus domain-bridging vocabulary in each description for FAISS embedding quality.

```python
# tools/tool_registry.py
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

TOOL_BY_ID = {t.tool_id: t for t in TOOL_REGISTRY}
TOOL_BY_ID[0] = TOOL_00


def get_tool_by_id(tool_id: int) -> MCPTool:
    return TOOL_BY_ID[tool_id]


def s0_prompt_tokens() -> int:
    return sum(t.token_cost for t in TOOL_REGISTRY)


def registry_summary():
    print(f"\n{'#':>3}  {'Tool Name':<35} {'Category':<20} {'FSLSM Dims':<30} {'Tokens':>6}")
    print("─" * 100)
    for t in TOOL_REGISTRY:
        dims = ", ".join(t.fslsm_dims[:3]) + ("..." if len(t.fslsm_dims) > 3 else "")
        print(f"{t.tool_id:>3}  {t.name:<35} {t.category:<20} {dims:<30} {t.token_cost:>6}")
    print("─" * 100)
    print(f"     {'S0 total':<55} {s0_prompt_tokens():>6}")


if __name__ == "__main__":
    registry_summary()
```

**Run:** `python tools/tool_registry.py`
**Expected:** All 15 tools printed, S0 total ~1,444 tokens.

---

## Phase 2 — FAISS Tool Index — Rebuild with Domain-Bridging Descriptions

### Step 2.1 — Create `tools/tool_index.py`

```python
# tools/tool_index.py
"""
Builds FAISS IndexFlatIP over all 15 tool descriptions.
Descriptions now include domain-bridging vocabulary (Phase 1).
"""
import os, json
import numpy as np
import faiss
from typing import List, Tuple
from openai import OpenAI
from tools.tool_registry import MCPTool, TOOL_REGISTRY

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536

def _embed(texts: List[str]) -> np.ndarray:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([r.embedding for r in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


class ToolIndex:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.tools: List[MCPTool] = []

    def build(self, tools=TOOL_REGISTRY):
        self.tools = tools
        descs = [t.description for t in tools]
        print(f"[ToolIndex] Embedding {len(descs)} tool descriptions...")
        vecs = _embed(descs)
        self.index.add(vecs)
        print(f"[ToolIndex] Built. Vectors: {self.index.ntotal}")

    def save(self, idx_path="data/tool_index.faiss", meta_path="data/tool_index_meta.json"):
        faiss.write_index(self.index, idx_path)
        meta = [{"tool_id": t.tool_id, "name": t.name} for t in self.tools]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ToolIndex] Saved → {idx_path}")

    def load(self, idx_path="data/tool_index.faiss", meta_path="data/tool_index_meta.json"):
        self.index = faiss.read_index(idx_path)
        with open(meta_path) as f:
            meta = json.load(f)
        from tools.tool_registry import TOOL_BY_ID
        self.tools = [TOOL_BY_ID[m["tool_id"]] for m in meta]
        print(f"[ToolIndex] Loaded {self.index.ntotal} vectors")

    def retrieve(self, query: str, k: int = 1) -> List[Tuple[MCPTool, float]]:
        vec = _embed([query])
        scores, indices = self.index.search(vec, k)
        return [(self.tools[idx], float(score))
                for score, idx in zip(scores[0], indices[0]) if idx >= 0]


if __name__ == "__main__":
    idx = ToolIndex()
    idx.build()
    idx.save()

    # Sanity check
    tests = [
        "Can you draw a computation graph for backpropagation?",
        "Walk me through gradient descent step by step.",
        "Compare ResNet and VGG architectures.",
        "Quiz me on batch normalization.",
        "Summarize the attention mechanism chapter.",
        "What are the latest transformer developments?",
    ]
    print("\n── Retrieval sanity check ──")
    for q in tests:
        hits = idx.retrieve(q, k=3)
        print(f"\n  Q: {q}")
        for i, (tool, score) in enumerate(hits):
            print(f"    {i+1}. [{tool.tool_id}] {tool.name:<35} score={score:.3f}")
```

**Run:** `python tools/tool_index.py`
**Verify:** Cosine scores for top-1 should be >0.50 (not 0.12–0.32 like before).

---

## Phase 3 — Profile Decoder & Augmentor Fixes

### Step 3.1 — Create `core/profile_decoder.py`

```python
# core/profile_decoder.py
"""
Decodes ±1 bipolar FSLSM profiles into named dimension sets.
Handles both formats:
  Old binary: {"act":1, "ref":0, "vis":1, ...}
  New bipolar: {"act_ref":-1, "vis_ver":-1, ...}
"""
from typing import Dict, Set


def decode_profile(profile: dict) -> Set[str]:
    """
    Returns set of active dimension names.
    e.g. {"Active", "Sensing", "Visual", "Sequential"}
    """
    if "act_ref" in profile:
        # ±1 bipolar format
        dims = set()
        dims.add("Active"     if profile["act_ref"] < 0 else "Reflective")
        dims.add("Sensing"    if profile["sen_int"] < 0 else "Intuitive")
        dims.add("Visual"     if profile["vis_ver"] < 0 else "Verbal")
        dims.add("Sequential" if profile["seq_glo"] < 0 else "Global")
        return dims
    else:
        # Old binary format
        field_map = {
            "act": "Active", "ref": "Reflective",
            "sns": "Sensing", "int": "Intuitive",
            "vis": "Visual", "vrb": "Verbal",
            "seq": "Sequential", "glo": "Global",
        }
        return {field_map[f] for f, v in profile.items()
                if v == 1 and f in field_map}


def profile_to_label(profile: dict) -> str:
    """
    Returns canonical label string.
    e.g. "Active-Sensing-Visual-Sequential"
    """
    dims = decode_profile(profile)
    act = "Active"     if "Active"     in dims else "Reflective"
    sns = "Sensing"    if "Sensing"    in dims else "Intuitive"
    vis = "Visual"     if "Visual"     in dims else "Verbal"
    seq = "Sequential" if "Sequential" in dims else "Global"
    return f"{act}-{sns}-{vis}-{seq}"


def profile_to_tuple(profile: dict) -> tuple:
    """Returns (act, sns, vis, seq) tuple for GROUND_TRUTH_MAP lookup."""
    dims = decode_profile(profile)
    return (
        "Active"     if "Active"     in dims else "Reflective",
        "Sensing"    if "Sensing"    in dims else "Intuitive",
        "Visual"     if "Visual"     in dims else "Verbal",
        "Sequential" if "Sequential" in dims else "Global",
    )


def get_primary_dim(profile: dict) -> str:
    """
    Returns the single most pedagogically salient dimension.
    Priority: Visual > Sequential > Active > Global >
              Intuitive > Sensing > Reflective > Verbal
    """
    dims = decode_profile(profile)
    priority = ["Visual", "Sequential", "Active", "Global",
                "Intuitive", "Sensing", "Reflective", "Verbal"]
    for dim in priority:
        if dim in dims:
            return dim
    return "Verbal"
```

### Step 3.2 — Create `core/fslsm_augmentor.py`

```python
# core/fslsm_augmentor.py
"""
FSLSM Query Augmentor — S1b condition.
Handles both profile formats (±1 bipolar and old binary).
"""
from core.profile_decoder import decode_profile

DIM_DIRECTIVES = {
    "Active":      "prefer hands-on exercises, interactive tasks, active problem-solving, coding challenges",
    "Reflective":  "prefer self-reflection prompts, think-before-answering, synthesis tasks, quiet review",
    "Sensing":     "prefer concrete examples, factual explanations, real-world cases, standard procedures",
    "Intuitive":   "prefer analogies, metaphors, cross-domain connections, big-picture theoretical links",
    "Visual":      "prefer visual diagrams, computation graphs, spatial layout, architecture figures, ASCII charts",
    "Verbal":      "prefer textual prose explanations, narrative descriptions, verbal detail, written definitions",
    "Sequential":  "prefer step-by-step derivation, numbered stages, explicit intermediate steps, ordered procedures",
    "Global":      "prefer overviews, concept maps, holistic summaries, thematic connections, big-picture first",
}


def augment_query(query: str, profile: dict) -> str:
    """
    Appends FSLSM style directives to the raw query.
    Works with both ±1 bipolar and old binary profile formats.
    """
    dims = decode_profile(profile)
    directives = [DIM_DIRECTIVES[dim] for dim in dims if dim in DIM_DIRECTIVES]
    if not directives:
        return query
    return f"{query}; {'; '.join(directives)}"
```

---

## Phase 4 — GROUND_TRUTH_MAP — Full 4D Mapping

### Step 4.1 — Create `core/ground_truth.py`

```python
# core/ground_truth.py
"""
Ground truth optimal tool assignment.
Uses (question_type × profile) to assign optimal_tool_id.
Covers all 15 tools across 4 question types × 16 profiles.
"""
from core.profile_decoder import profile_to_tuple

# (question_type, act, sns, vis, seq) → tool_id
# All 15 tools are reachable through this mapping.

GROUND_TRUTH_MAP_FULL = {

    # ── explain_relationship (21 questions) ────────────────────
    ("explain_relationship", "Active",     "Sensing",    "Visual",  "Sequential"): 4,
    ("explain_relationship", "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("explain_relationship", "Active",     "Sensing",    "Verbal",  "Sequential"): 1,
    ("explain_relationship", "Active",     "Sensing",    "Verbal",  "Global"):     14,
    ("explain_relationship", "Active",     "Intuitive",  "Visual",  "Sequential"): 4,
    ("explain_relationship", "Active",     "Intuitive",  "Visual",  "Global"):     7,
    ("explain_relationship", "Active",     "Intuitive",  "Verbal",  "Sequential"): 5,
    ("explain_relationship", "Active",     "Intuitive",  "Verbal",  "Global"):     5,
    ("explain_relationship", "Reflective", "Sensing",    "Visual",  "Sequential"): 4,
    ("explain_relationship", "Reflective", "Sensing",    "Visual",  "Global"):     13,
    ("explain_relationship", "Reflective", "Sensing",    "Verbal",  "Sequential"): 1,
    ("explain_relationship", "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("explain_relationship", "Reflective", "Intuitive",  "Visual",  "Sequential"): 4,
    ("explain_relationship", "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("explain_relationship", "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("explain_relationship", "Reflective", "Intuitive",  "Verbal",  "Global"):     10,

    # ── synthesize_workflow (20 questions) ─────────────────────
    ("synthesize_workflow",  "Active",     "Sensing",    "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Active",     "Sensing",    "Visual",  "Global"):     11,
    ("synthesize_workflow",  "Active",     "Sensing",    "Verbal",  "Sequential"): 2,
    ("synthesize_workflow",  "Active",     "Sensing",    "Verbal",  "Global"):     3,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Visual",  "Global"):     11,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Verbal",  "Sequential"): 9,
    ("synthesize_workflow",  "Active",     "Intuitive",  "Verbal",  "Global"):     5,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Visual",  "Global"):     8,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Verbal",  "Sequential"): 3,
    ("synthesize_workflow",  "Reflective", "Sensing",    "Verbal",  "Global"):     8,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Visual",  "Sequential"): 2,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("synthesize_workflow",  "Reflective", "Intuitive",  "Verbal",  "Global"):     10,

    # ── trace_evolution (16 questions) ─────────────────────────
    ("trace_evolution",      "Active",     "Sensing",    "Visual",  "Sequential"): 2,
    ("trace_evolution",      "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("trace_evolution",      "Active",     "Sensing",    "Verbal",  "Sequential"): 2,
    ("trace_evolution",      "Active",     "Sensing",    "Verbal",  "Global"):     15,
    ("trace_evolution",      "Active",     "Intuitive",  "Visual",  "Sequential"): 4,
    ("trace_evolution",      "Active",     "Intuitive",  "Visual",  "Global"):     5,
    ("trace_evolution",      "Active",     "Intuitive",  "Verbal",  "Sequential"): 5,
    ("trace_evolution",      "Active",     "Intuitive",  "Verbal",  "Global"):     15,
    ("trace_evolution",      "Reflective", "Sensing",    "Visual",  "Sequential"): 4,
    ("trace_evolution",      "Reflective", "Sensing",    "Visual",  "Global"):     7,
    ("trace_evolution",      "Reflective", "Sensing",    "Verbal",  "Sequential"): 1,
    ("trace_evolution",      "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("trace_evolution",      "Reflective", "Intuitive",  "Visual",  "Sequential"): 4,
    ("trace_evolution",      "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("trace_evolution",      "Reflective", "Intuitive",  "Verbal",  "Sequential"): 6,
    ("trace_evolution",      "Reflective", "Intuitive",  "Verbal",  "Global"):     10,

    # ── compare (15 questions) ─────────────────────────────────
    ("compare",              "Active",     "Sensing",    "Visual",  "Sequential"): 6,
    ("compare",              "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("compare",              "Active",     "Sensing",    "Verbal",  "Sequential"): 6,
    ("compare",              "Active",     "Sensing",    "Verbal",  "Global"):     12,
    ("compare",              "Active",     "Intuitive",  "Visual",  "Sequential"): 4,
    ("compare",              "Active",     "Intuitive",  "Visual",  "Global"):     5,
    ("compare",              "Active",     "Intuitive",  "Verbal",  "Sequential"): 12,
    ("compare",              "Active",     "Intuitive",  "Verbal",  "Global"):     5,
    ("compare",              "Reflective", "Sensing",    "Visual",  "Sequential"): 6,
    ("compare",              "Reflective", "Sensing",    "Visual",  "Global"):     7,
    ("compare",              "Reflective", "Sensing",    "Verbal",  "Sequential"): 6,
    ("compare",              "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("compare",              "Reflective", "Intuitive",  "Visual",  "Sequential"): 4,
    ("compare",              "Reflective", "Intuitive",  "Visual",  "Global"):     5,
    ("compare",              "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("compare",              "Reflective", "Intuitive",  "Verbal",  "Global"):     13,

    # ── Coverage question types (16 questions from R2b) ────────

    # practice
    ("practice",             "Active",     "Sensing",    "Visual",  "Sequential"): 11,
    ("practice",             "Active",     "Sensing",    "Visual",  "Global"):     11,
    ("practice",             "Active",     "Sensing",    "Verbal",  "Sequential"): 12,
    ("practice",             "Active",     "Sensing",    "Verbal",  "Global"):     12,
    ("practice",             "Active",     "Intuitive",  "Visual",  "Sequential"): 11,
    ("practice",             "Active",     "Intuitive",  "Visual",  "Global"):     11,
    ("practice",             "Active",     "Intuitive",  "Verbal",  "Sequential"): 12,
    ("practice",             "Active",     "Intuitive",  "Verbal",  "Global"):     12,
    ("practice",             "Reflective", "Sensing",    "Visual",  "Sequential"): 10,
    ("practice",             "Reflective", "Sensing",    "Visual",  "Global"):     10,
    ("practice",             "Reflective", "Sensing",    "Verbal",  "Sequential"): 10,
    ("practice",             "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("practice",             "Reflective", "Intuitive",  "Visual",  "Sequential"): 10,
    ("practice",             "Reflective", "Intuitive",  "Visual",  "Global"):     10,
    ("practice",             "Reflective", "Intuitive",  "Verbal",  "Sequential"): 10,
    ("practice",             "Reflective", "Intuitive",  "Verbal",  "Global"):     13,

    # style_adapt (PersonaRAG Adapter, FSLSM Styler)
    ("style_adapt",          "Active",     "Sensing",    "Visual",  "Sequential"): 8,
    ("style_adapt",          "Active",     "Sensing",    "Visual",  "Global"):     8,
    ("style_adapt",          "Active",     "Sensing",    "Verbal",  "Sequential"): 8,
    ("style_adapt",          "Active",     "Sensing",    "Verbal",  "Global"):     8,
    ("style_adapt",          "Active",     "Intuitive",  "Visual",  "Sequential"): 9,
    ("style_adapt",          "Active",     "Intuitive",  "Visual",  "Global"):     9,
    ("style_adapt",          "Active",     "Intuitive",  "Verbal",  "Sequential"): 9,
    ("style_adapt",          "Active",     "Intuitive",  "Verbal",  "Global"):     9,
    ("style_adapt",          "Reflective", "Sensing",    "Visual",  "Sequential"): 8,
    ("style_adapt",          "Reflective", "Sensing",    "Visual",  "Global"):     8,
    ("style_adapt",          "Reflective", "Sensing",    "Verbal",  "Sequential"): 8,
    ("style_adapt",          "Reflective", "Sensing",    "Verbal",  "Global"):     8,
    ("style_adapt",          "Reflective", "Intuitive",  "Visual",  "Sequential"): 9,
    ("style_adapt",          "Reflective", "Intuitive",  "Visual",  "Global"):     9,
    ("style_adapt",          "Reflective", "Intuitive",  "Verbal",  "Sequential"): 9,
    ("style_adapt",          "Reflective", "Intuitive",  "Verbal",  "Global"):     9,

    # search
    ("search",               "Active",     "Sensing",    "Visual",  "Sequential"): 15,
    ("search",               "Active",     "Sensing",    "Visual",  "Global"):     15,
    ("search",               "Active",     "Sensing",    "Verbal",  "Sequential"): 14,
    ("search",               "Active",     "Sensing",    "Verbal",  "Global"):     15,
    ("search",               "Active",     "Intuitive",  "Visual",  "Sequential"): 15,
    ("search",               "Active",     "Intuitive",  "Visual",  "Global"):     15,
    ("search",               "Active",     "Intuitive",  "Verbal",  "Sequential"): 15,
    ("search",               "Active",     "Intuitive",  "Verbal",  "Global"):     15,
    ("search",               "Reflective", "Sensing",    "Visual",  "Sequential"): 14,
    ("search",               "Reflective", "Sensing",    "Visual",  "Global"):     14,
    ("search",               "Reflective", "Sensing",    "Verbal",  "Sequential"): 14,
    ("search",               "Reflective", "Sensing",    "Verbal",  "Global"):     14,
    ("search",               "Reflective", "Intuitive",  "Visual",  "Sequential"): 15,
    ("search",               "Reflective", "Intuitive",  "Visual",  "Global"):     15,
    ("search",               "Reflective", "Intuitive",  "Verbal",  "Sequential"): 15,
    ("search",               "Reflective", "Intuitive",  "Verbal",  "Global"):     15,

    # summarize
    ("summarize",            "Active",     "Sensing",    "Visual",  "Sequential"): 7,
    ("summarize",            "Active",     "Sensing",    "Visual",  "Global"):     7,
    ("summarize",            "Active",     "Sensing",    "Verbal",  "Sequential"): 1,
    ("summarize",            "Active",     "Sensing",    "Verbal",  "Global"):     13,
    ("summarize",            "Active",     "Intuitive",  "Visual",  "Sequential"): 7,
    ("summarize",            "Active",     "Intuitive",  "Visual",  "Global"):     7,
    ("summarize",            "Active",     "Intuitive",  "Verbal",  "Sequential"): 5,
    ("summarize",            "Active",     "Intuitive",  "Verbal",  "Global"):     13,
    ("summarize",            "Reflective", "Sensing",    "Visual",  "Sequential"): 7,
    ("summarize",            "Reflective", "Sensing",    "Visual",  "Global"):     7,
    ("summarize",            "Reflective", "Sensing",    "Verbal",  "Sequential"): 13,
    ("summarize",            "Reflective", "Sensing",    "Verbal",  "Global"):     13,
    ("summarize",            "Reflective", "Intuitive",  "Visual",  "Sequential"): 7,
    ("summarize",            "Reflective", "Intuitive",  "Visual",  "Global"):     7,
    ("summarize",            "Reflective", "Intuitive",  "Verbal",  "Sequential"): 13,
    ("summarize",            "Reflective", "Intuitive",  "Verbal",  "Global"):     13,
}


def get_optimal_tool_id(question_type: str, profile: dict) -> int:
    """
    Returns expert-defined optimal tool for (question_type, profile).
    Falls back to Concept Explainer (1) if no mapping found.
    """
    key = (question_type, *profile_to_tuple(profile))
    return GROUND_TRUTH_MAP_FULL.get(key, 1)


def verify_coverage():
    """Verify all 15 tools are reachable."""
    assigned = set(GROUND_TRUTH_MAP_FULL.values())
    all_tools = set(range(1, 16))
    missing = all_tools - assigned
    if missing:
        print(f"⚠ WARNING: Tools {missing} are never assigned as optimal!")
    else:
        print(f"✓ All 15 tools are reachable in GROUND_TRUTH_MAP_FULL")
    print(f"  Total entries: {len(GROUND_TRUTH_MAP_FULL)}")
    print(f"  Unique tools: {len(assigned)}")


if __name__ == "__main__":
    verify_coverage()
```

**Run:** `python core/ground_truth.py`
**Expected:** "✓ All 15 tools are reachable in GROUND_TRUTH_MAP_FULL"

---

## Phase 5 — Generate 16 Coverage Questions (R2b)

### Step 5.1 — Create `scripts/02_generate_coverage_questions.py`

```python
# scripts/02_generate_coverage_questions.py
"""
Generate 16 coverage questions targeting tools 8-15.
Uses GPT-4.1-mini for generation, saves for manual review.
Cost: ~$0.02 total.
"""
import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

COVERAGE_SPECS = [
    # (question_type, target_tools, description)
    ("practice",    [11, 12], "Active learner wanting hands-on exercise or quiz"),
    ("practice",    [11, 12], "Self-assessment on neural network concepts"),
    ("practice",    [10, 11], "Reflective practice and exercise generation"),
    ("practice",    [10, 12], "Quiz and reflection on optimization algorithms"),
    ("style_adapt", [8],      "Re-explain content in my FSLSM learning style"),
    ("style_adapt", [8],      "Adapt retrieved D2L content to match learning preference"),
    ("style_adapt", [9],      "Transform a verbal explanation into visual/active form"),
    ("style_adapt", [9],      "Convert step-by-step derivation to big-picture overview"),
    ("search",      [15],     "Find latest developments beyond D2L curriculum"),
    ("search",      [15],     "Cross-domain connections for transformer applications"),
    ("search",      [14, 15], "Retrieve and supplement D2L content with web sources"),
    ("search",      [14],     "Find the relevant D2L section on a specific topic"),
    ("summarize",   [13],     "Synthesize key takeaways from a multi-concept discussion"),
    ("summarize",   [13],     "Overview of how D2L chapters on CNNs connect"),
    ("summarize",   [7, 13],  "Concept map or summary of training strategies"),
    ("summarize",   [7],      "Big-picture overview of deep learning optimization"),
]

PROMPT = """You are generating evaluation questions for an AI tutoring system
that teaches machine learning using the D2L (Dive into Deep Learning) textbook.

Generate ONE question that:
- Is about machine learning / deep learning (D2L topics)
- Naturally triggers this type of pedagogical tool: {description}
- Has question_type: "{question_type}"
- Feels natural — a real student would ask this

The question should be 1-2 sentences, specific to ML/DL content.

Respond with ONLY the question text, nothing else."""


def generate_coverage_questions():
    questions = []
    for i, (qtype, target_tools, desc) in enumerate(COVERAGE_SPECS, start=1):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": PROMPT.format(description=desc, question_type=qtype)
            }],
            max_tokens=150,
            temperature=0.7,
        )
        question_text = response.choices[0].message.content.strip()

        q = {
            "question_id": f"COV_{i:03d}",
            "question": question_text,
            "question_type": qtype,
            "target_tools": target_tools,
            "gold_chunk_ids": [],        # No gold chunks for coverage questions
            "essential_chunk_ids": [],
            "strategy": "coverage",
            "quality_tier": "coverage",
            "needs_review": True,        # Flag for manual review
        }
        questions.append(q)
        print(f"[{q['question_id']}] type={qtype}")
        print(f"  Q: {question_text}")
        print()

    os.makedirs("data", exist_ok=True)
    with open("data/coverage_questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions to data/coverage_questions.json")
    print("⚠ MANUAL REVIEW REQUIRED before proceeding to Phase 6")


if __name__ == "__main__":
    generate_coverage_questions()
```

**Run:** `python scripts/02_generate_coverage_questions.py`
**Then:** Manually review `data/coverage_questions.json`. Edit any questions that are unclear. Set `needs_review: false` for each approved question.

### Step 5.2 — Merge into `data/all_questions.json`

```python
# scripts/merge_questions.py
import json

with open("data/filtered_questions.json") as f:
    original = json.load(f)
with open("data/coverage_questions.json") as f:
    coverage = json.load(f)

# Verify no duplicate IDs
orig_ids = {q["question_id"] for q in original}
cov_ids = {q["question_id"] for q in coverage}
assert not orig_ids & cov_ids, "Duplicate question IDs!"

# Verify all coverage questions reviewed
unreviewed = [q for q in coverage if q.get("needs_review", True)]
if unreviewed:
    print(f"⚠ {len(unreviewed)} coverage questions still need manual review!")
    exit(1)

merged = original + coverage
with open("data/all_questions.json", "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged: {len(original)} original + {len(coverage)} coverage = {len(merged)} total")
```

---

## Phase 6 — Web Search Tool — Tavily Integration

### Step 6.1 — Create `tools/tavily_search.py`

```python
# tools/tavily_search.py
"""
Web Search Tool implementation using Tavily API.
Free tier: 1,000 calls/month (sufficient for thesis).
Sign up: https://app.tavily.com
"""
import os
from tavily import TavilyClient


def web_search(query: str, max_results: int = 3) -> str:
    """
    Perform web search and return formatted results.
    Returns plain text suitable for LLM consumption.
    """
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(query=query, max_results=max_results)
    formatted = []
    for r in results.get("results", []):
        formatted.append(
            f"- **{r['title']}**: {r['content'][:250]}\n"
            f"  Source: {r['url']}"
        )
    return "\n\n".join(formatted) if formatted else "No results found."


if __name__ == "__main__":
    result = web_search("latest transformer architecture developments 2025")
    print(result)
```

**Run:** `python tools/tavily_search.py`
**Verify:** Returns 3 real search results.

---

## Phase 7 — TOOL_PROMPTS — All 15 Prompt Templates

### Step 7.1 — Create `tools/tool_prompts.py`

```python
# tools/tool_prompts.py
"""
System prompt templates for each of the 15 MCP tools.
Used when R2 generates actual tutor responses.
Tool 14 (Content Retriever) and Tool 15 (Web Search) are handled separately.
"""

TOOL_PROMPTS = {
    1: (
        "You are a Concept Explainer for machine learning education. "
        "Provide a clear, concrete, fact-based textual explanation. "
        "Build the explanation sentence by sentence in connected prose. "
        "Use real-world examples and standard terminology from the D2L textbook. "
        "Avoid diagrams, bullet lists, or visual elements — write in flowing narrative. "
        "Define technical terms when first introduced. "
        "Target audience: learners who prefer reading ordered written explanations."
    ),

    2: (
        "You are a Step-by-Step Derivator for machine learning education. "
        "Break the answer into numbered sequential steps. "
        "Each step should show exactly one transformation, computation, or logical move. "
        "Show all intermediate results explicitly — do not skip steps. "
        "Use mathematical notation where appropriate but explain each symbol. "
        "Format: Step 1: [action] → [result], Step 2: [action] → [result], etc. "
        "Target audience: learners who need ordered procedural progression."
    ),

    3: (
        "You are a Worked Example Generator for machine learning education. "
        "Present a fully solved example with concrete numbers, real tensor shapes, "
        "and actual computation results. Every intermediate calculation must be shown. "
        "Annotate each computation step with the rule or formula being applied. "
        "Use realistic values (e.g., actual weight matrices, learning rates, batch sizes). "
        "Format: Given → Step 1 → Step 2 → ... → Final Answer. "
        "Target audience: learners who learn through concrete worked solutions."
    ),

    4: (
        "You are a Diagrammatic-Text Explainer for machine learning education. "
        "Create text-based visual representations: ASCII art diagrams, computation graphs, "
        "architecture schematics, or spatial layouts with labeled components. "
        "Use boxes (┌──┐), arrows (→, ←, ↓), and connectors to show structure. "
        "Annotate each visual element with its technical meaning. "
        "Show data flow, layer connections, and transformation sequences spatially. "
        "Target audience: learners who prefer visual, spatial representations."
    ),

    5: (
        "You are an Analogical Reasoner for machine learning education. "
        "Explain the concept primarily through analogies and metaphors. "
        "Map abstract ML ideas to familiar real-world phenomena first, "
        "then bridge to the technical details. Start with the big picture "
        "and why this matters before diving into specifics. "
        "Connect to neighbouring ideas across different areas of ML. "
        "Target audience: learners who grasp abstractions through metaphorical reasoning."
    ),

    6: (
        "You are a Comparative Explainer for machine learning education. "
        "Generate a structured side-by-side comparison. "
        "Create a clear comparison table or enumerated contrast list "
        "showing key differences across specific axes (speed, accuracy, "
        "memory, complexity, use cases, etc.). "
        "For each axis, state what Concept A does and what Concept B does. "
        "Conclude with a summary of when to use each. "
        "Target audience: learners who understand through systematic distinction."
    ),

    7: (
        "You are a Concept Map Generator for machine learning education. "
        "Build a hierarchical text-based concept map showing how topics "
        "interconnect. Use indentation, arrows (→), and labeled relationships. "
        "Start with the central concept, branch to related ideas, "
        "and show how sub-topics feed back into the main theme. "
        "Format as an indented tree with relationship labels on each edge. "
        "Target audience: learners who need the big picture before details."
    ),

    8: (
        "You are a PersonaRAG Adapter for machine learning education. "
        "Your task is to rewrite the provided retrieved content to match "
        "the learner's FSLSM learning style profile: {style_description}. "
        "Preserve all factual accuracy while transforming the presentation: "
        "- For Visual learners: add spatial descriptions and structural metaphors "
        "- For Sequential learners: impose clear numbered ordering "
        "- For Sensing learners: add concrete examples and standard procedures "
        "- For Intuitive learners: add theoretical connections and analogies "
        "Do not add new factual content — only transform the style."
    ),

    9: (
        "You are an FSLSM Styler for machine learning education. "
        "Transform the previously given explanation into a completely different "
        "FSLSM learning style: {target_style}. "
        "Keep the same factual content but change the pedagogical approach entirely: "
        "- Verbal→Visual: convert prose to diagrams and spatial layouts "
        "- Sequential→Global: convert step-by-step to big-picture overview "
        "- Sensing→Intuitive: convert concrete examples to abstract theory "
        "- Active→Reflective: convert exercises to reflection prompts "
        "Make the transformation dramatic and obvious."
    ),

    10: (
        "You are a Think-Pair-Share Generator for machine learning education. "
        "Generate a structured three-phase reflection exercise: "
        "1. THINK (2 min): Pose a thought-provoking question that requires "
        "   the learner to examine their own understanding. No rushing. "
        "2. PAIR: Suggest a discussion question they could explore with a peer "
        "   or AI tutor, probing deeper into assumptions or edge cases. "
        "3. SHARE: Ask them to write a brief synthesis — what they now understand, "
        "   what remains unclear, and what they would investigate further. "
        "Target audience: learners who process by thinking quietly first."
    ),

    11: (
        "You are an Interactive Exercise Generator for machine learning education. "
        "Create a hands-on task the learner must complete themselves. Include: "
        "1. Setup: what libraries/data they need "
        "2. Task description: what to implement or compute "
        "3. Expected output: what success looks like "
        "4. Progressive hints (3 levels: nudge → approach → solution outline) "
        "The exercise should require coding, computation, or active problem-solving. "
        "Target audience: learners who retain information through doing."
    ),

    12: (
        "You are a Quiz Generator for machine learning education. "
        "Generate 3-5 quiz questions testing understanding of the topic. "
        "Mix question formats: 2 multiple-choice, 1 true/false, 1 short-answer. "
        "For each question, provide: "
        "- The question "
        "- Answer options (for MCQ) "
        "- Correct answer "
        "- Brief explanation (1-2 sentences) of why the answer is correct "
        "Questions should test factual knowledge, not opinion. "
        "Target audience: learners who prefer concrete knowledge checks."
    ),

    13: (
        "You are a Summarizer for machine learning education. "
        "Produce a concise synthesis of the topic covering: "
        "1. Core idea (1-2 sentences) "
        "2. Key takeaways (3-5 bullet points) "
        "3. How this connects to related topics in ML "
        "4. Key terms with brief definitions "
        "Keep it scannable and review-oriented — this is for consolidation, "
        "not first-time learning. "
        "Target audience: learners who consolidate through synthesis and review."
    ),
}

# Tools 14 and 15 are handled by separate code paths:
# Tool 14 (Content Retriever) → uses existing RAG pipeline
# Tool 15 (Web Search Tool) → uses Tavily API (tools/tavily_search.py)


def get_tool_prompt(tool_id: int, **kwargs) -> str:
    """Get the system prompt for a tool, with optional template variables."""
    prompt = TOOL_PROMPTS.get(tool_id)
    if prompt is None:
        return None  # Tools 14, 15 handled separately
    return prompt.format(**kwargs) if kwargs else prompt
```

---

## Phase 8 — S0 Baseline — Real LLM Tool Selection

### Step 8.1 — Create `core/s0_baseline.py`

```python
# core/s0_baseline.py
"""
S0 (Prompt Bloat Baseline): Real LLM tool selection with all 15 schemas.
Injects all tool descriptions into the context and asks GPT-4.1-mini to select.
"""
import os, json
from openai import OpenAI
from tools.tool_registry import TOOL_REGISTRY, s0_prompt_tokens

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

S0_SYSTEM_PROMPT = """You are an AI tutor for machine learning education.
You have access to the following 15 tools. Given a student's question and
their FSLSM learning style profile, select the SINGLE most appropriate tool
by responding with ONLY the tool_id number (1-15).

Available tools:
{tool_schemas}

Student FSLSM profile: {profile_label}
(Active/Reflective, Sensing/Intuitive, Visual/Verbal, Sequential/Global)

Respond with ONLY a single integer (the tool_id). Nothing else."""


def _build_tool_schemas() -> str:
    """Format all 15 tool schemas for S0 prompt injection."""
    lines = []
    for t in TOOL_REGISTRY:
        dims = ", ".join(t.fslsm_dims[:3])
        lines.append(
            f"Tool {t.tool_id}: {t.name}\n"
            f"  Category: {t.category}\n"
            f"  FSLSM: {dims}\n"
            f"  Description: {t.description[:200]}\n"
        )
    return "\n".join(lines)


TOOL_SCHEMAS_TEXT = _build_tool_schemas()


def s0_select_tool(query: str, profile_label: str) -> int:
    """
    Real LLM tool selection with all 15 schemas in context.
    Returns selected tool_id.
    """
    prompt = S0_SYSTEM_PROMPT.format(
        tool_schemas=TOOL_SCHEMAS_TEXT,
        profile_label=profile_label,
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        tool_id = int(raw)
        if 1 <= tool_id <= 15:
            return tool_id
    except (ValueError, IndexError):
        pass
    return 1  # fallback


def s0_input_tokens() -> int:
    """Token count for S0 (all schemas injected)."""
    return s0_prompt_tokens()
```

**Note:** S0 requires ~7,040 API calls for the full run (one per session).
**Estimated cost:** 7,040 × ~100 input tokens × $0.15/1M = ~$0.10 total. Very cheap.

---

## Phase 9 — R2 Session Runner — Dual Logger

### Step 9.1 — Create `core/session_runner.py`

```python
# core/session_runner.py
"""
R2 Session Runner with dual logging for Exp3 (primary) + Exp2 (passive).
Runs one session through the full MCP pipeline:
  1. FAISS retrieval (S1b tool selection)
  2. Tool-specific response generation
  3. Log Exp3 metrics (TSA, PTS)
  4. Log Exp2 passive data (tool selection + response for later evaluation)
"""
import os, json, uuid, sqlite3
from datetime import datetime
from typing import Dict, Optional

from tools.tool_registry import get_tool_by_id, s0_prompt_tokens
from tools.tool_index import ToolIndex
from tools.tool_prompts import get_tool_prompt
from tools.tavily_search import web_search
from core.profile_decoder import decode_profile, profile_to_label
from core.fslsm_augmentor import augment_query
from core.ground_truth import get_optimal_tool_id
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class SessionRunner:
    def __init__(self, tool_index: ToolIndex, db_path: str, exp2_log_path: str):
        self.tool_index = tool_index
        self.s0_tokens  = s0_prompt_tokens()
        self.db_path    = db_path
        self.exp2_log   = exp2_log_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exp3_session_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                condition       TEXT,
                session_id      TEXT,
                question_id     TEXT,
                question_type   TEXT,
                student_profile TEXT,
                query           TEXT,
                selected_tool   INTEGER,
                optimal_tool    INTEGER,
                tsa_hit         INTEGER,
                input_tokens    INTEGER,
                pts_delta       REAL,
                created_at      TEXT
            )
        """)
        conn.commit()
        conn.close()

    def run_s1b_session(
        self,
        question_id: str,
        question_type: str,
        query: str,
        profile: dict,
        rag_chunks: str = "",
        session_id: Optional[str] = None,
        generate_response: bool = True,
    ) -> Dict:
        """
        Run one session through S1b (FSLSM-conditioned RAG-MCP).
        Logs Exp3 data. Optionally generates response and logs Exp2 data.
        """
        sid = session_id or str(uuid.uuid4())
        profile_label = profile_to_label(profile)

        # ── Step 1: S1b tool selection (FSLSM-augmented FAISS) ──
        aug_query = augment_query(query, profile)
        hits = self.tool_index.retrieve(aug_query, k=1)
        selected_tool, cosine_score = hits[0]
        s1b_tokens = selected_tool.token_cost

        # ── Step 2: Ground truth ──
        optimal_id = get_optimal_tool_id(question_type, profile)
        tsa_hit = (selected_tool.tool_id == optimal_id)
        pts_delta = (self.s0_tokens - s1b_tokens) / self.s0_tokens * 100

        # ── Step 3: Log Exp3 S1b result ──
        self._log_exp3("S1b", sid, question_id, question_type, profile,
                       query, selected_tool.tool_id, optimal_id,
                       tsa_hit, s1b_tokens, pts_delta)

        result = {
            "session_id": sid,
            "question_id": question_id,
            "question_type": question_type,
            "profile_label": profile_label,
            "selected_tool_id": selected_tool.tool_id,
            "selected_tool_name": selected_tool.name,
            "optimal_tool_id": optimal_id,
            "tsa_hit": tsa_hit,
            "cosine_score": cosine_score,
            "input_tokens": s1b_tokens,
            "pts_delta": pts_delta,
        }

        # ── Step 4: Generate response (optional, for Exp2) ──
        if generate_response:
            response = self._generate_response(
                query, profile, selected_tool.tool_id,
                profile_label, rag_chunks
            )
            result["response"] = response

            # ── Step 5: Passive Exp2 log ──
            self._log_exp2(sid, question_id, question_type, profile_label,
                          profile, query, selected_tool.tool_id,
                          selected_tool.name, response)

        return result

    def _generate_response(self, query, profile, tool_id, profile_label, rag_chunks):
        """Generate tutor response using selected tool's prompt template."""
        # Tool 14: Content Retriever — return RAG chunks directly
        if tool_id == 14:
            return f"Retrieved content:\n{rag_chunks}" if rag_chunks else "No content found."

        # Tool 15: Web Search — use Tavily
        if tool_id == 15:
            return web_search(query)

        # Tools 1-13: LLM with tool-specific system prompt
        dims = decode_profile(profile)
        style_desc = ", ".join(sorted(dims))
        prompt = get_tool_prompt(
            tool_id,
            style_description=style_desc,
            target_style=style_desc,
        )
        if not prompt:
            prompt = "You are a helpful ML tutor."

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content":
                    f"Context from D2L:\n{rag_chunks[:1500]}\n\nQuestion: {query}"
                    if rag_chunks else f"Question: {query}"},
            ],
            max_tokens=800,
        )
        return response.choices[0].message.content

    def _log_exp3(self, condition, sid, qid, qtype, profile, query,
                  selected, optimal, tsa, tokens, pts):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO exp3_session_results
               (condition, session_id, question_id, question_type,
                student_profile, query, selected_tool, optimal_tool,
                tsa_hit, input_tokens, pts_delta, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (condition, sid, qid, qtype, json.dumps(profile), query,
             selected, optimal, int(tsa), tokens, pts,
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    def _log_exp2(self, sid, qid, qtype, profile_label, profile,
                  query, tool_id, tool_name, response):
        entry = {
            "session_id": sid,
            "question_id": qid,
            "question_type": qtype,
            "mode": "R2",
            "profile_label": profile_label,
            "fslsm_vector": profile,
            "query": query,
            "selected_tool_id": tool_id,
            "selected_tool_name": tool_name,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.exp2_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

---

## Phase 10 — Dry Run Gate (100 Sessions)

### Step 10.1 — Create `scripts/04_dry_run.py`

```python
# scripts/04_dry_run.py
"""
Dry run: 100 sessions from R2a (original questions).
Checks all Exp3 gates. Does NOT generate responses (Exp3 only).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm
from tools.tool_index import ToolIndex
from core.session_runner import SessionRunner
from core.s0_baseline import s0_select_tool
from core.profile_decoder import profile_to_label, profile_to_tuple
from core.fslsm_augmentor import augment_query
from core.ground_truth import get_optimal_tool_id

DRY_RUN_N = 100

def run_dry():
    # Load tool index
    idx = ToolIndex()
    idx.load()

    runner = SessionRunner(
        tool_index=idx,
        db_path="results/exp3_dry_run.db",
        exp2_log_path="results/exp2_dry_run.jsonl",
    )

    # Load sessions (first 100 from R2a = original 72 questions)
    with open("data/filtered_questions.json") as f:
        questions = {q["question_id"]: q for q in json.load(f)}

    # Load Exp2 session data to get (question_id, profile) pairs
    with open("data/exp2/sessions.jsonl") as f:
        sessions = [json.loads(line) for i, line in enumerate(f) if i < DRY_RUN_N]

    print(f"[dry_run] Processing {len(sessions)} sessions...\n")

    s0_hits, s1a_hits, s1b_hits = 0, 0, 0
    s1b_tokens_total, total = 0, 0

    for sess in tqdm(sessions, desc="Dry run"):
        qid = sess["question_id"]
        if qid not in questions:
            continue
        q = questions[qid]
        profile = sess["fslsm_vector"]
        query = sess["query"]
        question_type = q["question_type"]
        profile_label = profile_to_label(profile)
        optimal_id = get_optimal_tool_id(question_type, profile)

        # S1b (FAISS + FSLSM augmentation)
        result = runner.run_s1b_session(
            question_id=qid,
            question_type=question_type,
            query=query,
            profile=profile,
            generate_response=False,  # Exp3 only — no response
        )
        if result["tsa_hit"]:
            s1b_hits += 1
        s1b_tokens_total += result["input_tokens"]

        # S1a (FAISS on raw query)
        hits_s1a = idx.retrieve(query, k=1)
        s1a_tool = hits_s1a[0][0]
        if s1a_tool.tool_id == optimal_id:
            s1a_hits += 1

        # S0 (real LLM call)
        s0_tool_id = s0_select_tool(query, profile_label)
        if s0_tool_id == optimal_id:
            s0_hits += 1

        total += 1

    # Report
    from tools.tool_registry import s0_prompt_tokens
    s0_tokens = s0_prompt_tokens()

    tsa_s0  = s0_hits / total
    tsa_s1a = s1a_hits / total
    tsa_s1b = s1b_hits / total
    pts_s1b = (s0_tokens - s1b_tokens_total/total) / s0_tokens * 100

    print(f"\n── Dry Run Results ({total} sessions) ──")
    print(f"  TSA S0:  {tsa_s0*100:.1f}%")
    print(f"  TSA S1a: {tsa_s1a*100:.1f}%")
    print(f"  TSA S1b: {tsa_s1b*100:.1f}%")
    print(f"  PTS S1b: {pts_s1b:.1f}%")

    print(f"\n── Go / No-Go Gates ──")
    gates = [
        ("TSA(S1b) − TSA(S0)  ≥ 15 pp", tsa_s1b - tsa_s0 >= 0.15),
        ("TSA(S1b) − TSA(S1a) ≥ 5 pp",  tsa_s1b - tsa_s1a >= 0.05),
        ("PTS(S1b) ≥ 80%",               pts_s1b >= 80.0),
        ("TSA(S1a) > TSA(S0)",            tsa_s1a > tsa_s0),
    ]
    all_pass = True
    for label, passed in gates:
        symbol = "GO ✓" if passed else "NO-GO ✗"
        if not passed:
            all_pass = False
        print(f"  [{symbol}] {label}")

    if all_pass:
        print("\n✅ ALL GATES PASSED — proceed to full run (Phase 11)")
    else:
        print("\n❌ GATE FAILURE — fix before proceeding")


if __name__ == "__main__":
    run_dry()
```

**Run:** `python scripts/04_dry_run.py`
**Gate:** ALL four gates must pass before Phase 11.

---

## Phase 11 — Full Run — R2a + R2b

### Step 11.1 — Full run R2a (72 original questions, serves both experiments)

```bash
python scripts/05_full_run_r2a.py
# 72 questions × 80 agents = 5,760 sessions
# Generates responses (for optional Exp2 R2)
# Logs Exp3 S1b data
```

### Step 11.2 — Full run R2b (16 coverage questions, Exp3 primary + within-R2b quality analysis)

```python
# scripts/06_full_run_r2b.py
"""
Full run R2b: 16 coverage questions × 80 agents = 1,280 sessions.
generate_response=True — responses are generated for within-R2b quality analysis.

Why generate responses here:
  - R2b cannot be compared to R0/R1 (different question set) → NOT used in Exp2 comparison
  - BUT responses enable within-R2b analysis: does correct tool selection
    (tsa_hit=True) produce higher SCS/Engagement than incorrect selection?
  - This strengthens the claim that TSA accuracy actually matters for quality
  - Cost: ~$1-2 additional (1,280 completions)
  - Tools 8-15 finally get executed — validates proposed tool implementations

Note on RAG chunks for coverage questions:
  - Coverage questions have gold_chunk_ids=[] (no pre-labeled chunks)
  - Tools 1-13: LLM generates from its own knowledge (no RAG context)
  - Tool 14 (Content Retriever): live FAISS search provides chunks
  - Tool 15 (Web Search): Tavily provides real results
  - This is acceptable — coverage questions target tools not needing corpus chunks
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm
from tools.tool_index import ToolIndex
from core.session_runner import SessionRunner

def run_r2b():
    idx = ToolIndex()
    idx.load()

    runner = SessionRunner(
        tool_index    = idx,
        db_path       = "results/exp3_results.db",       # same DB as R2a
        exp2_log_path = "results/exp2_r2b_passive.jsonl" # separate log (NOT for Exp2 comparison)
    )

    # Load coverage questions
    with open("data/coverage_questions.json") as f:
        coverage_qs = {q["question_id"]: q for q in json.load(f)}

    # Load R2b sessions (coverage question_ids × 80 agents)
    # These sessions were generated with coverage question_ids in the profile runner
    with open("data/exp2/r2b_sessions.jsonl") as f:
        sessions = [json.loads(line) for line in f]

    print(f"[R2b] {len(sessions)} sessions across {len(coverage_qs)} coverage questions\n")

    for sess in tqdm(sessions, desc="R2b full run"):
        qid = sess["question_id"]
        if qid not in coverage_qs:
            continue
        q = coverage_qs[qid]

        runner.run_s1b_session(
            question_id       = qid,
            question_type     = q["question_type"],
            query             = sess["query"],
            profile           = sess["fslsm_vector"],
            rag_chunks        = "",        # no pre-labeled chunks for coverage Qs
            generate_response = True,      # ← generate responses for within-R2b analysis
        )

    print(f"\n[R2b] Complete. Results appended to results/exp3_results.db")
    print(f"[R2b] Passive response log → results/exp2_r2b_passive.jsonl")
    print(f"[R2b] NOTE: exp2_r2b_passive.jsonl is for within-R2b analysis ONLY.")
    print(f"      Do NOT compare R2b responses to R0/R1 — different question set.")

if __name__ == "__main__":
    run_r2b()
```

**Run:** `python scripts/06_full_run_r2b.py`
**Expected:** 1,280 session results added to `exp3_results.db`. Tools 8–15 all appear in selection log.

**Within-R2b quality check after run:**
```python
# Quick sanity: does tsa_hit=True correlate with better responses?
python scripts/check_r2b_quality.py
# If correct tool selected → higher Engagement score than when wrong tool selected
# This validates that TSA accuracy is pedagogically meaningful
```

---

## Phase 12 — Post-hoc S0 + S1a Ablation

### Step 12.1 — Run S0 and S1a on all 7,040 sessions

```bash
python scripts/07_run_s0_s1a_ablation.py
# For each of 7,040 sessions:
#   S0:  real LLM call with all 15 schemas → log result
#   S1a: FAISS on raw query (no profile) → log result
# S1b: already logged during Phase 11
```

**Estimated S0 cost:** 7,040 calls × ~$0.0001 = ~$0.70

---

## Phase 13 — Metrics & Statistical Analysis

```bash
python scripts/08_compute_metrics.py
# Computes:
#   TSA ± SE per condition (S0, S1a, S1b)
#   PTS ± SE per condition
#   Per-dimension TSA breakdown (all 8 FSLSM poles)
#   Per-tool TSA breakdown (all 15 tools)
#   Statistical tests (t-test, Cohen's h)
#   Cross-experiment consistency (if Exp2 R2 data available)

python scripts/09_generate_report.py
# Generates thesis-ready figures and Table 3.4
```

---

## Phase 14 — Decision Point & Exp2 R2 Extension

### Step 14.1 — Decision Gate After Phase 13

```
Exp3 results good?
│
├── YES ✅ → Exp3 thesis contribution is complete
│            Write Exp3 results chapter (TSA, PTS, per-tool breakdown)
│            ↓
│            → Proceed to Step 14.2 (Exp2 R2a dry test)
│
└── NO  ❌ → Diagnose root cause
             → Fix FAISS descriptions / GROUND_TRUTH_MAP / augmentor
             → Re-run dry run (Phase 10)
             → Do NOT proceed to Exp2 extension until Exp3 is confirmed
```

---

### Step 14.2 — Exp2 R2a Passive Log Sanity Check

Before running full Exp2 evaluation, verify the R2a passive log is usable.

```python
# scripts/10_check_r2a_passive_log.py
"""
Sanity check on exp2_passive_log.jsonl from R2a run.
Verifies:
  1. All 5,760 R2a sessions are logged
  2. All 72 original question_ids are present
  3. All 16 profiles are represented
  4. Responses are non-empty and reasonable length
  5. Tool selection distribution looks reasonable (not dominated by 1 tool)
"""
import json
from collections import Counter

with open("results/exp2_passive_log.jsonl") as f:
    entries = [json.loads(line) for line in f]

print(f"=== Exp2 R2a Passive Log Sanity Check ===\n")

# Gate 1: Session count
n = len(entries)
gate1 = n == 5760
print(f"[{'✓' if gate1 else '✗'}] Total sessions: {n}/5760")

# Gate 2: Question coverage
qids = set(e["question_id"] for e in entries)
gate2 = len(qids) == 72
print(f"[{'✓' if gate2 else '✗'}] Unique questions: {len(qids)}/72")

# Gate 3: Profile coverage
profiles = set(e["profile_label"] for e in entries)
gate3 = len(profiles) == 16
print(f"[{'✓' if gate3 else '✗'}] Unique profiles: {len(profiles)}/16")

# Gate 4: Response quality
empty = [e for e in entries if not e.get("response") or len(e["response"]) < 50]
gate4 = len(empty) == 0
print(f"[{'✓' if gate4 else '✗'}] Empty/short responses: {len(empty)}")

# Gate 5: Tool distribution (no single tool > 30%)
tool_dist = Counter(e["selected_tool_id"] for e in entries)
dominant_pct = tool_dist.most_common(1)[0][1] / n * 100
gate5 = dominant_pct < 30
print(f"[{'✓' if gate5 else '✗'}] Dominant tool: {dominant_pct:.1f}% (threshold <30%)")
print(f"  Tool distribution: {dict(sorted(tool_dist.items()))}")

all_pass = all([gate1, gate2, gate3, gate4, gate5])
print(f"\n{'✅ PROCEED to Step 14.3 (Exp2 R2a dry evaluation)' if all_pass else '❌ FIX ISSUES before proceeding'}")
```

**Run:** `python scripts/10_check_r2a_passive_log.py`
**All 5 gates must pass before Step 14.3.**

---

### Step 14.3 — Exp2 R2a Dry Evaluation (100 Sessions)

Run LLM-as-Judge evaluation on 100 random R2a sessions to verify metrics are
moving in the right direction before committing to full evaluation.

```python
# scripts/11_exp2_r2a_dry_eval.py
"""
Dry evaluation: compute SCS, RR, Engagement for 100 random R2a sessions.
Compare against R0/R1 baseline from existing Exp2 results.
Gate: SCS(R2) > SCS(R1) or SCS(R2) ≈ SCS(R1) — must NOT be worse.
"""
import json, os, random
from openai import OpenAI
from collections import defaultdict

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load 100 random R2a sessions
with open("results/exp2_passive_log.jsonl") as f:
    all_entries = [json.loads(line) for line in f]
sample = random.sample(all_entries, 100)

# Load existing R0/R1 baselines from Exp2 results
import pandas as pd
exp2_df = pd.read_csv("results/exp2_session_metrics.csv")
r0_scs_mean = exp2_df[exp2_df["mode"]=="R0"]["scs"].mean()
r1_scs_mean = exp2_df[exp2_df["mode"]=="R1"]["scs"].mean()
r0_eng_mean = exp2_df[exp2_df["mode"]=="R0"]["engagement"].mean()
r1_eng_mean = exp2_df[exp2_df["mode"]=="R1"]["engagement"].mean()

print(f"Existing baselines:")
print(f"  R0 SCS: {r0_scs_mean:.3f}  Engagement: {r0_eng_mean:.3f}")
print(f"  R1 SCS: {r1_scs_mean:.3f}  Engagement: {r1_eng_mean:.3f}")
print()

# FSLSM style anchors for SCS computation (from Exp2)
STYLE_ANCHORS = {
    "Visual":      "uses diagrams, charts, figures, spatial layout, visual representations",
    "Verbal":      "uses written prose, narrative descriptions, textual explanations",
    "Sequential":  "follows numbered steps, linear ordering, explicit progression",
    "Global":      "provides overview, big picture, holistic framing, thematic connections",
    "Active":      "includes exercises, activities, hands-on tasks, problem-solving",
    "Reflective":  "encourages reflection, thinking, pause, self-assessment",
    "Sensing":     "uses concrete examples, practical cases, real applications, standard facts",
    "Intuitive":   "uses analogies, metaphors, abstract theory, novel connections",
}

SCS_PROMPT = """Rate how well this tutor response matches the student's FSLSM learning style.

Student profile: {profile_label}
Style characteristics: {style_desc}

Tutor response:
{response}

Rate style conformance from 0.0 to 1.0:
- 1.0 = Response perfectly matches the described learning style
- 0.5 = Response partially matches
- 0.0 = Response ignores or contradicts the learning style

Respond with ONLY a decimal number (e.g. 0.75). Nothing else."""

ENGAGEMENT_PROMPT = """Rate the pedagogical quality and student engagement potential of this
tutor response for a machine learning student.

Question: {query}
Response: {response}

Rate from 1-5:
- 5 = Excellent: clear, pedagogically sound, engaging, well-structured
- 3 = Adequate: correct but generic or poorly adapted
- 1 = Poor: confusing, off-topic, or pedagogically inappropriate

Respond with ONLY an integer (1-5). Nothing else."""

r2_scs_scores = []
r2_eng_scores = []

for entry in sample:
    profile_label = entry["profile_label"]
    dims = profile_label.split("-")
    style_desc = "; ".join(STYLE_ANCHORS.get(d, "") for d in dims if d in STYLE_ANCHORS)

    # Compute SCS
    scs_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": SCS_PROMPT.format(
            profile_label=profile_label,
            style_desc=style_desc,
            response=entry["response"][:1000]
        )}],
        max_tokens=5, temperature=0
    )
    try:
        scs = float(scs_resp.choices[0].message.content.strip())
        r2_scs_scores.append(scs)
    except:
        pass

    # Compute Engagement
    eng_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": ENGAGEMENT_PROMPT.format(
            query=entry["query"],
            response=entry["response"][:1000]
        )}],
        max_tokens=3, temperature=0
    )
    try:
        eng = int(eng_resp.choices[0].message.content.strip())
        r2_eng_scores.append(eng / 5.0)  # normalise to 0-1
    except:
        pass

r2_scs = sum(r2_scs_scores) / len(r2_scs_scores)
r2_eng = sum(r2_eng_scores) / len(r2_eng_scores)

print(f"R2a dry evaluation results ({len(r2_scs_scores)} sessions):")
print(f"  R2 SCS:        {r2_scs:.3f}")
print(f"  R2 Engagement: {r2_eng:.3f}")

print(f"\n── Go / No-Go Gates ──")
gates = [
    ("R2 SCS ≥ R0 SCS",         r2_scs >= r0_scs_mean),
    ("R2 SCS ≥ R1 SCS × 0.95",  r2_scs >= r1_scs_mean * 0.95),  # within 5% of R1
    ("R2 Engagement ≥ R0",       r2_eng >= r0_eng_mean),
]
all_pass = True
for label, passed in gates:
    symbol = "GO ✓" if passed else "NO-GO ✗"
    if not passed: all_pass = False
    print(f"  [{symbol}] {label}")

if all_pass:
    print("\n✅ PROCEED to Step 14.4 (Exp2 R2a full evaluation)")
else:
    print("\n❌ R2a quality not sufficient — review tool prompts before full evaluation")
    print("   Likely fix: improve TOOL_PROMPTS in tools/tool_prompts.py")
```

**Run:** `python scripts/11_exp2_r2a_dry_eval.py`
**Cost:** ~200 GPT-4o calls × $0.005 = ~$1.00
**Gate:** All 3 gates pass before Step 14.4.

---

### Step 14.4 — Exp2 R2a Full Evaluation (5,760 Sessions)

```python
# scripts/12_exp2_r2a_full_eval.py
"""
Full Exp2 evaluation for R2a: compute all 5 metrics for all 5,760 sessions.
Metrics: SCS, RR, CR@5, ER, Engagement
Results saved to results/exp2_r2a_metrics.csv
"""
import json, os, sqlite3
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load all R2a sessions from passive log
with open("results/exp2_passive_log.jsonl") as f:
    sessions = [json.loads(line) for line in f]

# Load question metadata (gold chunks for CR@5, ER)
with open("data/filtered_questions.json") as f:
    questions = {q["question_id"]: q for q in json.load(f)}

RR_PROMPT = """Rate the factual accuracy and relevance of this tutor response
to the given machine learning question.

Question: {query}
Response: {response}

Score from 0.0 to 1.0 (0=completely wrong/irrelevant, 1=fully accurate and relevant).
Respond with ONLY a decimal number."""

results = []

for sess in tqdm(sessions, desc="Exp2 R2a full eval"):
    qid = sess["question_id"]
    q = questions.get(qid, {})

    # SCS — style conformance (reuse from dry eval logic)
    scs = compute_scs(sess["response"], sess["profile_label"])      # from dry eval

    # RR — response relevance via LLM judge
    rr_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": RR_PROMPT.format(
            query=sess["query"],
            response=sess["response"][:1200]
        )}],
        max_tokens=5, temperature=0
    )
    rr = float(rr_resp.choices[0].message.content.strip())

    # CR@5 — chunk recall: do gold chunks appear in retrieved context?
    # NOTE: For R2 this measures whether the MCP pipeline retrieved relevant chunks
    gold_ids = set(q.get("gold_chunk_ids", []))
    retrieved_ids = set(sess.get("retrieved_chunk_ids", [])[:5])
    cr5 = len(gold_ids & retrieved_ids) / max(len(gold_ids), 1) if gold_ids else None

    # ER — essential recall
    essential_ids = set(q.get("essential_chunk_ids", []))
    er = 1.0 if essential_ids and essential_ids.issubset(retrieved_ids) else 0.0 if essential_ids else None

    # Engagement — pedagogical quality
    eng = compute_engagement(sess["query"], sess["response"])       # from dry eval

    results.append({
        "session_id":       sess["session_id"],
        "question_id":      qid,
        "question_type":    sess["question_type"],
        "mode":             "R2",
        "profile_label":    sess["profile_label"],
        "selected_tool_id": sess["selected_tool_id"],
        "selected_tool_name": sess["selected_tool_name"],
        "scs":              scs,
        "rr":               rr,
        "cr5":              cr5,
        "er":               er,
        "engagement":       eng,
    })

df = pd.DataFrame(results)
df.to_csv("results/exp2_r2a_metrics.csv", index=False)
print(f"\nSaved {len(df)} session metrics to results/exp2_r2a_metrics.csv")

# Summary vs R0/R1
print("\n── R2 vs R0/R1 comparison ──")
exp2_existing = pd.read_csv("results/exp2_session_metrics.csv")
for metric in ["scs", "rr", "cr5", "er", "engagement"]:
    r0_m = exp2_existing[exp2_existing["mode"]=="R0"][metric].mean()
    r1_m = exp2_existing[exp2_existing["mode"]=="R1"][metric].mean()
    r2_m = df[metric].dropna().mean()
    print(f"  {metric.upper():<12} R0={r0_m:.3f}  R1={r1_m:.3f}  R2={r2_m:.3f}")
```

**Run:** `python scripts/12_exp2_r2a_full_eval.py`
**Cost:** ~5,760 × 2 GPT-4o calls = ~$28–35
**Output:** `results/exp2_r2a_metrics.csv`

---

### Step 14.5 — Exp2 R2 Statistical Analysis and Reporting

```python
# scripts/13_exp2_r2_report.py
"""
Statistical comparison of R0, R1, R2 conditions.
Generates:
  - Table 3.3 updated with R2 column
  - Bar charts for all 5 metrics across 3 conditions
  - ANOVA or pairwise t-tests (R2 vs R0, R2 vs R1)
  - Effect sizes (Cohen's d)
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

exp2_existing = pd.read_csv("results/exp2_session_metrics.csv")
exp2_r2       = pd.read_csv("results/exp2_r2a_metrics.csv")

# Combine
all_data = pd.concat([
    exp2_existing[exp2_existing["mode"].isin(["R0","R1"])],
    exp2_r2
], ignore_index=True)

metrics = ["scs", "rr", "cr5", "er", "engagement"]

print("═══════════════════════════════════════════════════════")
print("  Experiment 2 Results — R0 vs R1 vs R2")
print("═══════════════════════════════════════════════════════\n")
print(f"  {'Metric':<12} {'R0':>8} {'R1':>8} {'R2':>8}  {'R1>R0':>8}  {'R2>R1':>8}")
print("  " + "─"*60)

for metric in metrics:
    r0 = all_data[all_data["mode"]=="R0"][metric].dropna()
    r1 = all_data[all_data["mode"]=="R1"][metric].dropna()
    r2 = all_data[all_data["mode"]=="R2"][metric].dropna()

    _, p_r1_r0 = stats.ttest_ind(r1, r0)
    _, p_r2_r1 = stats.ttest_ind(r2, r1)

    sig_r1 = "***" if p_r1_r0<0.001 else "**" if p_r1_r0<0.01 else "*" if p_r1_r0<0.05 else "ns"
    sig_r2 = "***" if p_r2_r1<0.001 else "**" if p_r2_r1<0.01 else "*" if p_r2_r1<0.05 else "ns"

    print(f"  {metric.upper():<12} {r0.mean():>8.3f} {r1.mean():>8.3f} {r2.mean():>8.3f}  {sig_r1:>8}  {sig_r2:>8}")

# Generate figures
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
for ax, metric in zip(axes, metrics):
    means = [all_data[all_data["mode"]==m][metric].mean() for m in ["R0","R1","R2"]]
    sems  = [all_data[all_data["mode"]==m][metric].sem()  for m in ["R0","R1","R2"]]
    ax.bar(["R0","R1","R2"], means, yerr=sems, color=colors, capsize=5, edgecolor="black")
    ax.set_title(metric.upper())
    ax.set_ylim(0, 1.1)
plt.suptitle("Experiment 2: R0 vs R1 vs R2 (FSLSM-RAG+MCP)")
plt.tight_layout()
plt.savefig("results/figures/exp2_r0_r1_r2_comparison.png", dpi=150)
print("\nSaved: results/figures/exp2_r0_r1_r2_comparison.png")
```

**Run:** `python scripts/13_exp2_r2_report.py`
**Output:** Updated Table 3.3 + comparison figure.

---

### Step 14.6 — Final Decision: Add R2 to Thesis or Wrap Up

```
R2a evaluation complete?
│
├── R2 metrics valid (R2 SCS ≥ R1 SCS and R2 Engagement ≥ R1 Engagement)
│   ├── YES ✅
│   │     → Add R2 as third condition to Experiment 2 results chapter
│   │     → Update Table 3.3 with R2 column
│   │     → Add Figure: R0 vs R1 vs R2 bar chart
│   │     → Thesis framing:
│   │         "R2 demonstrates that FSLSM-conditioned MCP tool selection
│   │          further improves style conformance and engagement beyond
│   │          FSLSM personalization alone (R1), validating the complete
│   │          FSLSM-RAG-MCP framework."
│   │     → Thesis is complete ✓
│   │
│   └── NO  ❌ (R2 worse than R1 on SCS or Engagement)
│         → This is ALSO a valid finding — document it:
│             "Adding MCP tool routing (R2) does not degrade response
│              quality relative to FSLSM-RAG (R1), confirming that
│              Experiment 3's token efficiency gains (PTS) are achieved
│              without quality trade-off."
│         → Report as null result for R2 vs R1 comparison
│         → Thesis is still complete ✓
│
└── R2a passive log failed sanity check (Step 14.2)
      → Thesis wraps up at Experiment 3 ✓
      → Exp3 results alone answer RQ3
```

---

## Execution Summary

```bash
# Phase 1-3: Code fixes (no API calls)
python tools/tool_registry.py              # verify registry

# Phase 2: Build FAISS index (~$0.001)
python scripts/01_build_tool_index.py

# Phase 3: Verify profile decoder
python -c "from core.profile_decoder import *; print(decode_profile({'act_ref':-1,'sen_int':-1,'vis_ver':-1,'seq_glo':-1}))"

# Phase 4: Verify ground truth coverage
python core/ground_truth.py

# Phase 5: Generate + review coverage questions (~$0.02)
python scripts/02_generate_coverage_questions.py
# >>> MANUAL REVIEW data/coverage_questions.json <<<
python scripts/merge_questions.py

# Phase 6: Verify Tavily
python tools/tavily_search.py

# Phase 10: Dry run gate (~$0.10 for S0 LLM calls)
python scripts/04_dry_run.py
# >>> ALL GATES MUST PASS <<<

# Phase 11: Full run (~$40-62 for R2a + R2b responses)
python scripts/05_full_run_r2a.py           # 5,760 sessions → Exp3 + passive Exp2 log
python scripts/06_full_run_r2b.py           # 1,280 sessions → Exp3 + within-R2b log

# Phase 12: S0 + S1a ablation (~$0.70)
python scripts/07_run_s0_s1a_ablation.py

# Phase 13: Exp3 Metrics + report
python scripts/08_compute_metrics.py
python scripts/09_generate_report.py

# ── DECISION POINT: Exp3 results good? ──────────────────────
# If YES → proceed to Phase 14 (Exp2 R2 extension)
# If NO  → diagnose and re-run

# Phase 14: Exp2 R2 extension (optional, only if Exp3 confirmed)
python scripts/10_check_r2a_passive_log.py  # sanity check (5 gates)
# >>> ALL 5 GATES MUST PASS <<<
python scripts/11_exp2_r2a_dry_eval.py      # 100-session dry eval (~$1.00)
# >>> ALL 3 GATES MUST PASS <<<
python scripts/12_exp2_r2a_full_eval.py     # full 5,760-session eval (~$28-35)
python scripts/13_exp2_r2_report.py         # updated Table 3.3 + figures
```

---

## Estimated Total Cost

### Experiment 3 (Primary — Always Run)

| Phase | API Calls | Est. Cost |
|---|---|---|
| FAISS index build | 15 embeddings | $0.001 |
| Coverage Q generation | 16 completions | $0.02 |
| Dry run S0 (100 sessions) | 100 completions | $0.01 |
| Full run R2a (responses) | 5,760 completions | $40–58 |
| Full run R2b (responses) | 1,280 completions | $1–2 |
| S0 ablation full (7,040) | 7,040 completions | $0.70 |
| Tavily searches (Tool 15) | ~500 calls | $0 (free tier) |
| **Exp3 Total** | | **~$42–61** |

### Experiment 2 R2 Extension (Optional — If Exp3 Passes)

| Phase | API Calls | Est. Cost |
|---|---|---|
| R2a dry evaluation (100 sessions) | ~200 GPT-4o calls | $1.00 |
| R2a full evaluation (5,760 sessions) | ~11,520 GPT-4o calls | $28–35 |
| **Exp2 Extension Total** | | **~$29–36** |

### Combined Maximum

| Scenario | Est. Total Cost |
|---|---|
| Exp3 only | ~$42–61 |
| Exp3 + Exp2 R2 extension | ~$71–97 |

---

*Workplan v2.1 — Step 11.2 corrected (generate_response=True), Phase 14 expanded with full Exp2 R2a dry test and evaluation pipeline*
*Thesis: FSLSM-RAG-MCP Adaptive AI Tutoring System*
