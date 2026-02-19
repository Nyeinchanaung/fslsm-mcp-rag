# D2L-QA Multi-Hop Ground Truth Dataset Generator (5-Chunk)

Generate multi-hop question-answer pairs from the *Dive into Deep Learning* (D2L) textbook, where each answer requires synthesizing information across **5 related chunks**. Designed for RAG pipeline evaluation and chatbot benchmarking alongside the [single-hop dataset](../single_hop/README.md).

## Features

✅ **5-Chunk Synthesis**: Each question requires information from multiple passages across different chapters  
✅ **3 Grouping Strategies**: Intra-cluster (same topic), prerequisite chain, and cross-topic synthesis  
✅ **Ablation Validation**: Automated leave-one-out testing ensures questions truly need all 5 chunks  
✅ **Quality Tiering**: Gold / Silver / Bronze tiers based on multi-hop necessity scores  
✅ **Smart API Routing**: Auto-detects OpenAI vs Anthropic based on model name — no manual switching  
✅ **Compatible with Single-Hop**: Shares the same corpus chunks for unified RAG evaluation  

## How It Differs from Single-Hop

| Aspect | Single-Hop | Multi-Hop (5-Chunk) |
|--------|-----------|---------------------|
| Chunks per question | 1 | 5 |
| Answer source | Single passage | Synthesized across 5 passages |
| Question types | Factual, conceptual, etc. | Compare, trace evolution, synthesize workflow, etc. |
| Retrieval metric | Recall@K (binary hit) | Chunk Recall, Chunk Precision, Chunk F1 |
| Difficulty | Easy–Hard | Medium–Hard |
| Purpose | Baseline retrieval accuracy | Tests cross-section reasoning & synthesis |

## Installation

Same as the single-hop dataset — see [single-hop README](../single_hop/README.md#installation).

## Quick Start

### Test Run (Recommended First)
**Cost**: ~$2–4 | **Time**: ~10–15 min | **Questions**: 30

```bash
python d2l/multi_hop/d2l_multihop_generator.py ./d2l-en \
  --output-dir ./d2l/multi_hop/output \
  --model gpt-4o-mini \
  --questions-per-group 1 \
  --num-groups 30 \
  --skip-validation \
  --corpus-path ./d2l/single_hop/output/d2l_corpus_chunks.json
```

> **Note**: The folder is named `multi_hop` (underscore) so Python module imports work correctly.

### Full Production Run
**Cost**: ~$40–60 | **Time**: ~4–6 hours | **Questions**: 150

```bash
python d2l/multi_hop/d2l_multihop_generator.py ./d2l-en \
  --output-dir ./d2l/multi_hop/output \
  --model gpt-4o \
  --questions-per-group 2 \
  --num-groups 100 \
  --corpus-path ./d2l/single_hop/output/d2l_corpus_chunks.json
```

### Using as a Python Module

```python
from d2l.multi_hop.d2l_multihop_generator import main

main(
    d2l_root="./d2l-en",
    output_dir="./d2l/multi_hop/output",
    model="gpt-4o-mini",       # or "claude-sonnet-4-20250514"
    n_questions_per_group=2,
    num_groups=100,
    skip_validation=True
)
```

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                D2L-QA Multi-Hop Pipeline (5-Chunk)                │
│                                                                   │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────────┐ │
│  │ D2L Book │──▶│ Chunk & Parse│──▶│ Cluster Chunks by Topic   │ │
│  │ (source) │   │  (shared)    │   │    (Step 1)               │ │
│  └──────────┘   └──────────────┘   └─────────────┬─────────────┘ │
│                                                   │               │
│                                    ┌──────────────▼────────────┐  │
│                                    │ Find 5-Chunk Groups       │  │
│                                    │ • Intra-cluster           │  │
│                                    │ • Prerequisite chain      │  │
│                                    │ • Cross-topic synthesis   │  │
│                                    │    (Step 2)               │  │
│                                    └──────────────┬────────────┘  │
│                                                   │               │
│                                    ┌──────────────▼────────────┐  │
│                                    │ Generate QA from 5 Chunks │  │
│                                    │    (Step 3)               │  │
│                                    └──────────────┬────────────┘  │
│                                                   │               │
│                                    ┌──────────────▼────────────┐  │
│                                    │ Ablation Validation       │  │
│                                    │ • Full 5 → should pass    │  │
│                                    │ • Leave-one-out (4 each)  │  │
│                                    │ • Single chunk → must fail│  │
│                                    │    (Step 4)               │  │
│                                    └──────────────┬────────────┘  │
│                                                   │               │
│                                    ┌──────────────▼────────────┐  │
│                                    │ Tier & Format Dataset     │  │
│                                    │    (Step 5)               │  │
│                                    └──────────────────────────-┘  │
└───────────────────────────────────────────────────────────────────┘
```

## Chunk Grouping Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| **Intra-cluster** | Same topic across different chapters | 5 chunks about "optimization" from Preliminaries, Linear NN, CNN, RNN, and Attention chapters |
| **Prerequisite chain** | Ordered concept dependency | Gradient descent → Loss functions → SGD → Momentum → Adam |
| **Cross-topic synthesis** | 2+2+1 from related topic clusters | 2 CNN chunks + 2 Attention chunks + 1 Vision Transformer chunk |

## Question Types

| Type | Description | Example |
|------|-------------|---------|
| **Compare** | Similarities/differences across concepts | "Compare how dropout is applied in CNNs vs. RNNs and explain why the strategies differ" |
| **Trace Evolution** | Follow concept development across topics | "Trace how the vanishing gradient problem led from vanilla RNNs to LSTMs to Transformers" |
| **Explain Relationship** | Connect related concepts | "Explain the relationship between weight decay, L2 regularization, and batch normalization" |
| **Synthesize Workflow** | Combine steps from multiple areas | "Describe a complete training pipeline for image classification, covering data augmentation, architecture choice, optimizer selection, learning rate scheduling, and evaluation" |
| **Debug Scenario** | Diagnose problem using cross-topic knowledge | "A model's training loss decreases but validation loss increases after epoch 5 — using your knowledge of overfitting, regularization, and learning rate schedules, diagnose and propose fixes" |

## Output

### 1. `d2l_multihop_ground_truth.json`
Complete multi-hop dataset:

```json
{
  "metadata": {
    "name": "D2L-QA-MultiHop",
    "version": "1.0",
    "total_questions": 150,
    "chunks_per_question": 5,
    "total_corpus_chunks": 2467,
    "grouping_strategies": {"intra_cluster": 55, "prerequisite_chain": 50, "cross_topic": 45},
    "question_types": {"compare": 38, "trace_evolution": 32, "explain_relationship": 30, "synthesize_workflow": 28, "debug_scenario": 22},
    "quality_tiers": {"gold": 80, "silver": 70}
  },
  "corpus": "shared with single-hop (see ../single_hop/output/d2l_corpus_chunks.json)",
  "questions": [
    {
      "question_id": "d2l_mh_0012",
      "question": "Trace how the vanishing gradient problem in vanilla RNNs motivated the LSTM design, and explain how attention mechanisms later addressed LSTM's remaining limitations for long sequences.",
      "gold_answer": "Vanilla RNNs suffer from vanishing gradients during BPTT because gradients shrink exponentially through time steps. LSTMs addressed this with a cell state and forget gate creating an additive gradient path. However, LSTMs still process sequences sequentially. Attention mechanisms solved this by allowing direct connections between any positions, which the Transformer architecture then used to remove recurrence entirely.",
      "answer_type": "trace_evolution",
      "difficulty": "hard",
      "strategy": "prerequisite_chain",
      "quality_tier": "gold",
      "gold_chunk_ids": [
        "rnn_basics_c3",
        "bptt_gradient_c1",
        "lstm_design_c2",
        "attention_mechanism_c0",
        "transformer_arch_c1"
      ],
      "chunk_contributions": {
        "rnn_basics_c3": "Defines vanilla RNN forward pass and hidden state",
        "bptt_gradient_c1": "Explains vanishing gradient math in BPTT",
        "lstm_design_c2": "LSTM gates and additive cell state path",
        "attention_mechanism_c0": "Attention as direct position connections",
        "transformer_arch_c1": "Self-attention replacing recurrence entirely"
      },
      "essential_chunk_ids": ["bptt_gradient_c1", "lstm_design_c2", "attention_mechanism_c0"],
      "supporting_chunk_ids": ["rnn_basics_c3", "transformer_arch_c1"],
      "validation": {
        "full_5_score": 9,
        "avg_leave_one_out": 4.8,
        "max_single_chunk": 3
      }
    }
  ]
}
```

### 2. Shared Corpus
Multi-hop uses the same `d2l_corpus_chunks.json` as single-hop — no duplication needed. The file is at `d2l/single_hop/output/d2l_corpus_chunks.json`.

## Quality Tiering (Ablation Validation)

Each question is validated by testing if it can be answered with fewer chunks:

| Test | What It Checks |
|------|---------------|
| **Full 5 chunks** | Answer should score ≥7 (correct with all context) |
| **Leave-one-out (4 chunks)** | Score should drop — avg must be <6 |
| **Single chunk only** | Must fail — max score must be <4 |

Questions are tiered based on results:

| Tier | Criteria | Meaning |
|------|----------|---------|
| **Gold** | Full ≥8, Leave-one-out avg <5, Single max <3 | Truly requires all 5 chunks |
| **Silver** | Full ≥7, Leave-one-out avg <6, Single max <5 | Needs most chunks |
| **Reject** | Doesn't meet Silver criteria | Answerable from fewer chunks — removed |

## Evaluation

### Retrieval Metrics (5-Chunk)

```python
from d2l.multi_hop.d2l_multihop_generator import evaluate_5chunk_retrieval
import json

with open("d2l/multi_hop/output/d2l_multihop_ground_truth.json") as f:
    dataset = json.load(f)

metrics = evaluate_5chunk_retrieval(dataset, your_retriever, k=5)
print(metrics)
# {
#   'avg_chunk_recall': 0.48,      # found 2.4 out of 5 gold chunks on average
#   'avg_chunk_precision': 0.48,   # 2.4 out of 5 retrieved were relevant
#   'avg_chunk_f1': 0.48,
#   'avg_essential_recall': 0.55,  # found essential chunks more often
#   'perfect_match_rate': 0.12,    # only 12% found all 5
# }
```

### Answer Metrics

```python
from d2l.multi_hop.d2l_multihop_generator import evaluate_5chunk_answers

results = evaluate_5chunk_answers(dataset, your_rag_pipeline)
# {
#   'string_em_accuracy': 0.22,
#   'avg_judge_score': 4.9,
#   'judge_pass_rate': 0.38,
# }
```

### Combined Evaluation (Single-Hop + Multi-Hop)

```python
# Load both datasets
with open("d2l/single_hop/output/d2l_qa_ground_truth.json") as f:
    single_hop = json.load(f)
with open("d2l/multi_hop/output/d2l_multihop_ground_truth.json") as f:
    multi_hop = json.load(f)

# Compare performance across both
for name, dataset in [("Single-Hop", single_hop), ("Multi-Hop", multi_hop)]:
    retrieval = evaluate_retrieval(dataset, your_retriever)
    answers = evaluate_answers(dataset, your_rag_pipeline)
    print(f"\n{name}:")
    print(f"  Recall@5:    {retrieval['recall@5']:.2f}")
    print(f"  Judge Score: {answers['avg_judge_score']:.1f}")
```

## Benchmarking Metrics

### Retrieval Metrics
- **Chunk Recall**: Fraction of 5 gold chunks found in top-5 retrieved
- **Chunk Precision**: Fraction of top-5 retrieved that are gold chunks
- **Chunk F1**: Harmonic mean of Chunk Recall and Precision
- **Essential Recall**: Did retriever find the must-have chunks?
- **Perfect Match Rate**: Found all 5 gold chunks exactly

### Answer Metrics
- **StringEM**: Exact string match (strict)
- **LLM Judge Score**: Semantic similarity (0–9 scale)
- **Judge Pass Rate**: % scoring ≥6

### Expected Performance (Multi-Hop vs Single-Hop)

| Metric | Single-Hop | Multi-Hop (5-Chunk) |
|--------|-----------|---------------------|
| Recall@5 | ~0.83 | ~0.48 (chunk recall) |
| Perfect Match | N/A | ~0.12 |
| StringEM | ~0.43 | ~0.22 |
| Judge Score | ~6.8 | ~4.9 |
| Judge Pass Rate | ~0.71 | ~0.38 |

> The performance drop on multi-hop is expected — it demonstrates where current RAG pipelines struggle with cross-section synthesis.

### Key Finding: Chunk Retrieval Drives Answer Quality

```
Answer Quality vs Gold Chunks Retrieved:
  Found 1/5 → Avg Judge Score: 2.8
  Found 2/5 → Avg Judge Score: 3.9
  Found 3/5 → Avg Judge Score: 5.4
  Found 4/5 → Avg Judge Score: 6.7
  Found 5/5 → Avg Judge Score: 7.8
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-4o` | LLM model — auto-detects OpenAI or Anthropic |
| `--questions-per-group` | `2` | Questions to generate per 5-chunk group |
| `--num-groups` | `100` | Number of 5-chunk groups to form |
| `--skip-validation` | `False` | Skip ablation validation (faster, lower cost) |
| `--min-similarity` | `0.3` | Minimum TF-IDF similarity for chunk pairing |
| `--max-similarity` | `0.7` | Maximum TF-IDF similarity (avoids near-duplicates) |
| `--output-dir` | `.` | Directory to save output JSON files |

> **API keys** are loaded automatically from `.env`. No `--api-key` flag needed.

## Actual Output Scale (D2L English Edition)

```
Corpus (shared):        2,467 chunks
5-chunk groups formed:  ~200 candidate groups
After coherence filter: ~150 usable groups
Raw QA pairs:           ~300 (at 2 questions/group)
After ablation filter:  ~180 (60% pass multi-hop validation)
Quality tiers:
  Gold:                 ~80  (truly needs all 5 chunks)
  Silver:               ~70  (needs most chunks)
  Rejected:             ~30  (answerable from fewer chunks)
Final dataset:          ~150 questions
```

## Folder Structure

```
d2l/
├── requirements.txt
├── single_hop/
│   ├── d2l_qa_generator.py          # Single-hop generator
│   ├── README.md                    # Single-hop README
│   ├── d2l_qa_ground_truth_guide.md
│   └── output/
│       ├── d2l_qa_ground_truth.json # Single-hop dataset
│       └── d2l_corpus_chunks.json   # Shared corpus (used by both)
└── multi_hop/
    ├── d2l_multihop_generator.py    # Multi-hop generator
    ├── README.md                    # ← You are here
    └── output/
        └── d2l_multihop_ground_truth.json  # Multi-hop dataset
```

## Best Practices

1. **Generate single-hop first**: Multi-hop reuses the same corpus chunks
2. **Start small**: Use `--num-groups 30 --skip-validation` to verify setup
3. **Validation is expensive but worth it**: Ablation runs 6 LLM calls per question (full, 4× leave-one-out, 1× single) — budget accordingly
4. **Check tier distribution**: If most questions are rejected, your chunk grouping similarity range may need tuning
5. **Combine with single-hop for reporting**: The performance gap between single-hop and multi-hop is your key experimental finding
6. **Version your dataset**: Tag with D2L commit hash for reproducibility

## Troubleshooting

### Too Many Rejected Questions
Chunk groups are too dissimilar — questions become forced. Try:
```bash
--min-similarity 0.35 --max-similarity 0.65
```

### Validation Too Expensive
Skip for initial experiments, enable for final dataset:
```bash
--skip-validation    # saves ~$20-30 on full run
```

### Questions Answerable from Single Chunk
The ablation validator catches these. If you skipped validation, manually check a sample — if >30% are single-hop in disguise, re-run with validation enabled.

### JSON Parse Errors
D2L's heavy LaTeX (e.g. `\alpha`, `\frac`, `\nabla`) causes LLMs to include bare
backslashes in JSON responses. This is **handled automatically** via a 3-stage fallback:
1. Direct `json.loads` (fast path)
2. Regex-escape invalid backslashes, retry
3. `json-repair` library (catches apostrophes, trailing commas, etc.)

Parse failures are now rare — only if the LLM returns completely garbled output.

## Citation

Based on methodology from:
- PersonaRAG evaluation framework (5-passage retrieval evaluation)
- Chatbot Arena benchmarking (LLM-as-judge with self-consistency)
- Natural Questions (NQ) dataset format

## License

MIT License — See textbook license at https://d2l.ai
