# D2L-QA Ground Truth Dataset Generator

Generate a high-quality question-answer ground truth dataset from the *Dive into Deep Learning* (D2L) textbook for RAG pipeline evaluation and chatbot benchmarking.

## Features

✅ **Multi-type Question Generation**: Factual, conceptual, procedural, comparative, mathematical, and code-based questions  
✅ **Smart API Routing**: Auto-detects OpenAI vs Anthropic based on model name — no manual switching  
✅ **Quality Filtering**: Automated groundedness checks, LLM-as-judge validation, and deduplication  
✅ **Evaluation Metrics**: Retrieval evaluation (Recall@K, MRR) and answer correctness (StringEM, LLM Judge)  
✅ **Stratified Sampling**: Ensures balanced coverage across chapters and difficulty levels  

## Installation

```bash
# 1. Clone the D2L repository
git clone https://github.com/d2l-ai/d2l-en.git

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r d2l/requirements.txt
# Includes: anthropic, openai, scikit-learn, python-dotenv, json-repair
```

## API Key Setup

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# For OpenAI (gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=sk-proj-your-key-here

# For Anthropic (claude-* models)
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

The script automatically loads `.env` and picks the right API based on the model name you specify.

## Quick Start

### Test Run (Recommended First)
**Cost**: ~$1–2 | **Time**: ~5–10 min | **Questions**: 50

```bash
python d2l/single_hop/d2l_qa_generator.py ./d2l-en \
  --output-dir ./d2l/single_hop/output \
  --model gpt-4o-mini \
  --questions-per-chunk 2 \
  --sample-size 50 \
  --skip-validation
```

### Full Production Run
**Cost**: ~$30–40 | **Time**: ~3–4 hours | **Questions**: 500

```bash
python d2l/single_hop/d2l_qa_generator.py ./d2l-en \
  --output-dir ./d2l/single_hop/output \
  --model gpt-4o \
  --questions-per-chunk 3 \
  --sample-size 500
```

### Using as a Python Module

```python
from d2l.single_hop.d2l_qa_generator import main

main(
    d2l_root="./d2l-en",
    output_dir="./d2l/single_hop/output",
    model="gpt-4o-mini",       # or "claude-sonnet-4-20250514"
    n_questions_per_chunk=2,
    sample_size=500,
    skip_validation=True
)
```

## Supported Models

| Provider | Models | Auto-detected by |
|----------|--------|-----------------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `o1-*`, `o3-*` | `gpt`, `o1`, `o3` prefix |
| **Anthropic** | `claude-*`, `claude-sonnet-*`, `claude-opus-*` | `claude`, `sonnet`, `opus`, `haiku` prefix |

## Output

The generator produces two JSON files:

### 1. `d2l_qa_ground_truth.json`
Complete dataset with questions, answers, and metadata:

```json
{
  "metadata": {
    "name": "D2L-QA",
    "version": "1.0",
    "total_questions": 500,
    "total_corpus_chunks": 2467,
    "question_types": {"factual": 148, "conceptual": 126, ...},
    "difficulty_distribution": {"easy": 125, "medium": 250, "hard": 125}
  },
  "corpus": [...],
  "questions": [
    {
      "question_id": "d2l_q_0042",
      "question": "What are the two main differences between GRU and LSTM?",
      "gold_answer": "GRU uses two gates while LSTM uses three...",
      "answer_type": "comparative",
      "difficulty": "medium",
      "gold_chunk_ids": ["a3f2b1c9d4e5_2"],
      "chapter": "09_recurrent-modern",
      "section": "Gated Recurrent Units (GRU)"
    }
  ]
}
```

### 2. `d2l_corpus_chunks.json`
All corpus chunks for RAG indexing.

## Evaluation

Use the generated dataset to evaluate your RAG pipeline:

```python
from d2l.single_hop.d2l_qa_generator import evaluate_retrieval, evaluate_answers
import json

with open("d2l/single_hop/output/d2l_qa_ground_truth.json") as f:
    dataset = json.load(f)

# Evaluate retrieval
retrieval_metrics = evaluate_retrieval(dataset, your_retriever)
print(retrieval_metrics)
# {'recall@1': 0.52, 'recall@3': 0.74, 'recall@5': 0.83, 'mrr': 0.61}

# Evaluate answers
answer_metrics = evaluate_answers(dataset, your_rag_pipeline)
print(answer_metrics)
# {'string_em_accuracy': 0.43, 'avg_judge_score': 6.8, 'judge_pass_rate': 0.71}
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    D2L-QA Pipeline                           │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │ D2L Book │──▶│ Chunk & Parse│──▶│ QA Pair Generation│   │
│  │ (source) │   │  (Step 1)    │   │    (Step 2)       │   │
│  └──────────┘   └──────────────┘   └─────────┬─────────┘   │
│                                               │             │
│                                    ┌──────────▼─────────┐   │
│                                    │ Quality Filtering  │   │
│                                    │    (Step 3)        │   │
│                                    └──────────┬─────────┘   │
│                                               │             │
│                                    ┌──────────▼─────────┐   │
│                                    │ Final Dataset JSON │   │
│                                    │    (Step 4)        │   │
│                                    └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `claude-sonnet-4-20250514` | LLM model — auto-detects OpenAI or Anthropic |
| `--questions-per-chunk` | `3` | Number of questions to generate per chunk |
| `--sample-size` | `500` | Final dataset size (stratified sampling) |
| `--skip-validation` | `False` | Skip LLM-as-judge validation (faster, lower cost) |
| `--output-dir` | `.` | Directory to save output JSON files |

> **API keys** are loaded automatically from `.env`. No `--api-key` flag needed.

## Actual Output Scale (D2L English Edition)

```
Total sections parsed:  1,508
Total chunks generated: 2,467
Raw QA pairs:           ~3,800  (at 2 questions/chunk)
After filtering:        ~2,800  (75% retained)
After deduplication:    ~2,750
Final dataset:          configurable (50–2,000+)
```

## Question Types

| Type | Example |
|------|---------|
| **Factual** | "What is the default learning rate in Adam?" |
| **Conceptual** | "Why does batch normalization help with covariate shift?" |
| **Procedural** | "What are the steps to implement dropout?" |
| **Comparative** | "How does GRU differ from LSTM?" |
| **Mathematical** | "What is the cross-entropy loss formula?" |
| **Code-based** | "What PyTorch function defines a custom dataset?" |

## Benchmarking Metrics

### Retrieval Metrics
- **Recall@K**: Did retriever find gold chunk in top K results?
- **MRR**: Mean Reciprocal Rank of gold chunk

### Answer Metrics
- **StringEM**: Exact string match (strict, from PersonaRAG)
- **LLM Judge Score**: Semantic similarity (0–9 scale)
- **Judge Pass Rate**: % scoring ≥6

### Target Performance
| Metric | Target |
|--------|--------|
| Recall@3 | > 0.70 |
| Recall@5 | > 0.80 |
| MRR | > 0.60 |
| StringEM | > 0.40 |
| Judge Score | avg > 6.0 |
| Judge Pass Rate | > 0.65 |

## Best Practices

1. **Start small**: Use `--sample-size 50 --skip-validation` to verify setup before a full run
2. **Model choice**: `gpt-4o-mini` is cost-effective; use `gpt-4o` or `claude-sonnet-*` for higher quality
3. **Skip validation first**: Generation + filtering alone gives good results; add validation for production
4. **Version your dataset**: Tag with D2L commit hash for reproducibility
5. **Combine metrics**: StringEM is strict; use alongside LLM judge score for a balanced view

## Troubleshooting

### JSON Parse Errors (`Invalid \escape`)
D2L contains heavy LaTeX math (e.g. `\alpha`, `\frac`). The LLM may include bare
backslashes in JSON responses that `json.loads` rejects.

**This is handled automatically** via a 3-stage fallback in `parse_json_response`:
1. Direct `json.loads` (fast path)
2. Regex-escape invalid backslashes, retry
3. `json-repair` library (catches apostrophes, trailing commas, etc.)

Failures are now rare — only if the LLM returns completely malformed output.

### API Rate Limits
Add delays between chunks:
```python
import time
# In generate_qa_pairs loop:
time.sleep(0.5)
```

### Low Quality Questions
- Reduce `--questions-per-chunk` to 2
- Use a stronger model (`gpt-4o` instead of `gpt-4o-mini`)
- Enable validation: remove `--skip-validation`

## Citation

Based on methodology from:
- PersonaRAG evaluation framework
- Chatbot Arena benchmarking
- Natural Questions (NQ) dataset format

## License

MIT License — See textbook license at https://d2l.ai
