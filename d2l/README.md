# D2L-QA Ground Truth Dataset Generator

Generate a high-quality question-answer ground truth dataset from the *Dive into Deep Learning* (D2L) textbook for RAG pipeline evaluation and chatbot benchmarking.

## Features

✅ **Multi-type Question Generation**: Factual, conceptual, procedural, comparative, mathematical, and code-based questions  
✅ **Quality Filtering**: Automated groundedness checks, LLM-as-judge validation, and deduplication  
✅ **Evaluation Metrics**: Retrieval evaluation (Recall@K, MRR) and answer correctness (StringEM, LLM Judge)  
✅ **Stratified Sampling**: Ensures balanced coverage across chapters and difficulty levels  

## Installation

```bash
# Clone the D2L repository
git clone https://github.com/d2l-ai/d2l-en.git

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
python d2l_qa_generator.py /path/to/d2l-en \
  --output-dir ./output \
  --model claude-sonnet-4-20250514 \
  --questions-per-chunk 3 \
  --sample-size 500 \
  --api-key YOUR_API_KEY
```

### Using as a Python Module

```python
from d2l_qa_generator import main

main(
    d2l_root="/path/to/d2l-en",
    output_dir="./output",
    model="claude-sonnet-4-20250514",
    n_questions_per_chunk=3,
    sample_size=500,
    api_key="YOUR_API_KEY"
)
```

### Fast Mode (Skip LLM Validation)

```bash
python d2l_qa_generator.py /path/to/d2l-en \
  --skip-validation \
  --output-dir ./output
```

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
    "total_corpus_chunks": 1247,
    "question_types": {...},
    "difficulty_distribution": {...}
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
Corpus chunks for RAG indexing (subset of the full dataset for convenience).

## Evaluation

Use the generated dataset to evaluate your RAG pipeline:

```python
from d2l_qa_generator import evaluate_retrieval, evaluate_answers
import json

# Load dataset
with open("output/d2l_qa_ground_truth.json") as f:
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
| `--model` | `claude-sonnet-4-20250514` | LLM model for generation & validation |
| `--questions-per-chunk` | `3` | Number of questions to generate per chunk |
| `--sample-size` | `500` | Final dataset size (stratified sampling) |
| `--skip-validation` | `False` | Skip LLM-as-judge validation (faster) |
| `--api-key` | `None` | API key (or set `ANTHROPIC_API_KEY` env var) |

## Expected Output Scale

```
Total sections parsed: ~250–400
Total chunks generated: ~800–1500
Raw QA pairs:          ~2400–4500
After filtering:       ~1200–2500
Final dataset:         500 (configurable)
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
- **LLM Judge Score**: Semantic similarity (0-9 scale)
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

1. **Use a strong model for generation**: Claude Sonnet 4.5 or GPT-4 recommended
2. **Version your dataset**: Tag with D2L commit hash for reproducibility
3. **Combine metrics**: StringEM is strict; use with LLM judge for balance
4. **Start with 500 questions**: Following PersonaRAG methodology
5. **Enable validation for production**: Skip only for rapid prototyping

## Troubleshooting

### API Rate Limits
Add delays between chunks or use batch processing:
```python
import time
# In generate_qa_pairs loop:
time.sleep(1)  # 1 second between chunks
```

### Low Quality Questions
- Reduce `n_questions_per_chunk` (e.g., 2 instead of 3)
- Use a stronger model for generation
- Enable validation with `n_validations=5` for stricter filtering

### Memory Issues
Process in batches:
```python
# Process chunks 100 at a time
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    generate_qa_pairs(batch, ...)
```

## Citation

Based on methodology from:
- PersonaRAG evaluation framework
- Chatbot Arena benchmarking
- Natural Questions (NQ) dataset format

## License

MIT License - See textbook license at https://d2l.ai
