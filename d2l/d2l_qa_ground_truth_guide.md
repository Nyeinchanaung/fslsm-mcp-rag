# Generating a D2L-QA Ground Truth Dataset

## Technical Guide for RAG Pipeline Evaluation & Chatbot Benchmarking

---

## 1. Overview & Goals

You want to create a **question-answer ground truth dataset** from the *Dive into Deep Learning* (D2L) textbook that can be used to:

- **Evaluate a RAG pipeline**: measure retrieval accuracy (did it find the right chunk?) and answer correctness (did the LLM produce the right answer?)
- **Benchmark chatbot accuracy**: compare different models/configurations using StringEM, BLEU, and LLM-as-judge scoring (as used in PersonaRAG and Chatbot Arena evaluations)

The dataset format follows the structure used by NQ, TriviaQA, and WebQ (randomly sampled questions with gold answers mapped to source passages).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    D2L-QA Pipeline                           │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │ D2L Book │──▶│ Chunk & Parse│──▶│ QA Pair Generation│   │
│  │ (source) │   │  (Step 3)    │   │    (Step 4)       │   │
│  └──────────┘   └──────────────┘   └─────────┬─────────┘   │
│                                               │             │
│                                    ┌──────────▼─────────┐   │
│                                    │ Quality Filtering  │   │
│                                    │    (Step 5)        │   │
│                                    └──────────┬─────────┘   │
│                                               │             │
│                                    ┌──────────▼─────────┐   │
│                                    │ Final Dataset JSON │   │
│                                    │    (Step 6)        │   │
│                                    └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Step 1 — Source Extraction & Chunking

### 3.1 Obtain D2L Content

The D2L textbook is open-source and available as HTML, PDF, or Jupyter notebooks. The HTML or notebook format is preferred because it preserves code blocks and math.

```python
# Option A: Clone the D2L repo (preferred — structured .md/.ipynb)
# git clone https://github.com/d2l-ai/d2l-en.git

# Option B: If you have a PDF
# Use PyMuPDF or pdfplumber to extract text per page
```

### 3.2 Parse into Sections

```python
import json, re, os, hashlib
from pathlib import Path

def parse_d2l_chapters(d2l_root: str) -> list[dict]:
    """
    Parse D2L textbook into structured sections.
    Each section = one chunking unit with metadata.
    """
    sections = []
    
    for md_file in sorted(Path(d2l_root).rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        
        # Split by headings (## or ###)
        parts = re.split(r'\n(#{2,3}\s+.+)\n', text)
        
        chapter_name = md_file.stem
        current_heading = chapter_name
        
        for i, part in enumerate(parts):
            if re.match(r'^#{2,3}\s+', part):
                current_heading = part.strip("# ").strip()
                continue
            
            content = part.strip()
            if len(content) < 50:  # skip trivially short sections
                continue
            
            section_id = hashlib.md5(
                f"{chapter_name}:{current_heading}".encode()
            ).hexdigest()[:12]
            
            sections.append({
                "section_id": section_id,
                "chapter": chapter_name,
                "heading": current_heading,
                "content": content,
                "source_file": str(md_file),
                "char_count": len(content),
            })
    
    return sections
```

### 3.3 Chunk the Sections

```python
def chunk_sections(
    sections: list[dict],
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    chars_per_token: float = 4.0
) -> list[dict]:
    """
    Split sections into overlapping chunks suitable for embedding/retrieval.
    Each chunk retains metadata about its parent section.
    """
    max_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)
    chunks = []
    
    for section in sections:
        text = section["content"]
        
        if len(text) <= max_chars:
            chunks.append({
                **section,
                "chunk_id": f"{section['section_id']}_0",
                "chunk_index": 0,
                "chunk_text": text,
            })
        else:
            # Sliding window with sentence-boundary alignment
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_len = 0
            chunk_idx = 0
            
            for sentence in sentences:
                if current_len + len(sentence) > max_chars and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        **section,
                        "chunk_id": f"{section['section_id']}_{chunk_idx}",
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text,
                    })
                    chunk_idx += 1
                    
                    # Keep overlap
                    overlap_text = ""
                    overlap_sents = []
                    for s in reversed(current_chunk):
                        if len(overlap_text) + len(s) > overlap_chars:
                            break
                        overlap_sents.insert(0, s)
                        overlap_text = " ".join(overlap_sents)
                    current_chunk = overlap_sents
                    current_len = len(overlap_text)
                
                current_chunk.append(sentence)
                current_len += len(sentence) + 1
            
            # Final chunk
            if current_chunk:
                chunks.append({
                    **section,
                    "chunk_id": f"{section['section_id']}_{chunk_idx}",
                    "chunk_index": chunk_idx,
                    "chunk_text": " ".join(current_chunk),
                })
    
    return chunks
```

**Expected output after Step 1:**

```
Total sections parsed: ~250–400 (depending on D2L edition)
Total chunks generated: ~800–1500
Average chunk length:   ~400 tokens
Chapters covered:       Preliminaries, Linear NN, Deep Learning Basics,
                        CNN, RNN, Attention, Optimization, CV, NLP, etc.
```

---

## 4. Step 2 — QA Pair Generation

### 4.1 Question Taxonomy

Generate multiple *types* of questions per chunk to ensure coverage:

| Type | Description | Example |
|------|-------------|---------|
| **Factual** | Direct fact from text | "What is the default learning rate used in the Adam optimizer example?" |
| **Conceptual** | Understanding a concept | "Why does batch normalization help with internal covariate shift?" |
| **Procedural** | How-to / steps | "What are the steps to implement dropout regularization in a neural network?" |
| **Comparative** | Differences between concepts | "How does GRU differ from LSTM in terms of gating mechanisms?" |
| **Mathematical** | Formulas / derivations | "What is the formula for the cross-entropy loss function?" |
| **Code-based** | About code examples | "What PyTorch function is used to define a custom dataset in D2L?" |

### 4.2 Generation Prompt Template

```python
QA_GENERATION_PROMPT = """You are an expert AI/ML educator creating a QA evaluation dataset
from the textbook "Dive into Deep Learning" (D2L).

Given the following textbook passage, generate {n_questions} question-answer pairs.

RULES:
1. Each question must be answerable SOLELY from the given passage.
2. Answers must be concise (1-3 sentences) and directly grounded in the passage.
3. Include the EXACT quote or key phrase from the passage that supports the answer.
4. Generate a MIX of question types: factual, conceptual, procedural, comparative, code-based.
5. Questions should be phrased as a student would naturally ask them.
6. Avoid trivial yes/no questions.
7. For mathematical content, include the relevant formula in the answer.

PASSAGE:
Chapter: {chapter}
Section: {heading}
---
{chunk_text}
---

Respond in JSON format:
[
  {{
    "question": "...",
    "answer": "...",
    "answer_type": "factual|conceptual|procedural|comparative|mathematical|code_based",
    "supporting_quote": "exact text from passage that grounds this answer",
    "difficulty": "easy|medium|hard"
  }}
]
"""
```

### 4.3 Generation Pseudocode

```python
import openai  # or anthropic, etc.

def generate_qa_pairs(
    chunks: list[dict],
    model: str = "claude-sonnet-4-20250514",
    n_questions_per_chunk: int = 3,
    temperature: float = 0.4,
) -> list[dict]:
    """
    Generate QA pairs from each chunk using an LLM.
    """
    all_qa_pairs = []
    
    for chunk in chunks:
        prompt = QA_GENERATION_PROMPT.format(
            n_questions=n_questions_per_chunk,
            chapter=chunk["chapter"],
            heading=chunk["heading"],
            chunk_text=chunk["chunk_text"],
        )
        
        response = call_llm(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=2000,
        )
        
        qa_pairs = parse_json_response(response)
        
        for qa in qa_pairs:
            qa["chunk_id"] = chunk["chunk_id"]
            qa["chapter"] = chunk["chapter"]
            qa["heading"] = chunk["heading"]
            qa["source_passage"] = chunk["chunk_text"]
        
        all_qa_pairs.extend(qa_pairs)
    
    return all_qa_pairs


def call_llm(model, prompt, temperature, max_tokens):
    """Wrapper — replace with your actual API call."""
    # For Anthropic:
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model=model,
    #     max_tokens=max_tokens,
    #     temperature=temperature,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.content[0].text
    
    # For OpenAI:
    # client = openai.OpenAI()
    # response = client.chat.completions.create(...)
    pass


def parse_json_response(response_text: str) -> list[dict]:
    """Safely parse JSON from LLM response, handling markdown fences."""
    text = response_text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)
```

**Expected output after Step 2:**

```
Total raw QA pairs generated: ~2400–4500 (3 per chunk × 800–1500 chunks)
Distribution by type:
  - Factual:      ~30%
  - Conceptual:   ~25%
  - Procedural:   ~15%
  - Comparative:  ~10%
  - Mathematical: ~10%
  - Code-based:   ~10%
```

---

## 5. Step 3 — Quality Filtering & Validation

This is **critical**. Raw LLM-generated QA pairs contain noise. Apply multiple filters.

### 5.1 Automated Filters

```python
def filter_qa_pairs(qa_pairs: list[dict]) -> list[dict]:
    """Apply automated quality filters to remove low-quality pairs."""
    filtered = []
    rejection_reasons = Counter()
    
    for qa in qa_pairs:
        # Filter 1: Answer must be grounded in source
        if not is_grounded(qa["answer"], qa["source_passage"]):
            rejection_reasons["not_grounded"] += 1
            continue
        
        # Filter 2: Question not too short or too generic
        if len(qa["question"].split()) < 5:
            rejection_reasons["question_too_short"] += 1
            continue
        
        # Filter 3: Answer not too long (>3 sentences is not a good GT)
        if len(qa["answer"].split('.')) > 4:
            rejection_reasons["answer_too_long"] += 1
            continue
        
        # Filter 4: No self-referential questions
        if any(phrase in qa["question"].lower() for phrase in 
               ["this passage", "the above", "according to the text"]):
            rejection_reasons["self_referential"] += 1
            continue
        
        # Filter 5: Deduplication by question similarity
        # (checked later in batch)
        
        filtered.append(qa)
    
    print(f"Kept {len(filtered)}/{len(qa_pairs)} pairs")
    print(f"Rejections: {dict(rejection_reasons)}")
    return filtered


def is_grounded(answer: str, passage: str) -> bool:
    """
    Check if the answer is grounded in the passage.
    Uses token overlap ratio as a quick heuristic.
    """
    answer_tokens = set(answer.lower().split())
    passage_tokens = set(passage.lower().split())
    
    # Remove stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                 "at", "to", "for", "of", "and", "or", "it", "this", "that"}
    answer_tokens -= stopwords
    passage_tokens -= stopwords
    
    if not answer_tokens:
        return False
    
    overlap = len(answer_tokens & passage_tokens) / len(answer_tokens)
    return overlap >= 0.5  # At least 50% of answer tokens appear in passage
```

### 5.2 LLM-as-Judge Validation (following Chatbot Arena methodology)

```python
VALIDATION_PROMPT = """You are validating a QA pair for a ground truth evaluation dataset.
The QA pair was generated from a textbook passage.

PASSAGE:
{passage}

QUESTION: {question}
PROPOSED ANSWER: {answer}

Rate this QA pair on a scale of 1-5 for each criterion:
1. **Answerable**: Is the question clearly answerable from the passage alone? (1=no, 5=yes)
2. **Correct**: Is the proposed answer factually correct given the passage? (1=wrong, 5=correct)
3. **Specific**: Is the question specific enough to have a clear answer? (1=vague, 5=specific)
4. **Useful**: Would this question meaningfully test a RAG system? (1=trivial, 5=challenging)

Respond as JSON:
{{"answerable": N, "correct": N, "specific": N, "useful": N, "overall_pass": true/false}}

Mark overall_pass=true only if ALL scores >= 3 and at least two scores >= 4.
"""

def validate_with_llm_judge(
    qa_pairs: list[dict],
    judge_model: str = "claude-sonnet-4-20250514",
    n_validations: int = 3,  # run each pair N times for consistency
) -> list[dict]:
    """
    Use LLM-as-judge with self-consistency (majority vote)
    following the approach from Chatbot Arena / PersonaRAG evaluations.
    """
    validated = []
    
    for qa in qa_pairs:
        votes = []
        for _ in range(n_validations):
            prompt = VALIDATION_PROMPT.format(
                passage=qa["source_passage"],
                question=qa["question"],
                answer=qa["answer"],
            )
            result = call_llm(judge_model, prompt, temperature=0.3, max_tokens=200)
            parsed = parse_json_response(result)
            votes.append(parsed["overall_pass"])
        
        # Majority vote (self-consistency prompting)
        passed = sum(votes) > len(votes) / 2
        
        if passed:
            qa["validation_score"] = sum(votes) / len(votes)
            validated.append(qa)
    
    return validated
```

### 5.3 Deduplication

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_questions(qa_pairs: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate questions using TF-IDF cosine similarity."""
    questions = [qa["question"] for qa in qa_pairs]
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(questions)
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    to_remove = set()
    for i in range(len(questions)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(questions)):
            if sim_matrix[i, j] > threshold:
                to_remove.add(j)  # keep earlier, remove later
    
    deduped = [qa for idx, qa in enumerate(qa_pairs) if idx not in to_remove]
    print(f"Deduplication: {len(qa_pairs)} → {len(deduped)} ({len(to_remove)} removed)")
    return deduped
```

**Expected output after Step 3:**

```
After groundedness filter:  ~85% retained
After LLM judge:           ~70% of remainder pass
After deduplication:        ~90% of remainder kept
────────────────────────────────────────────────
Final dataset size:         ~1200–2500 QA pairs
```

---

## 6. Step 4 — Dataset Formatting

### 6.1 Final JSON Schema

```python
def format_final_dataset(
    qa_pairs: list[dict],
    chunks: list[dict],
    sample_size: int = 500,  # following PersonaRAG's sampling approach
) -> dict:
    """
    Format into final evaluation dataset.
    Optionally sample a manageable subset (PersonaRAG used 500 per dataset).
    """
    import random
    
    # Build chunk lookup for retrieval evaluation
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    
    # Stratified sampling by chapter to ensure coverage
    by_chapter = defaultdict(list)
    for qa in qa_pairs:
        by_chapter[qa["chapter"]].append(qa)
    
    sampled = []
    per_chapter = max(1, sample_size // len(by_chapter))
    for chapter, pairs in by_chapter.items():
        sampled.extend(random.sample(pairs, min(per_chapter, len(pairs))))
    
    # Trim to exact sample size
    if len(sampled) > sample_size:
        sampled = random.sample(sampled, sample_size)
    
    # Format output
    dataset = {
        "metadata": {
            "name": "D2L-QA",
            "version": "1.0",
            "source": "Dive into Deep Learning (d2l.ai)",
            "total_questions": len(sampled),
            "total_corpus_chunks": len(chunks),
            "sampling_rate": f"{len(sampled)/len(qa_pairs)*100:.1f}%",
            "question_types": dict(Counter(qa["answer_type"] for qa in sampled)),
            "difficulty_distribution": dict(Counter(qa["difficulty"] for qa in sampled)),
        },
        "corpus": [
            {
                "chunk_id": c["chunk_id"],
                "chapter": c["chapter"],
                "heading": c["heading"],
                "text": c["chunk_text"],
            }
            for c in chunks
        ],
        "questions": [
            {
                "question_id": f"d2l_q_{i:04d}",
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "answer_type": qa["answer_type"],
                "difficulty": qa["difficulty"],
                "supporting_quote": qa["supporting_quote"],
                "gold_chunk_ids": [qa["chunk_id"]],  # for retrieval eval
                "chapter": qa["chapter"],
                "section": qa["heading"],
            }
            for i, qa in enumerate(sampled)
        ],
    }
    
    return dataset
```

### 6.2 Example Output Record

```json
{
  "question_id": "d2l_q_0042",
  "question": "What are the two main differences between GRU and LSTM gating mechanisms?",
  "gold_answer": "GRU uses two gates (reset and update) while LSTM uses three gates (input, forget, output). GRU merges the cell state and hidden state into a single hidden state, making it computationally simpler.",
  "answer_type": "comparative",
  "difficulty": "medium",
  "supporting_quote": "The GRU has two gates: the reset gate and the update gate... Unlike LSTM, GRU merges the cell state and hidden state.",
  "gold_chunk_ids": ["a3f2b1c9d4e5_2"],
  "chapter": "09_recurrent-modern",
  "section": "Gated Recurrent Units (GRU)"
}
```

---

## 7. Step 5 — Evaluation Metrics

Once you have the dataset, use these metrics to evaluate your RAG pipeline and chatbot.

### 7.1 Retrieval Evaluation

```python
def evaluate_retrieval(dataset: dict, retriever, k_values=[1, 3, 5]):
    """
    Measure whether the retriever finds the correct chunk(s).
    """
    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    
    for q in dataset["questions"]:
        retrieved = retriever.search(q["question"], top_k=max(k_values))
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        gold_ids = set(q["gold_chunk_ids"])
        
        # Recall@K
        for k in k_values:
            top_k_ids = set(retrieved_ids[:k])
            hit = len(gold_ids & top_k_ids) > 0
            results[f"recall@{k}"].append(hit)
        
        # MRR (Mean Reciprocal Rank)
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in gold_ids:
                results["mrr"].append(1.0 / rank)
                break
        else:
            results["mrr"].append(0.0)
    
    return {k: sum(v)/len(v) for k, v in results.items()}
```

### 7.2 Answer Correctness (StringEM + LLM Judge)

Following PersonaRAG's evaluation protocol:

```python
def evaluate_answers(dataset: dict, rag_pipeline):
    """
    Evaluate answer quality using:
    1. StringEM (exact match after normalization) — from PersonaRAG
    2. LLM-as-judge scoring (1-10 semantic similarity) — from Arena/PersonaRAG
    """
    string_em_scores = []
    judge_scores = []
    
    for q in dataset["questions"]:
        # Get RAG pipeline answer
        predicted = rag_pipeline.answer(q["question"])
        gold = q["gold_answer"]
        
        # --- Metric 1: StringEM ---
        pred_norm = predicted.lower().strip()
        gold_norm = gold.lower().strip()
        em = gold_norm in pred_norm  # contains match
        string_em_scores.append(em)
        
        # --- Metric 2: LLM Judge (semantic similarity) ---
        judge_prompt = f"""Rate the semantic similarity between the
predicted answer and the gold answer on a scale of 0-9.

Gold answer: {gold}
Predicted answer: {predicted}

Score (0-9):"""
        
        score = int(call_llm("claude-sonnet-4-20250514", judge_prompt, 0.0, 10))
        judge_scores.append(score)
    
    return {
        "string_em_accuracy": sum(string_em_scores) / len(string_em_scores),
        "avg_judge_score": sum(judge_scores) / len(judge_scores),
        "judge_pass_rate": sum(1 for s in judge_scores if s >= 6) / len(judge_scores),
    }
```

### 7.3 Summary Metrics Table

| Metric | What It Measures | Target Range |
|--------|-----------------|--------------|
| **Recall@3** | Retriever finds gold chunk in top 3 | >0.70 |
| **Recall@5** | Retriever finds gold chunk in top 5 | >0.80 |
| **MRR** | Avg reciprocal rank of gold chunk | >0.60 |
| **StringEM** | Exact match (gold answer in prediction) | >0.40 |
| **Judge Score** | Semantic correctness (0-9) | avg >6.0 |
| **Judge Pass Rate** | % scoring ≥6 | >0.65 |
| **BLEU-2** | N-gram overlap with gold answer | informational |

---

## 8. Full Pipeline — Putting It All Together

```python
def main():
    # ── Step 1: Parse & Chunk ──
    sections = parse_d2l_chapters("./d2l-en/chapter_*/")
    chunks = chunk_sections(sections, max_tokens=512, overlap_tokens=64)
    print(f"Parsed {len(sections)} sections → {len(chunks)} chunks")
    
    # ── Step 2: Generate QA Pairs ──
    raw_qa = generate_qa_pairs(chunks, n_questions_per_chunk=3)
    print(f"Generated {len(raw_qa)} raw QA pairs")
    
    # ── Step 3: Filter & Validate ──
    filtered = filter_qa_pairs(raw_qa)
    validated = validate_with_llm_judge(filtered, n_validations=3)
    deduped = deduplicate_questions(validated, threshold=0.85)
    print(f"After filtering: {len(deduped)} validated QA pairs")
    
    # ── Step 4: Format Dataset ──
    dataset = format_final_dataset(deduped, chunks, sample_size=500)
    
    # ── Save ──
    with open("d2l_qa_ground_truth.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    # ── Save corpus separately for RAG indexing ──
    with open("d2l_corpus_chunks.json", "w") as f:
        json.dump(dataset["corpus"], f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"D2L-QA Dataset Summary")
    print(f"{'='*50}")
    print(f"Questions:       {dataset['metadata']['total_questions']}")
    print(f"Corpus chunks:   {dataset['metadata']['total_corpus_chunks']}")
    print(f"Sampling rate:   {dataset['metadata']['sampling_rate']}")
    print(f"Question types:  {dataset['metadata']['question_types']}")
    print(f"Difficulty dist: {dataset['metadata']['difficulty_distribution']}")
    
    # ── Step 5: Evaluate your RAG pipeline ──
    # retrieval_results = evaluate_retrieval(dataset, your_retriever)
    # answer_results = evaluate_answers(dataset, your_rag_pipeline)


if __name__ == "__main__":
    main()
```

---

## 9. Expected Final Output

```
==================================================
D2L-QA Dataset Summary
==================================================
Questions:       500
Corpus chunks:   1,247
Sampling rate:   20.3%
Question types:  {'factual': 148, 'conceptual': 126, 'procedural': 78,
                  'comparative': 52, 'mathematical': 51, 'code_based': 45}
Difficulty dist: {'easy': 125, 'medium': 250, 'hard': 125}

Files created:
  - d2l_qa_ground_truth.json  (full dataset with questions + corpus)
  - d2l_corpus_chunks.json    (corpus only, for RAG indexing)
```

### Sample Evaluation Results (what to expect):

```
Retrieval Metrics:
  Recall@1: 0.52
  Recall@3: 0.74
  Recall@5: 0.83
  MRR:      0.61

Answer Metrics:
  StringEM Accuracy:  0.43
  Avg Judge Score:    6.8 / 9
  Judge Pass Rate:    0.71
```

---

## 10. Tips & Best Practices

**From the research literature in your project:**

1. **Sample size**: PersonaRAG used 500 questions per dataset as a cost-effective balance. Start there, scale up if needed.

2. **LLM-as-judge consistency**: Run each validation 3-5 times and use majority vote (self-consistency prompting), as described in the survey evaluation methodology. Mean variance should be <0.56 on a 0-9 scale.

3. **Use a stronger model for generation than for evaluation target**: Generate QA pairs with a frontier model (e.g., Claude Sonnet 4.5), then evaluate a smaller/cheaper model as your RAG answer generator.

4. **StringEM is strict**: Following PersonaRAG, convert both outputs to lowercase and check if the gold answer string is contained in the prediction. This is intentionally strict — combine it with the LLM judge score for a balanced view.

5. **Code questions are special**: D2L has extensive code examples. For code-based questions, consider using execution-based evaluation (does the code run and produce the expected output?) rather than string matching.

6. **Version your dataset**: D2L is actively maintained. Tag your dataset with the D2L commit hash or version number so results are reproducible.

7. **Difficulty calibration**: "Easy" questions should have answers directly stated in a single sentence. "Hard" questions should require synthesizing information across a paragraph or understanding mathematical notation.
