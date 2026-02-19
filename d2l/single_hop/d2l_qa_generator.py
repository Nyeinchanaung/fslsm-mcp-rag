"""
D2L-QA Ground Truth Dataset Generator

This module generates a question-answer ground truth dataset from the 
Dive into Deep Learning (D2L) textbook for RAG pipeline evaluation.

Based on: d2l_qa_ground_truth_guide.md
"""

import json
import re
import os
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional
import random

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Optional imports (install as needed)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Deduplication will be skipped.")

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Debug: Show if API keys are loaded (only first 10 chars for security)
if OPENAI_API_KEY:
    print(f"✓ OPENAI_API_KEY loaded: {OPENAI_API_KEY[:10]}...")
if ANTHROPIC_API_KEY:
    print(f"✓ ANTHROPIC_API_KEY loaded: {ANTHROPIC_API_KEY[:10]}...")


# ==============================================================================
# STEP 1: Source Extraction & Chunking
# ==============================================================================

def parse_d2l_chapters(d2l_root: str) -> List[Dict]:
    """
    Parse D2L textbook into structured sections.
    Each section = one chunking unit with metadata.
    
    Args:
        d2l_root: Path to D2L repository root
        
    Returns:
        List of section dictionaries
    """
    sections = []
    
    for md_file in sorted(Path(d2l_root).rglob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read {md_file}: {e}")
            continue
        
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


def chunk_sections(
    sections: List[Dict],
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    chars_per_token: float = 4.0
) -> List[Dict]:
    """
    Split sections into overlapping chunks suitable for embedding/retrieval.
    Each chunk retains metadata about its parent section.
    
    Args:
        sections: List of section dictionaries
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks
        chars_per_token: Approximate characters per token
        
    Returns:
        List of chunk dictionaries
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


# ==============================================================================
# STEP 2: QA Pair Generation
# ==============================================================================

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


def call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 2000,
    api_key: Optional[str] = None
) -> str:
    """
    Wrapper for LLM API calls.
    Automatically detects which API to use based on model name.
    
    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4')
        prompt: Prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: Optional API key (auto-detected if None)
        
    Returns:
        LLM response text
    """
    # Detect which API to use based on model name
    is_openai = any(prefix in model.lower() for prefix in ['gpt', 'o1', 'o3'])
    is_anthropic = any(prefix in model.lower() for prefix in ['claude', 'sonnet', 'opus', 'haiku'])
    
    # Use OpenAI
    if is_openai:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key or OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except ImportError:
            raise NotImplementedError("OpenAI library not installed. Run: pip install openai")
    
    # Use Anthropic
    elif is_anthropic:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except ImportError:
            raise NotImplementedError("Anthropic library not installed. Run: pip install anthropic")
    
    # Unknown model
    else:
        raise ValueError(
            f"Unknown model '{model}'. Use OpenAI models (gpt-4o, gpt-4o-mini) "
            f"or Anthropic models (claude-sonnet-4, claude-opus-4)"
        )


def parse_json_response(response_text: str) -> List[Dict]:
    """
    Robustly parse JSON from LLM response with 3-stage fallback:
      1. Direct json.loads (clean responses)
      2. Fix invalid LaTeX backslash escapes with regex, retry
      3. json-repair library (handles apostrophes, trailing commas, etc.)

    D2L's heavy LaTeX content causes LLMs to include bare backslashes
    (e.g. \\alpha, \\frac) in JSON strings, which json.loads rejects.
    """
    text = response_text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Stage 1: Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Stage 2: Fix invalid backslash escapes (LaTeX: \alpha, \frac, etc.)
    # Replace any \ not followed by a valid JSON escape char with \\\\
    fixed = re.sub(r'\\(?!["\\\//bfnrtu])', r'\\\\', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Stage 3: Fall back to json-repair for other malformed JSON
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, list):
            return result
        return [result] if isinstance(result, dict) else []
    except Exception:
        pass

    # All stages failed — re-raise original error for caller to catch
    raise json.JSONDecodeError("All parse strategies failed", text, 0)


def generate_qa_pairs(
    chunks: List[Dict],
    model: str = "claude-sonnet-4-20250514",
    n_questions_per_chunk: int = 3,
    temperature: float = 0.4,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Generate QA pairs from each chunk using an LLM.
    
    Args:
        chunks: List of chunk dictionaries
        model: LLM model to use
        n_questions_per_chunk: Number of questions per chunk
        temperature: Sampling temperature
        api_key: Optional API key
        
    Returns:
        List of QA pair dictionaries
    """
    all_qa_pairs = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        prompt = QA_GENERATION_PROMPT.format(
            n_questions=n_questions_per_chunk,
            chapter=chunk["chapter"],
            heading=chunk["heading"],
            chunk_text=chunk["chunk_text"],
        )
        
        try:
            response = call_llm(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=2000,
                api_key=api_key
            )
            
            qa_pairs = parse_json_response(response)
            
            for qa in qa_pairs:
                qa["chunk_id"] = chunk["chunk_id"]
                qa["chapter"] = chunk["chapter"]
                qa["heading"] = chunk["heading"]
                qa["source_passage"] = chunk["chunk_text"]
            
            all_qa_pairs.extend(qa_pairs)
        except Exception as e:
            print(f"\nWarning: Failed to generate QA pairs for chunk {chunk['chunk_id']}: {e}")
            continue
    
    print(f"\nGenerated {len(all_qa_pairs)} QA pairs from {len(chunks)} chunks")
    return all_qa_pairs


# ==============================================================================
# STEP 3: Quality Filtering & Validation
# ==============================================================================

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


def filter_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
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
        
        filtered.append(qa)
    
    print(f"Kept {len(filtered)}/{len(qa_pairs)} pairs")
    print(f"Rejections: {dict(rejection_reasons)}")
    return filtered


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
    qa_pairs: List[Dict],
    judge_model: str = "claude-sonnet-4-20250514",
    n_validations: int = 3,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Use LLM-as-judge with self-consistency (majority vote)
    following the approach from Chatbot Arena / PersonaRAG evaluations.
    """
    validated = []
    
    for i, qa in enumerate(qa_pairs):
        print(f"Validating {i+1}/{len(qa_pairs)}...")
        
        votes = []
        for _ in range(n_validations):
            prompt = VALIDATION_PROMPT.format(
                passage=qa["source_passage"][:1000],  # Truncate for API limits
                question=qa["question"],
                answer=qa["answer"],
            )
            
            try:
                result = call_llm(
                    judge_model, prompt, 
                    temperature=0.3, 
                    max_tokens=200,
                    api_key=api_key
                )
                parsed = parse_json_response(result)
                votes.append(parsed["overall_pass"])
            except Exception as e:
                print(f"\nWarning: Validation failed for QA pair: {e}")
                continue
        
        # Majority vote (self-consistency prompting)
        if votes:
            passed = sum(votes) > len(votes) / 2
            
            if passed:
                qa["validation_score"] = sum(votes) / len(votes)
                validated.append(qa)
    
    print(f"\nValidated {len(validated)}/{len(qa_pairs)} pairs")
    return validated


def deduplicate_questions(qa_pairs: List[Dict], threshold: float = 0.85) -> List[Dict]:
    """Remove near-duplicate questions using TF-IDF cosine similarity."""
    if not SKLEARN_AVAILABLE:
        print("Warning: sklearn not available. Skipping deduplication.")
        return qa_pairs
    
    # Handle empty or very small lists
    if len(qa_pairs) <= 1:
        print(f"Skipping deduplication: only {len(qa_pairs)} question(s)")
        return qa_pairs
    
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


# ==============================================================================
# STEP 4: Dataset Formatting
# ==============================================================================

def format_final_dataset(
    qa_pairs: List[Dict],
    chunks: List[Dict],
    sample_size: int = 500,
) -> Dict:
    """
    Format into final evaluation dataset.
    Optionally sample a manageable subset (PersonaRAG used 500 per dataset).
    """
    # Handle empty QA pairs
    if not qa_pairs:
        print("Warning: No QA pairs to format. Returning empty dataset.")
        return {
            "metadata": {
                "name": "D2L-QA",
                "version": "1.0",
                "source": "Dive into Deep Learning (d2l.ai)",
                "total_questions": 0,
                "total_corpus_chunks": len(chunks),
                "sampling_rate": "0.0%",
                "question_types": {},
                "difficulty_distribution": {},
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
            "questions": [],
        }
    
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
            "question_types": dict(Counter(qa.get("answer_type", "unknown") for qa in sampled)),
            "difficulty_distribution": dict(Counter(qa.get("difficulty", "medium") for qa in sampled)),
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
                "answer_type": qa.get("answer_type", "unknown"),
                "difficulty": qa.get("difficulty", "medium"),
                "supporting_quote": qa.get("supporting_quote", ""),
                "gold_chunk_ids": [qa["chunk_id"]],  # for retrieval eval
                "chapter": qa["chapter"],
                "section": qa["heading"],
            }
            for i, qa in enumerate(sampled)
        ],
    }
    
    return dataset


# ==============================================================================
# STEP 5: Evaluation Metrics
# ==============================================================================

def evaluate_retrieval(dataset: Dict, retriever, k_values=[1, 3, 5]):
    """
    Measure whether the retriever finds the correct chunk(s).
    
    Args:
        dataset: The D2L-QA dataset
        retriever: Object with a search(query, top_k) method
        k_values: List of k values for Recall@K
        
    Returns:
        Dictionary of retrieval metrics
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


def evaluate_answers(dataset: Dict, rag_pipeline, api_key: Optional[str] = None):
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
        
        try:
            score_text = call_llm(
                "claude-sonnet-4-20250514", 
                judge_prompt, 
                0.0, 
                10,
                api_key=api_key
            )
            score = int(re.search(r'\d+', score_text).group())
            judge_scores.append(score)
        except Exception as e:
            print(f"Warning: Judge scoring failed: {e}")
            judge_scores.append(0)
    
    return {
        "string_em_accuracy": sum(string_em_scores) / len(string_em_scores),
        "avg_judge_score": sum(judge_scores) / len(judge_scores),
        "judge_pass_rate": sum(1 for s in judge_scores if s >= 6) / len(judge_scores),
    }


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main(
    d2l_root: str,
    output_dir: str = ".",
    model: str = "claude-sonnet-4-20250514",
    n_questions_per_chunk: int = 3,
    sample_size: int = 500,
    api_key: Optional[str] = None,
    skip_validation: bool = False
):
    """
    Main pipeline to generate D2L-QA ground truth dataset.
    
    Args:
        d2l_root: Path to D2L repository
        output_dir: Directory to save outputs
        model: LLM model to use
        n_questions_per_chunk: Questions per chunk
        sample_size: Final dataset size
        api_key: Optional API key
        skip_validation: Skip LLM validation step (faster but lower quality)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Step 1: Parse & Chunk ──
    print("Step 1: Parsing and chunking D2L content...")
    sections = parse_d2l_chapters(d2l_root)
    chunks = chunk_sections(sections, max_tokens=512, overlap_tokens=64)
    print(f"Parsed {len(sections)} sections → {len(chunks)} chunks\n")
    
    # ── Step 2: Generate QA Pairs ──
    print("Step 2: Generating QA pairs...")
    raw_qa = generate_qa_pairs(
        chunks, 
        model=model, 
        n_questions_per_chunk=n_questions_per_chunk,
        api_key=api_key
    )
    print(f"Generated {len(raw_qa)} raw QA pairs\n")
    
    # ── Step 3: Filter & Validate ──
    print("Step 3: Filtering and validating...")
    filtered = filter_qa_pairs(raw_qa)
    
    if not skip_validation:
        validated = validate_with_llm_judge(filtered, judge_model=model, api_key=api_key)
    else:
        validated = filtered
        print("Skipping LLM validation")
    
    deduped = deduplicate_questions(validated, threshold=0.85)
    print(f"After filtering: {len(deduped)} validated QA pairs\n")
    
    # ── Step 4: Format Dataset ──
    print("Step 4: Formatting final dataset...")
    dataset = format_final_dataset(deduped, chunks, sample_size=sample_size)
    
    # ── Save ──
    output_path = os.path.join(output_dir, "d2l_qa_ground_truth.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    corpus_path = os.path.join(output_dir, "d2l_corpus_chunks.json")
    with open(corpus_path, "w") as f:
        json.dump(dataset["corpus"], f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"D2L-QA Dataset Summary")
    print(f"{'='*50}")
    print(f"Questions:       {dataset['metadata']['total_questions']}")
    print(f"Corpus chunks:   {dataset['metadata']['total_corpus_chunks']}")
    print(f"Sampling rate:   {dataset['metadata']['sampling_rate']}")
    print(f"Question types:  {dataset['metadata']['question_types']}")
    print(f"Difficulty dist: {dataset['metadata']['difficulty_distribution']}")
    print(f"\nFiles created:")
    print(f"  - {output_path}")
    print(f"  - {corpus_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate D2L-QA ground truth dataset")
    parser.add_argument("d2l_root", help="Path to D2L repository root")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LLM model")
    parser.add_argument("--questions-per-chunk", type=int, default=3, help="Questions per chunk")
    parser.add_argument("--sample-size", type=int, default=500, help="Final dataset size")
    parser.add_argument("--api-key", help="API key for LLM")
    parser.add_argument("--skip-validation", action="store_true", help="Skip LLM validation")
    
    args = parser.parse_args()
    
    main(
        d2l_root=args.d2l_root,
        output_dir=args.output_dir,
        model=args.model,
        n_questions_per_chunk=args.questions_per_chunk,
        sample_size=args.sample_size,
        api_key=args.api_key,
        skip_validation=args.skip_validation
    )
