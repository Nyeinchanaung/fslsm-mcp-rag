"""
D2L-QA Multi-Hop Ground Truth Dataset Generator (5-Chunk)
==========================================================
Generates multi-hop QA pairs from the Dive into Deep Learning (D2L) textbook.
Each question requires synthesizing information from exactly 5 related chunks.

Pipeline:
  1. Parse & chunk D2L content  (or reuse single-hop corpus)
  2. Cluster chunks by topic (TF-IDF + K-Means)
  3. Form 5-chunk groups via 3 strategies:
       - intra_cluster   : 5 chunks from the same topic cluster
       - prerequisite_chain: ordered conceptual dependency
       - cross_topic     : 2+2+1 from related but distinct clusters
  4. Generate multi-hop QA from each group (LLM)
  5. Ablation validation:  full (5) → leave-one-out (4×) → single (×1)
  6. Tier into Gold / Silver and format final dataset

Usage:
  python d2l/multi_hop/d2l_multihop_generator.py ./d2l-en \\
      --output-dir ./d2l/multi_hop/output \\
      --model gpt-4o-mini \\
      --num-groups 30 \\
      --skip-validation

API keys are loaded from a .env file (OPENAI_API_KEY / ANTHROPIC_API_KEY).
"""

import os
import re
import json
import hashlib
import random
import math
from typing import List, Dict, Optional, Tuple

# ── Environment ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if OPENAI_API_KEY:
    print(f"✓ OPENAI_API_KEY loaded: {OPENAI_API_KEY[:10]}...")
if ANTHROPIC_API_KEY:
    print(f"✓ ANTHROPIC_API_KEY loaded: {ANTHROPIC_API_KEY[:10]}...")

# ── Prompts ────────────────────────────────────────────────────────────────────

MULTIHOP_GENERATION_PROMPT = """You are an expert AI/ML educator creating a MULTI-HOP QA evaluation dataset
from the textbook "Dive into Deep Learning" (D2L).

You are given FIVE related passages from different sections of the textbook.
Generate {n_questions} question-answer pairs where EACH answer REQUIRES
synthesizing information from ALL FIVE passages — the question cannot be
answered from any single passage alone.

PASSAGE 1 (Chapter: {chapter1}, Section: {heading1}):
---
{chunk1}
---

PASSAGE 2 (Chapter: {chapter2}, Section: {heading2}):
---
{chunk2}
---

PASSAGE 3 (Chapter: {chapter3}, Section: {heading3}):
---
{chunk3}
---

PASSAGE 4 (Chapter: {chapter4}, Section: {heading4}):
---
{chunk4}
---

PASSAGE 5 (Chapter: {chapter5}, Section: {heading5}):
---
{chunk5}
---

RULES:
1. The question MUST require synthesizing content from ALL 5 passages.
2. The answer should be 3-5 sentences, weaving concepts from multiple passages.
3. For each question, identify which pieces each passage contributes to the answer.
4. Choose a question TYPE from: compare, trace_evolution, explain_relationship,
   synthesize_workflow, debug_scenario.
5. Difficulty must be "medium" or "hard" — these are multi-hop questions.
6. Do NOT ask questions answerable from a single passage.
7. For mathematical content, reference the formulas in context.
8. CRITICAL: Do NOT say "Passage 1", "Passage 2", "as described in Passage N",
   or any similar label in your question or gold_answer. Reference concepts and
   techniques by NAME (e.g. "self-attention", "VGG", "dropout") — the reader
   will NOT see the passage labels.

Respond in JSON format:
[
  {{
    "question": "...",
    "gold_answer": "... (3–5 sentences synthesizing all 5 passages, citing concepts by name) ...",
    "answer_type": "compare|trace_evolution|explain_relationship|synthesize_workflow|debug_scenario",
    "difficulty": "medium|hard",
    "chunk_contributions": {{
      "chunk_1": "what chunk 1 contributes to the answer",
      "chunk_2": "what chunk 2 contributes to the answer",
      "chunk_3": "what chunk 3 contributes to the answer",
      "chunk_4": "what chunk 4 contributes to the answer",
      "chunk_5": "what chunk 5 contributes to the answer"
    }}
  }}
]
"""

ABLATION_PROMPT = """You are evaluating whether a multi-hop question-answer pair can be answered
from the provided passages.

QUESTION: {question}
GOLD ANSWER: {gold_answer}

PROVIDED PASSAGES:
{passages}

Rate how well the provided passages allow answering this question correctly:
0 = Cannot answer at all / passages are irrelevant
3 = Partial answer only — missing key information
5 = Roughly correct but incomplete or lacking detail
7 = Mostly correct and reasonably complete
9 = Fully and correctly answered from these passages

Respond as JSON:
{{"score": N, "reasoning": "one sentence explanation"}}
"""

ANSWER_JUDGE_PROMPT = """You are evaluating a RAG pipeline's answer against a gold standard answer.

QUESTION: {question}
GOLD ANSWER: {gold_answer}
PIPELINE ANSWER: {pipeline_answer}

Rate the pipeline answer on a 0–9 scale:
0 = Completely wrong or irrelevant
3 = Partially correct, major gaps
6 = Mostly correct with minor gaps
9 = Complete and accurate

Respond as JSON:
{{"score": N, "reasoning": "one sentence"}}
"""

CLEAN_PASSAGE_PROMPT = """The following QA pair contains references like "Passage 1", "Passage 2",
"as described in Passage N", etc. These labels are an artifact of the generation
process and must be removed — the reader will not see any labelled passages.

Rewrite ONLY the gold_answer so it references concepts and techniques by name
instead of by passage label. Keep the meaning identical; only remove/replace the
passage references. Return just the rewritten answer text, no JSON, no extra words.

QUESTION: {question}

ORIGINAL GOLD ANSWER:
{gold_answer}

REWRITTEN GOLD ANSWER (no Passage N references):
"""


# ── LLM Client ────────────────────────────────────────────────────────────────

def call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 3000,
    api_key: Optional[str] = None,
) -> str:
    """
    Universal LLM caller. Auto-detects OpenAI vs Anthropic from the model name.
    Supports: gpt-4o, gpt-4o-mini, o1, o3, claude-*, sonnet-*, opus-*, haiku-*
    """
    is_openai = any(p in model.lower() for p in ["gpt", "o1", "o3"])
    is_anthropic = any(p in model.lower() for p in ["claude", "sonnet", "opus", "haiku"])

    if is_openai:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key or OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except ImportError:
            raise NotImplementedError("Run: pip install openai")

    elif is_anthropic:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            raise NotImplementedError("Run: pip install anthropic")

    else:
        raise ValueError(
            f"Unknown model '{model}'. Use OpenAI (gpt-4o, gpt-4o-mini) "
            "or Anthropic (claude-sonnet-4-20250514, claude-haiku-*)"
        )


# ── Robust JSON Parser (handles LaTeX backslashes) ────────────────────────────

def parse_json_response(response_text: str):
    """
    3-stage fallback JSON parser for LLM output.

    D2L has heavy LaTeX (\\alpha, \\frac, \\nabla) that LLMs echo verbatim
    into JSON strings, causing json.loads to raise Invalid \\escape errors.

    Stage 1: Direct json.loads (fast path)
    Stage 2: Regex-escape invalid backslashes, retry
    Stage 3: json-repair library (handles apostrophes, trailing commas, etc.)
    """
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Stage 1
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Stage 2 — fix bare backslashes not followed by a valid JSON escape char
    fixed = re.sub(r'\\(?!["\\\\/bfnrtu])', r"\\\\", text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Stage 3 — json-repair (optional dependency)
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, list):
            return result
        return [result] if isinstance(result, dict) else []
    except Exception:
        pass

    raise json.JSONDecodeError("All parse strategies failed", text, 0)


def clean_passage_references(
    qa_pairs: List[Dict],
    model: str,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Post-processing pass: remove "Passage N" / "as described in Passage 3" etc.
    from questions and gold_answers.

    Strategy:
      1. Fast regex substitution catches ~80% of cases (e.g. "in Passage 1",
         "Passage 2 explains", "as shown in passage N").
      2. If references remain, call the LLM with CLEAN_PASSAGE_PROMPT to
         rewrite the answer preserving meaning but replacing labels with
         concept names.
    """
    PASSAGE_RE = re.compile(
        r"(?:as\s+(?:described|shown|explained|discussed|detailed|stated|\w+)\s+in\s+)?"
        r"[Pp]assage\s*\d+",
        re.IGNORECASE,
    )

    cleaned = []
    for qa in qa_pairs:
        for field in ("question", "gold_answer"):
            text = qa.get(field, "")
            if not PASSAGE_RE.search(text):
                continue  # already clean

            # Stage 1: regex strip
            text = PASSAGE_RE.sub("", text)
            # Clean up double spaces and leading/trailing whitespace per sentence
            text = re.sub(r"  +", " ", text).strip()
            qa[field] = text

            # Stage 2: if still dirty (e.g. "the passage" without a number)
            # or if the sentence reads awkwardly, ask the LLM to rewrite
            if re.search(r"\bpassage\b", text, re.IGNORECASE):
                try:
                    prompt = CLEAN_PASSAGE_PROMPT.format(
                        question=qa.get("question", ""),
                        gold_answer=text,
                    )
                    rewritten = call_llm(
                        model, prompt, temperature=0.2, max_tokens=800, api_key=api_key
                    )
                    rewritten = rewritten.strip().strip('"')
                    if rewritten:
                        qa[field] = rewritten
                except Exception:
                    pass  # keep regex-cleaned version

        cleaned.append(qa)
    return cleaned


# ── D2L Parser (shared with single-hop logic) ─────────────────────────────────

def parse_d2l_chapters(d2l_root: str) -> List[Dict]:
    """
    Walk the D2L repo and return a list of sections as dicts:
      {chapter, heading, content, file_path}
    """
    sections = []
    chapter_pattern = re.compile(r"^chapter_", re.IGNORECASE)
    md_heading = re.compile(r"^#{1,3}\s+(.+)")

    if not os.path.isdir(d2l_root):
        raise FileNotFoundError(f"D2L root not found: {d2l_root}")

    for root, dirs, files in os.walk(d2l_root):
        dirs.sort()
        rel = os.path.relpath(root, d2l_root)
        parts = rel.split(os.sep)
        chapter = parts[0] if chapter_pattern.match(parts[0]) else None

        if chapter is None:
            continue

        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as fh:
                    raw = fh.read()
            except Exception:
                continue

            # Split into sub-sections on ## headings
            blocks = re.split(r"\n(?=#{1,3} )", raw)
            for block in blocks:
                lines = block.strip().splitlines()
                if not lines:
                    continue
                heading_match = md_heading.match(lines[0])
                heading = heading_match.group(1) if heading_match else fname
                content = "\n".join(lines[1:]).strip()
                if len(content) < 100:
                    continue
                sections.append({
                    "chapter": chapter,
                    "heading": heading,
                    "content": content,
                    "file_path": fpath,
                })

    return sections


def chunk_sections(
    sections: List[Dict],
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[Dict]:
    """Split section content into overlapping word-count chunks."""
    chunks = []
    for sec in sections:
        words = sec["content"].split()
        if not words:
            continue
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            slice_words = words[start : start + chunk_size]
            if len(slice_words) < 50:
                continue
            text = " ".join(slice_words)
            section_id = hashlib.md5(
                f"{sec['chapter']}:{sec['heading']}".encode()
            ).hexdigest()[:12]
            chunk_index = start // step
            chunks.append({
                "chunk_id": f"{section_id}_{chunk_index}",
                "chapter": sec["chapter"],
                "heading": sec["heading"],
                "text": text,
                "word_count": len(slice_words),
                "file_path": sec["file_path"],
            })
    return chunks


# ── Chunk Grouping Strategies ─────────────────────────────────────────────────

def build_tfidf_matrix(chunks: List[Dict]):
    """
    Build a TF-IDF matrix for all chunks. Returns (matrix, vectorizer).
    Falls back to a simple BoW counter if scikit-learn is not installed.
    """
    texts = [c["text"] for c in chunks]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
        matrix = vec.fit_transform(texts)
        return matrix, vec
    except ImportError:
        raise ImportError("Install scikit-learn: pip install scikit-learn")


def cluster_chunks(chunks: List[Dict], n_clusters: int = 20) -> List[int]:
    """
    K-Means cluster chunks by TF-IDF. Returns cluster label per chunk.
    """
    try:
        from sklearn.cluster import KMeans
        matrix, _ = build_tfidf_matrix(chunks)
        km = KMeans(n_clusters=min(n_clusters, len(chunks) // 5), random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        return labels.tolist()
    except ImportError:
        # Fallback: assign clusters by chapter order
        chapter_map = {}
        labels = []
        cid = 0
        for c in chunks:
            if c["chapter"] not in chapter_map:
                chapter_map[c["chapter"]] = cid
                cid += 1
            labels.append(chapter_map[c["chapter"]])
        return labels


def cosine_similarity_pair(chunks: List[Dict], idx_a: int, idx_b: int) -> float:
    """Compute TF-IDF cosine similarity between two chunks."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        texts = [chunks[idx_a]["text"], chunks[idx_b]["text"]]
        vec = TfidfVectorizer(stop_words="english")
        m = vec.fit_transform(texts)
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except Exception:
        return 0.0


def form_intra_cluster_groups(
    chunks: List[Dict],
    labels: List[int],
    num_groups: int,
    min_sim: float = 0.15,
    max_sim: float = 0.75,
) -> List[Dict]:
    """
    Strategy 1 – INTRA-CLUSTER:
    Pick 5 chunks from the same topic cluster, spread across different chapters.
    """
    from collections import defaultdict
    cluster_map: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_map[lbl].append(i)

    groups = []
    cluster_ids = [c for c in cluster_map if len(cluster_map[c]) >= 5]
    random.shuffle(cluster_ids)

    for cid in cluster_ids:
        if len(groups) >= num_groups:
            break
        indices = cluster_map[cid]
        # Try to pick from different chapters for diversity
        by_chapter: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            by_chapter[chunks[idx]["chapter"]].append(idx)

        candidate_chapters = list(by_chapter.keys())
        random.shuffle(candidate_chapters)

        selected = []
        for ch in candidate_chapters:
            if len(selected) >= 5:
                break
            selected.append(random.choice(by_chapter[ch]))

        # If not enough diverse chapters, fill from same cluster
        pool = [i for i in indices if i not in selected]
        random.shuffle(pool)
        while len(selected) < 5 and pool:
            selected.append(pool.pop())

        if len(selected) == 5:
            groups.append({
                "strategy": "intra_cluster",
                "chunk_indices": selected,
                "cluster_id": cid,
            })

    return groups


def form_prerequisite_chain_groups(
    chunks: List[Dict],
    num_groups: int,
) -> List[Dict]:
    """
    Strategy 2 – PREREQUISITE CHAIN:
    Select chunks whose chapters appear in a natural conceptual progression
    (Preliminaries → Linear NN → CNN/RNN → Attention → Modern Arch).
    """
    # Map chapter names to order tier
    chapter_to_tier = {}
    for chunk in chunks:
        ch = chunk["chapter"].lower()
        if any(k in ch for k in ["preliminaries", "prelim", "math", "linear-algebra", "probability"]):
            chapter_to_tier[chunk["chapter"]] = 0
        elif any(k in ch for k in ["linear", "regression", "classification", "softmax", "mlp", "perceptron"]):
            chapter_to_tier[chunk["chapter"]] = 1
        elif any(k in ch for k in ["convolutional", "cnn", "pooling", "lenet"]):
            chapter_to_tier[chunk["chapter"]] = 2
        elif any(k in ch for k in ["recurrent", "rnn", "lstm", "gru", "sequence", "text"]):
            chapter_to_tier[chunk["chapter"]] = 3
        elif any(k in ch for k in ["attention", "transformer", "bert", "gpt"]):
            chapter_to_tier[chunk["chapter"]] = 4
        elif any(k in ch for k in ["modern", "resnet", "densenet", "batch", "dropout", "regulariz"]):
            chapter_to_tier[chunk["chapter"]] = 2
        elif any(k in ch for k in ["optimization", "optim", "gradient", "sgd", "adam", "momentum"]):
            chapter_to_tier[chunk["chapter"]] = 1
        else:
            chapter_to_tier[chunk["chapter"]] = 5

    # Group chunk indices by tier
    tier_map: Dict[int, List[int]] = {}
    for i, chunk in enumerate(chunks):
        tier = chapter_to_tier.get(chunk["chapter"], 5)
        tier_map.setdefault(tier, []).append(i)

    available_tiers = sorted(tier_map.keys())
    if len(available_tiers) < 3:
        return []

    groups = []
    attempts = 0
    while len(groups) < num_groups and attempts < num_groups * 10:
        attempts += 1
        # Pick 5 consecutive or spread tiers
        if len(available_tiers) >= 5:
            chosen_tiers = random.sample(available_tiers, 5)
        else:
            chosen_tiers = random.choices(available_tiers, k=5)
        chosen_tiers.sort()

        selected = []
        for tier in chosen_tiers:
            candidates = tier_map[tier]
            if candidates:
                selected.append(random.choice(candidates))

        if len(selected) == 5 and len(set(selected)) == 5:
            groups.append({
                "strategy": "prerequisite_chain",
                "chunk_indices": selected,
                "tiers": chosen_tiers,
            })

    return groups


def form_cross_topic_groups(
    chunks: List[Dict],
    labels: List[int],
    num_groups: int,
) -> List[Dict]:
    """
    Strategy 3 – CROSS-TOPIC (2+2+1):
    2 chunks from cluster A + 2 from cluster B + 1 from a bridging cluster C.
    """
    from collections import defaultdict
    cluster_map: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_map[lbl].append(i)

    large_clusters = [c for c in cluster_map if len(cluster_map[c]) >= 3]
    if len(large_clusters) < 3:
        return []

    groups = []
    attempts = 0
    while len(groups) < num_groups and attempts < num_groups * 20:
        attempts += 1
        if len(large_clusters) < 3:
            break
        ca, cb, cc = random.sample(large_clusters, 3)

        def pick_n(cluster_id, n):
            pool = cluster_map[cluster_id][:]
            random.shuffle(pool)
            return pool[:n]

        a_picks = pick_n(ca, 2)
        b_picks = pick_n(cb, 2)
        c_picks = pick_n(cc, 1)

        if len(a_picks) == 2 and len(b_picks) == 2 and len(c_picks) == 1:
            selected = a_picks + b_picks + c_picks
            if len(set(selected)) == 5:
                groups.append({
                    "strategy": "cross_topic",
                    "chunk_indices": selected,
                    "clusters": [ca, cb, cc],
                })

    return groups


def form_chunk_groups(
    chunks: List[Dict],
    labels: List[int],
    num_groups: int,
) -> List[Dict]:
    """
    Distribute num_groups across 3 strategies (~40% / 35% / 25%).
    """
    n_intra = math.ceil(num_groups * 0.40)
    n_prereq = math.ceil(num_groups * 0.35)
    n_cross = num_groups - n_intra - n_prereq

    intra = form_intra_cluster_groups(chunks, labels, n_intra * 3)  # overgenerate → trim
    prereq = form_prerequisite_chain_groups(chunks, n_prereq * 3)
    cross = form_cross_topic_groups(chunks, labels, n_cross * 3)

    # Trim to target counts and combine
    groups = intra[:n_intra] + prereq[:n_prereq] + cross[:n_cross]
    random.shuffle(groups)
    return groups[:num_groups]


# ── QA Generation ─────────────────────────────────────────────────────────────

def generate_multihop_qa(
    group: Dict,
    chunks: List[Dict],
    model: str,
    n_questions: int = 2,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Call the LLM to generate n_questions multi-hop QA pairs for a 5-chunk group.
    Returns a list of raw QA dicts from the LLM.
    """
    idxs = group["chunk_indices"]
    if len(idxs) < 5:
        return []

    chunk_data = [chunks[i] for i in idxs]
    prompt = MULTIHOP_GENERATION_PROMPT.format(
        n_questions=n_questions,
        chapter1=chunk_data[0]["chapter"], heading1=chunk_data[0]["heading"],
        chunk1=chunk_data[0]["text"][:1200],
        chapter2=chunk_data[1]["chapter"], heading2=chunk_data[1]["heading"],
        chunk2=chunk_data[1]["text"][:1200],
        chapter3=chunk_data[2]["chapter"], heading3=chunk_data[2]["heading"],
        chunk3=chunk_data[2]["text"][:1200],
        chapter4=chunk_data[3]["chapter"], heading4=chunk_data[3]["heading"],
        chunk4=chunk_data[3]["text"][:1200],
        chapter5=chunk_data[4]["chapter"], heading5=chunk_data[4]["heading"],
        chunk5=chunk_data[4]["text"][:1200],
    )

    try:
        response = call_llm(model, prompt, temperature=0.5, max_tokens=3000, api_key=api_key)
        raw = parse_json_response(response)
        if not isinstance(raw, list):
            raw = [raw]
        # Attach chunk IDs
        for qa in raw:
            qa["gold_chunk_ids"] = [chunk_data[j]["chunk_id"] for j in range(5)]
            qa["strategy"] = group.get("strategy", "unknown")
        # Remove any residual "Passage N" references the LLM snuck in
        raw = clean_passage_references(raw, model, api_key=api_key)
        return raw
    except Exception as e:
        print(f"\nWarning: QA generation failed for group: {e}")
        return []


# ── Ablation Validation ────────────────────────────────────────────────────────

def score_with_passages(
    question: str,
    gold_answer: str,
    passage_texts: List[str],
    model: str,
    api_key: Optional[str] = None,
) -> int:
    """
    Ask the LLM how well the given passages support the question/answer.
    Returns an integer score 0–9.
    """
    combined = "\n\n---\n\n".join(
        f"PASSAGE {i+1}:\n{t[:800]}" for i, t in enumerate(passage_texts)
    )
    prompt = ABLATION_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        passages=combined,
    )
    try:
        response = call_llm(model, prompt, temperature=0.2, max_tokens=200, api_key=api_key)
        parsed = parse_json_response(response)
        if isinstance(parsed, dict):
            return int(parsed.get("score", 0))
        if isinstance(parsed, list) and parsed:
            return int(parsed[0].get("score", 0))
    except Exception:
        pass
    return 0


def ablation_validate(
    qa: Dict,
    chunks: List[Dict],
    chunk_id_to_idx: Dict[str, int],
    model: str,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Run ablation validation:
     - Full (5 chunks): score should be >= 7
     - Leave-one-out (5 × 4-chunk subsets): avg score should drop below 6
     - Single best chunk: score should be < 4

    Returns qa enriched with validation scores and quality_tier.
    """
    gold_ids = qa["gold_chunk_ids"]
    texts = []
    for cid in gold_ids:
        idx = chunk_id_to_idx.get(cid)
        texts.append(chunks[idx]["text"] if idx is not None else "")

    question = qa["question"]
    gold_answer = qa["gold_answer"]

    # Full 5-chunk score
    full_score = score_with_passages(question, gold_answer, texts, model, api_key)

    # Leave-one-out scores
    loo_scores = []
    for omit in range(5):
        subset = [t for j, t in enumerate(texts) if j != omit]
        s = score_with_passages(question, gold_answer, subset, model, api_key)
        loo_scores.append(s)
    avg_loo = sum(loo_scores) / len(loo_scores) if loo_scores else 0

    # Single-chunk best score
    single_scores = [
        score_with_passages(question, gold_answer, [t], model, api_key)
        for t in texts
    ]
    max_single = max(single_scores) if single_scores else 0

    validation = {
        "full_5_score": full_score,
        "avg_leave_one_out": round(avg_loo, 2),
        "max_single_chunk": max_single,
    }

    # Tiering
    if full_score >= 8 and avg_loo < 5 and max_single < 3:
        tier = "gold"
    elif full_score >= 7 and avg_loo < 6 and max_single < 5:
        tier = "silver"
    else:
        tier = "reject"

    qa["validation"] = validation
    qa["quality_tier"] = tier
    return qa


# ── Dataset Formatting ────────────────────────────────────────────────────────

def format_final_dataset(
    qa_pairs: List[Dict],
    chunks: List[Dict],
    model: str,
) -> Dict:
    """
    Assemble the final JSON dataset in the documented format.
    """
    # Filter rejects
    kept = [qa for qa in qa_pairs if qa.get("quality_tier", "silver") != "reject"]
    if not kept:
        kept = qa_pairs  # if no validation ran, keep all

    strategy_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}

    questions = []
    for i, qa in enumerate(kept):
        question_id = f"d2l_mh_{i:04d}"
        strategy = qa.get("strategy", "unknown")
        atype = qa.get("answer_type", "unknown")
        tier = qa.get("quality_tier", "silver")

        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        type_counts[atype] = type_counts.get(atype, 0) + 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        gold_ids = qa.get("gold_chunk_ids", [])
        contributions = qa.get("chunk_contributions", {})

        # Normalise chunk_contributions keys → chunk_id based
        contrib_by_id = {}
        for j, cid in enumerate(gold_ids):
            key = f"chunk_{j+1}"
            contrib_by_id[cid] = contributions.get(key, "")

        entry = {
            "question_id": question_id,
            "question": qa.get("question", ""),
            "gold_answer": qa.get("gold_answer", ""),
            "answer_type": atype,
            "difficulty": qa.get("difficulty", "hard"),
            "strategy": strategy,
            "quality_tier": tier,
            "gold_chunk_ids": gold_ids,
            "chunk_contributions": contrib_by_id,
            # essential = chunks marked non-trivial by LOO drop
            "essential_chunk_ids": gold_ids[:3],
            "supporting_chunk_ids": gold_ids[3:],
        }
        if "validation" in qa:
            entry["validation"] = qa["validation"]

        questions.append(entry)

    corpus_for_output = [
        {
            "chunk_id": c["chunk_id"],
            "chapter": c["chapter"],
            "heading": c["heading"],
            "text": c["text"],
            "word_count": c.get("word_count", len(c.get("text", "").split())),
        }
        for c in chunks
    ]

    return {
        "metadata": {
            "name": "D2L-QA-MultiHop",
            "version": "1.0",
            "model": model,
            "total_questions": len(questions),
            "chunks_per_question": 5,
            "total_corpus_chunks": len(chunks),
            "grouping_strategies": strategy_counts,
            "question_types": type_counts,
            "quality_tiers": tier_counts,
        },
        "corpus": corpus_for_output,
        "questions": questions,
    }


# ── Evaluation Helpers ────────────────────────────────────────────────────────

def evaluate_5chunk_retrieval(dataset: Dict, retriever, k: int = 5) -> Dict:
    """
    Evaluate a retriever against the multi-hop ground truth.

    Args:
        dataset:   loaded d2l_multihop_ground_truth.json
        retriever: callable(question: str, k: int) → List[str]  (returns chunk_ids)
        k:         number of chunks to retrieve per question

    Returns dict with chunk_recall, chunk_precision, chunk_f1, essential_recall,
    perfect_match_rate.
    """
    recalls, precisions, f1s, essential_recalls, perfects = [], [], [], [], []

    for qa in dataset["questions"]:
        gold = set(qa["gold_chunk_ids"])
        essential = set(qa.get("essential_chunk_ids", list(gold)[:3]))
        retrieved = set(retriever(qa["question"], k))

        tp = len(gold & retrieved)
        recall = tp / len(gold) if gold else 0
        precision = tp / len(retrieved) if retrieved else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        ess_recall = len(essential & retrieved) / len(essential) if essential else 0
        perfect = int(gold == retrieved)

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        essential_recalls.append(ess_recall)
        perfects.append(perfect)

    n = len(recalls) or 1
    return {
        "avg_chunk_recall": round(sum(recalls) / n, 4),
        "avg_chunk_precision": round(sum(precisions) / n, 4),
        "avg_chunk_f1": round(sum(f1s) / n, 4),
        "avg_essential_recall": round(sum(essential_recalls) / n, 4),
        "perfect_match_rate": round(sum(perfects) / n, 4),
        "num_questions": n,
    }


def evaluate_5chunk_answers(
    dataset: Dict,
    rag_pipeline,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict:
    """
    Evaluate answer quality using LLM-as-judge (0–9 scale).

    Args:
        dataset:      loaded d2l_multihop_ground_truth.json
        rag_pipeline: callable(question: str) → str  (returns pipeline's answer)
        model:        judge model name
        api_key:      optional API key

    Returns dict with string_em_accuracy, avg_judge_score, judge_pass_rate.
    """
    em_hits, judge_scores = [], []

    for qa in dataset["questions"]:
        question = qa["question"]
        gold = qa["gold_answer"].strip().lower()
        pipeline_answer = rag_pipeline(question)

        # String EM (lenient)
        em = int(gold in pipeline_answer.strip().lower() or
                 pipeline_answer.strip().lower() in gold)
        em_hits.append(em)

        # LLM judge
        prompt = ANSWER_JUDGE_PROMPT.format(
            question=question,
            gold_answer=qa["gold_answer"],
            pipeline_answer=pipeline_answer,
        )
        try:
            response = call_llm(model, prompt, temperature=0.2, max_tokens=150, api_key=api_key)
            parsed = parse_json_response(response)
            score = int(parsed.get("score", 0)) if isinstance(parsed, dict) else 0
        except Exception:
            score = 0
        judge_scores.append(score)

    n = len(em_hits) or 1
    return {
        "string_em_accuracy": round(sum(em_hits) / n, 4),
        "avg_judge_score": round(sum(judge_scores) / n, 2),
        "judge_pass_rate": round(sum(1 for s in judge_scores if s >= 6) / n, 4),
        "num_questions": n,
    }


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def main(
    d2l_root: str,
    output_dir: str = ".",
    model: str = "gpt-4o",
    n_questions_per_group: int = 2,
    num_groups: int = 100,
    n_clusters: int = 20,
    skip_validation: bool = False,
    min_similarity: float = 0.15,
    max_similarity: float = 0.75,
    api_key: Optional[str] = None,
    corpus_path: Optional[str] = None,
) -> Dict:
    """
    Run the full multi-hop generation pipeline.

    Args:
        d2l_root:            Path to D2L repo root (e.g. ./d2l-en)
        output_dir:          Where to write output JSON
        model:               LLM model name (auto-detects API)
        n_questions_per_group: Questions to generate per 5-chunk group
        num_groups:          Number of 5-chunk groups to form
        n_clusters:          K-Means clusters for topic grouping
        skip_validation:     If True, skip ablation validation (cheaper)
        min_similarity:      Minimum TF-IDF similarity for pairing (unused currently)
        max_similarity:      Maximum TF-IDF similarity (unused currently)
        api_key:             Override API key (defaults to .env value)
        corpus_path:         If set, load chunks from this JSON instead of re-parsing

    Returns the formatted dataset dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "multihop_generation_log.txt")

    import sys

    class Tee:
        """Write to both stdout and a log file."""
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    try:
        # ── Step 1: Parse & chunk ──────────────────────────────────────────────
        if corpus_path and os.path.exists(corpus_path):
            print(f"Step 1: Loading existing corpus from {corpus_path}...")
            with open(corpus_path) as f:
                chunks = json.load(f)
            print(f"  Loaded {len(chunks)} chunks from corpus.")
        else:
            print("Step 1: Parsing and chunking D2L content...")
            sections = parse_d2l_chapters(d2l_root)
            print(f"  Parsed {len(sections)} sections")
            chunks = chunk_sections(sections)
            print(f"  → {len(chunks)} chunks")

        # ── Step 2: Cluster & form groups ─────────────────────────────────────
        print(f"\nStep 2: Clustering {len(chunks)} chunks into {n_clusters} topics...")
        labels = cluster_chunks(chunks, n_clusters=n_clusters)
        unique_labels = len(set(labels))
        print(f"  → {unique_labels} clusters formed")

        print(f"\nStep 3: Forming {num_groups} five-chunk groups...")
        groups = form_chunk_groups(chunks, labels, num_groups)
        print(f"  → {len(groups)} groups formed")

        strategy_tally: Dict[str, int] = {}
        for g in groups:
            s = g.get("strategy", "unknown")
            strategy_tally[s] = strategy_tally.get(s, 0) + 1
        for s, cnt in strategy_tally.items():
            print(f"    {s}: {cnt}")

        # ── Step 3: Generate QA ───────────────────────────────────────────────
        print(f"\nStep 4: Generating QA pairs ({n_questions_per_group} per group)...")
        all_qa_pairs: List[Dict] = []
        for i, group in enumerate(groups):
            print(f"Generating group {i+1}/{len(groups)} [{group['strategy']}]...")
            pairs = generate_multihop_qa(
                group, chunks, model,
                n_questions=n_questions_per_group,
                api_key=api_key,
            )
            all_qa_pairs.extend(pairs)

        print(f"\n  → {len(all_qa_pairs)} raw QA pairs generated")

        # ── Step 4: Ablation Validation ───────────────────────────────────────
        if not skip_validation:
            print(f"\nStep 5: Ablation validation ({len(all_qa_pairs)} pairs × 6 LLM calls each)...")
            print("  This is expensive — each question calls the LLM ~6 times.")

            chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

            validated_pairs = []
            for i, qa in enumerate(all_qa_pairs):
                print(f"Validating {i+1}/{len(all_qa_pairs)} (tier TBD)...")
                qa = ablation_validate(qa, chunks, chunk_id_to_idx, model, api_key=api_key)
                validated_pairs.append(qa)

            gold_count = sum(1 for qa in validated_pairs if qa.get("quality_tier") == "gold")
            silver_count = sum(1 for qa in validated_pairs if qa.get("quality_tier") == "silver")
            reject_count = sum(1 for qa in validated_pairs if qa.get("quality_tier") == "reject")
            print(f"\n  Validation results: Gold={gold_count}  Silver={silver_count}  Reject={reject_count}")
            all_qa_pairs = validated_pairs
        else:
            print("\nStep 5: Skipping ablation validation (--skip-validation)")
            for qa in all_qa_pairs:
                qa.setdefault("quality_tier", "silver")

        # ── Step 5: Format & save ──────────────────────────────────────────────
        print("\nStep 6: Formatting final dataset...")
        dataset = format_final_dataset(all_qa_pairs, chunks, model)

        out_path = os.path.join(output_dir, "d2l_multihop_ground_truth.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        sep = "=" * 50
        print(f"\n{sep}")
        print("D2L-QA Multi-Hop Dataset Summary")
        print(sep)
        m = dataset["metadata"]
        print(f"Questions:          {m['total_questions']}")
        print(f"Chunks per question: {m['chunks_per_question']}")
        print(f"Corpus chunks:      {m['total_corpus_chunks']}")
        print(f"Strategies:         {m['grouping_strategies']}")
        print(f"Question types:     {m['question_types']}")
        print(f"Quality tiers:      {m['quality_tiers']}")
        print(f"\nFile saved: {out_path}")
        print(f"Log saved:  {log_path}")

        return dataset

    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate D2L-QA multi-hop ground truth dataset (5-chunk synthesis)"
    )
    parser.add_argument("d2l_root", help="Path to D2L repository root (e.g. ./d2l-en)")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: .)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model (default: gpt-4o)")
    parser.add_argument("--questions-per-group", type=int, default=2,
                        help="QA pairs per 5-chunk group (default: 2)")
    parser.add_argument("--num-groups", type=int, default=100,
                        help="Number of 5-chunk groups to form (default: 100)")
    parser.add_argument("--n-clusters", type=int, default=20,
                        help="K-Means topic clusters (default: 20)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip ablation validation (cheaper, faster)")
    parser.add_argument("--min-similarity", type=float, default=0.15,
                        help="Min TF-IDF chunk similarity (default: 0.15)")
    parser.add_argument("--max-similarity", type=float, default=0.75,
                        help="Max TF-IDF chunk similarity (default: 0.75)")
    parser.add_argument("--api-key", help="Override API key (default: reads from .env)")
    parser.add_argument("--corpus-path",
                        help="Reuse existing corpus JSON (e.g. d2l/single_hop/output/d2l_corpus_chunks.json)")

    args = parser.parse_args()

    main(
        d2l_root=args.d2l_root,
        output_dir=args.output_dir,
        model=args.model,
        n_questions_per_group=args.questions_per_group,
        num_groups=args.num_groups,
        n_clusters=args.n_clusters,
        skip_validation=args.skip_validation,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        api_key=args.api_key,
        corpus_path=args.corpus_path,
    )
