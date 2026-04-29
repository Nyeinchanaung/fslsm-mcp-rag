"""LLM-as-a-Judge prompt templates for Experiment 2 evaluation."""

RR_JUDGE_SYSTEM = (
    "You are an impartial educational evaluation judge. Your task is to assess "
    "the factual accuracy and pedagogical relevance of an AI tutor's response "
    "to a student's question about Introductory Machine Learning."
)

RR_JUDGE_PROMPT = """## Evaluation Task

Compare the tutor's response against the ground truth and assess its quality.

---

## [Student Query]
{student_query}

---

## [Ground Truth Reference Answer]
{gold_answer}

---

## [Retrieved Source Chunks (Evidence Available to Tutor)]
{source_chunks}

---

## [Tutor Response Under Evaluation]
{response}

---

## Scoring Rubric

Rate the tutor response on the following scale:

- **1 — Irrelevant:** Does not address the query, or contains significant factual errors
- **2 — Partially Relevant:** Addresses the topic superficially; notable omissions or inaccuracies
- **3 — Moderately Relevant:** Mostly accurate but lacks depth or completeness
- **4 — Relevant:** Accurate, complete, addresses the query well with only minor omissions
- **5 — Highly Relevant & Accurate:** Comprehensive, factually correct, well-structured, and pedagogically effective

## Evaluation Notes

- Focus on **factual accuracy and completeness** relative to the gold answer.
- **Do not penalise** stylistic adaptations (e.g., numbered steps, diagram references, analogy-based explanations) provided the factual content is complete and correct. These are valid pedagogical choices.
- **Do penalise** responses that omit key concepts from the gold answer or introduce claims not supported by the source chunks.
- Numbered steps, ASCII diagrams, tables, bullet points, and analogy-based explanations are valid pedagogical formats. A response using these structures must be evaluated on whether the factual content is present, not on whether it resembles the prose style of the gold answer.

Begin your evaluation. Then output your rating on the final line in this exact format:

Response Relevance Rating: [[score]]
"""
