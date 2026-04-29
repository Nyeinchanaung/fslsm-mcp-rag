"""LLM-as-a-Judge prompt templates for Experiment 2 evaluation."""

RR_JUDGE_SYSTEM = (
    "You are an impartial educational evaluation judge specialising in factual "
    "accuracy assessment. Your task is to determine whether an AI tutor's response "
    "covers the key factual content required to correctly answer a student's question "
    "about Introductory Machine Learning. "
    "You evaluate CONTENT COVERAGE, not writing style or presentation format."
)

RR_JUDGE_PROMPT = """## Evaluation Task

Determine whether the tutor's response covers the factual content in the ground truth.
The tutor may have adapted the presentation style for a specific learner profile —
this is intentional and must NOT affect your score.

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

## Your Evaluation Process

Follow these steps in order:

1. Extract the key factual claims from the Ground Truth Reference Answer.
2. Check whether each claim is present in the Tutor Response (it may be phrased
   differently, formatted as a list, shown in a table, or expressed as an analogy —
   all of these count as present if the underlying fact is conveyed).
3. Identify any key claims from the ground truth that are completely absent from
   the tutor response.
4. Assign a score based on coverage completeness (see rubric below).

---

## Scoring Rubric

- **1 — No coverage:** Response does not address the query or contains major factual errors
- **2 — Partial coverage:** Covers fewer than half the key factual claims from the ground truth
- **3 — Moderate coverage:** Covers most claims but with notable gaps or one significant inaccuracy
- **4 — Good coverage:** Covers all key claims with only minor omissions or imprecision
- **5 — Complete coverage:** All key factual claims present and accurate; no misleading content

## Critical Instructions

- A response that uses numbered steps, bullet points, ASCII diagrams, tables,
  analogies, or any other formatting is NOT penalised for that choice.
  Judge ONLY whether the underlying facts are present and accurate.
- The ground truth is written in plain prose. The tutor response may look very
  different in format. This is expected and correct — do not penalise it.
- If a response contains all the factual content but presents it visually or
  structurally differently from the gold answer, it scores 4 or 5.
- Only penalise for missing facts, factual errors, or content not supported
  by the source chunks.

Begin your evaluation by listing the key factual claims from the ground truth
(2–4 bullet points), then note which are present or absent in the tutor response.
Then output your rating on the final line in this exact format:

Response Relevance Rating: [[score]]
"""