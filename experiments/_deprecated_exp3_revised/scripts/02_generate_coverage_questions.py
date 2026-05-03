"""
Phase 5: Generate 16 coverage questions targeting tools 8–15.
Uses GPT-4.1-mini for generation, saves for manual review.
Cost: ~$0.02 total.

After running:
  1. Review data/coverage_questions.json
  2. Edit any unclear questions
  3. Set needs_review=false for each approved question
  4. Run scripts/merge_questions.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI
from experiments.exp3_revised.config import COVERAGE_Q_PATH

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
                "content": PROMPT.format(description=desc, question_type=qtype),
            }],
            max_tokens=150,
            temperature=0.7,
        )
        question_text = response.choices[0].message.content.strip()

        q = {
            "question_id":       f"COV_{i:03d}",
            "question":          question_text,
            "question_type":     qtype,
            "target_tools":      target_tools,
            "gold_chunk_ids":    [],
            "essential_chunk_ids": [],
            "strategy":          "coverage",
            "quality_tier":      "coverage",
            "needs_review":      True,
        }
        questions.append(q)
        print(f"[{q['question_id']}] type={qtype}")
        print(f"  Q: {question_text}")
        print()

    COVERAGE_Q_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COVERAGE_Q_PATH, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions to {COVERAGE_Q_PATH}")
    print("⚠ MANUAL REVIEW REQUIRED before proceeding to merge_questions.py")


if __name__ == "__main__":
    generate_coverage_questions()
