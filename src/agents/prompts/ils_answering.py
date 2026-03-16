"""Prompt builder for ILS questionnaire questions."""
from __future__ import annotations


def build_ils_question_prompt(question: dict) -> str:
    """
    Format a single ILS question as a user prompt for an LLM agent.

    Args:
        question: Dict with keys q_num, text, option_a, option_b
                  (from ils_questionnaire.json).
    """
    return (
        "You are taking a learning style survey. Answer the following question "
        "honestly based on your learning preferences as described in your profile.\n\n"
        f"Question {question['q_num']}: {question['text']}\n\n"
        "Options:\n"
        f"  (a) {question['option_a']['text']}\n"
        f"  (b) {question['option_b']['text']}\n\n"
        'Respond with ONLY the letter "a" or "b". Do not explain your answer.'
    )
