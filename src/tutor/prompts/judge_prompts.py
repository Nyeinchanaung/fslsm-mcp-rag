"""LLM-as-a-Judge prompt templates for Experiment 2 evaluation."""

RR_JUDGE_PROMPT = """You are an expert evaluator assessing the relevance of an AI tutor's response.

## Gold Answer (Ground Truth):
{gold_answer}

## Tutor Response:
{response}

## Scoring Rubric:
1 = Irrelevant: Response does not address the question or contains major factual errors
2 = Partially Relevant: Addresses the topic but misses key concepts from the gold answer
3 = Moderately Relevant: Covers main concepts but lacks depth or has minor inaccuracies
4 = Relevant: Accurately covers the key concepts with good depth
5 = Highly Relevant: Comprehensive, accurate, and well-structured — matches or exceeds gold answer

Rate the tutor response. Respond with ONLY a single integer (1-5).
"""
