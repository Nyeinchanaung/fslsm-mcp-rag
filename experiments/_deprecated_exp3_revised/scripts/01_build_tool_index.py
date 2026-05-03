"""
Phase 2: Build FAISS IndexFlatIP over the 15 revised tool descriptions.
Requires OPENAI_API_KEY. Costs ~$0.001 (15 embeddings).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_revised.tools.tool_index import ToolIndex

if __name__ == "__main__":
    idx = ToolIndex()
    idx.build()
    idx.save()

    print("\n── Retrieval sanity check (top-1 per test query) ──")
    tests = [
        "Can you draw a computation graph for backpropagation?",
        "Walk me through gradient descent step by step.",
        "Compare ResNet and VGG architectures.",
        "Quiz me on batch normalization.",
        "Summarize the attention mechanism chapter.",
        "What are the latest transformer developments?",
        "Give me a hands-on coding exercise for implementing a CNN.",
        "Reflect on the role of regularization in deep learning.",
    ]
    for q in tests:
        hits = idx.retrieve(q, k=3)
        print(f"\n  Q: {q}")
        for i, (tool, score) in enumerate(hits):
            print(f"    {i+1}. [{tool.tool_id:>2}] {tool.name:<35} score={score:.3f}")

    print("\n✅ Tool index built. Verify scores are >0.50 (was 0.12–0.32 before).")
