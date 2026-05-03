from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from experiments.exp3_mcp_runtime.core.profile_ground_truth import (
    build_profile_target_tool_ids,
    is_profile_eval_eligible,
)
from experiments.exp3_mcp_runtime.core.profile_sets import load_canonical_profiles
from experiments.exp3_mcp_runtime.core.retriever import D2LRetriever


@dataclass(frozen=True)
class CoreQuestionDraft:
    question_family: str
    question: str
    gold_answer_brief: str
    retrieval_query: str = ""
    source_text: str = ""
    target_evidence_criteria: list[str] | None = None
    profile_target_overrides: dict[str, int] | None = None


@dataclass(frozen=True)
class ToolSpec:
    tool_id: int
    tool_name: str
    grounding_mode: str
    target_behavior: str
    expected_question_patterns: list[str]
    disallowed_confounders: list[str]
    drafts: list[CoreQuestionDraft]


TOOL_SPEC_MATRIX: list[ToolSpec] = [
    ToolSpec(
        tool_id=1,
        tool_name="Concept Explainer",
        grounding_mode="d2l",
        target_behavior="Clear factual explanation in connected prose.",
        expected_question_patterns=["what is", "explain", "describe", "how does"],
        disallowed_confounders=["explicit comparison requests", "diagram requests", "quiz framing"],
        drafts=[
            CoreQuestionDraft("concept_explain", "Explain what gradient descent is and why the learning rate matters during training.", "Define gradient descent and connect the learning rate to step size, convergence, and instability.", retrieval_query="gradient descent learning rate training"),
            CoreQuestionDraft("concept_explain", "What does batch normalization do during neural network training?", "Explain normalization of minibatch activations and its effect on optimization stability.", retrieval_query="batch normalization training neural network"),
            CoreQuestionDraft("concept_explain", "Describe what word embeddings are and why they are useful in NLP models.", "Explain dense vector representations and how they capture semantic relationships.", retrieval_query="word embeddings NLP dense vector representations"),
            CoreQuestionDraft("concept_explain", "How does dropout help reduce overfitting in deep learning?", "Explain random unit dropping during training and the regularization effect.", retrieval_query="dropout overfitting regularization deep learning"),
        ],
    ),
    ToolSpec(
        tool_id=2,
        tool_name="Step-by-Step Derivator",
        grounding_mode="d2l",
        target_behavior="Ordered derivation or algorithm walkthrough.",
        expected_question_patterns=["step by step", "derive", "walk through", "trace"],
        disallowed_confounders=["pure summary requests", "open web requests"],
        drafts=[
            CoreQuestionDraft("derive_steps", "Walk through backpropagation for a single linear neuron step by step.", "Show forward pass, loss gradient, parameter gradients, and the update sequence.", retrieval_query="backpropagation linear neuron step by step"),
            CoreQuestionDraft("derive_steps", "Derive the output shape of a convolution layer step by step.", "Show how kernel size, padding, and stride determine the final shape.", retrieval_query="convolution output shape stride padding kernel"),
            CoreQuestionDraft("derive_steps", "Trace one minibatch SGD update from loss to parameter update.", "Show gradient computation and the final parameter update in sequence.", retrieval_query="minibatch stochastic gradient descent parameter update"),
            CoreQuestionDraft("derive_steps", "Step through how scaled dot-product attention computes attention weights.", "Show query-key scores, scaling, softmax, and weighted sum output.", retrieval_query="scaled dot product attention softmax weighted sum"),
        ],
    ),
    ToolSpec(
        tool_id=3,
        tool_name="Worked Example Generator",
        grounding_mode="d2l",
        target_behavior="Concrete solved example with intermediate calculations.",
        expected_question_patterns=["show me an example", "solve", "calculate", "worked example"],
        disallowed_confounders=["open-ended reflection", "chapter summary"],
        drafts=[
            CoreQuestionDraft("worked_example", "Give a worked example of computing cross-entropy loss from logits and a gold label.", "Present logits, probabilities, and the loss value with every intermediate step.", retrieval_query="cross entropy loss logits probabilities example"),
            CoreQuestionDraft("worked_example", "Show a worked example of one gradient descent weight update for linear regression.", "Use concrete numbers to compute prediction error, gradient, and updated weight.", retrieval_query="linear regression gradient descent weight update example"),
            CoreQuestionDraft("worked_example", "Provide a solved example of matrix multiplication in a neural network forward pass.", "Use explicit matrices and show the resulting activations.", retrieval_query="matrix multiplication neural network forward pass example"),
            CoreQuestionDraft("worked_example", "Work through a concrete example of computing the output size of a CNN feature map.", "Use specific input size, padding, stride, and kernel values.", retrieval_query="cnn feature map output size example"),
        ],
    ),
    ToolSpec(
        tool_id=4,
        tool_name="Diagrammatic-Text Explainer",
        grounding_mode="d2l",
        target_behavior="Visual or spatial textual explanation using diagram-like layout.",
        expected_question_patterns=["visualize", "draw", "diagram", "illustrate"],
        disallowed_confounders=["quiz or assessment framing", "recent developments"],
        drafts=[
            CoreQuestionDraft("diagram", "Draw a text-based diagram of a transformer encoder block with attention, residuals, and feed-forward layers.", "Show the encoder flow with labeled components and data movement.", retrieval_query="transformer encoder block attention residual feed forward"),
            CoreQuestionDraft("diagram", "Illustrate a CNN pipeline from input image to feature maps to classifier using an ASCII-style layout.", "Show the major stages and how spatial dimensions evolve.", retrieval_query="cnn feature maps pooling classifier pipeline"),
            CoreQuestionDraft("diagram", "Create a diagrammatic explanation of the backpropagation computation graph for a small network.", "Show nodes for activations, loss, and gradient flow.", retrieval_query="backpropagation computation graph neural network"),
            CoreQuestionDraft("diagram", "Visualize sequence-to-sequence attention with encoder states, attention weights, and decoder outputs.", "Depict the interaction between encoder outputs and decoder steps.", retrieval_query="sequence to sequence attention encoder decoder"),
        ],
    ),
    ToolSpec(
        tool_id=5,
        tool_name="Analogical Reasoner",
        grounding_mode="d2l",
        target_behavior="Big-picture intuitive explanation through analogy.",
        expected_question_patterns=["intuition", "why does this matter", "analogy", "relate this"],
        disallowed_confounders=["explicit step derivation", "table comparison"],
        drafts=[
            CoreQuestionDraft("analogy", "Give an intuitive analogy for how self-attention decides what information to focus on.", "Map attention to a familiar selection process while preserving the key computational intuition.", retrieval_query="self attention intuition mechanism"),
            CoreQuestionDraft("analogy", "Explain regularization with an analogy that makes overfitting easier to understand.", "Use a familiar scenario to explain constraint, simplicity, and generalization.", retrieval_query="regularization overfitting intuition"),
            CoreQuestionDraft("analogy", "Use an analogy to explain why residual connections help deep networks train.", "Connect skip connections to preserving useful signal through many layers.", retrieval_query="residual connections deep networks training"),
            CoreQuestionDraft("analogy", "Relate word embeddings to a real-world spatial analogy.", "Explain how semantic proximity is represented in vector space.", retrieval_query="word embeddings vector space semantics intuition"),
        ],
    ),
    ToolSpec(
        tool_id=6,
        tool_name="Comparative Explainer",
        grounding_mode="d2l",
        target_behavior="Systematic comparison across explicit axes.",
        expected_question_patterns=["compare", "contrast", "difference between", "versus"],
        disallowed_confounders=["single-concept explanation", "open web updates"],
        drafts=[
            CoreQuestionDraft("compare", "Compare CNNs and RNNs for sequence modeling, including their strengths and limits.", "Contrast inductive bias, memory behavior, and typical use cases.", retrieval_query="CNNs vs RNNs sequence modeling"),
            CoreQuestionDraft("compare", "What are the main differences between Adam and SGD as optimizers?", "Compare update behavior, adaptivity, and practical tradeoffs.", retrieval_query="Adam optimizer vs SGD"),
            CoreQuestionDraft("compare", "Compare ResNet and VGG architectures on depth, parameter efficiency, and training behavior.", "Highlight residual connections, network depth, and optimization implications.", retrieval_query="ResNet vs VGG architecture residual connections"),
            CoreQuestionDraft("compare", "Contrast batch normalization and layer normalization.", "Explain where normalization is applied and when each variant is useful.", retrieval_query="batch normalization vs layer normalization"),
        ],
    ),
    ToolSpec(
        tool_id=7,
        tool_name="Concept Map Generator",
        grounding_mode="d2l",
        target_behavior="Hierarchical or networked map of connected ideas.",
        expected_question_patterns=["overview", "map the relationships", "big picture", "how do these connect"],
        disallowed_confounders=["recent web updates", "single numeric example"],
        drafts=[
            CoreQuestionDraft("concept_map", "Create a concept map that connects optimization ideas such as loss, gradients, learning rate, and regularization.", "Show how training objectives, gradients, and optimization controls relate.", retrieval_query="optimization loss gradients learning rate regularization"),
            CoreQuestionDraft("concept_map", "Map the relationship between tokenization, embeddings, attention, and sequence modeling.", "Connect representation learning and sequence-processing components.", retrieval_query="tokenization embeddings attention sequence modeling"),
            CoreQuestionDraft("concept_map", "Build a concept map for the main parts of a computer vision training pipeline.", "Show data, convolutional representation, classification, and evaluation links.", retrieval_query="computer vision training pipeline convolution classification evaluation"),
            CoreQuestionDraft("concept_map", "Show the big-picture connections among training, validation, testing, and evaluation metrics.", "Organize the stages of model development and how metrics relate to them.", retrieval_query="training validation testing evaluation metrics"),
        ],
    ),
    ToolSpec(
        tool_id=8,
        tool_name="PersonaRAG Adapter",
        grounding_mode="style_fixture",
        target_behavior="Adapt retrieved content into the student's learning style.",
        expected_question_patterns=["adapt this to my style", "re-explain this for me", "rewrite this chunk"],
        disallowed_confounders=["new retrieval requests", "open web questions"],
        drafts=[
            CoreQuestionDraft("adapt_retrieved_text", "Rewrite this retrieved explanation of gradient descent in the student's learning style.", "Preserve the factual meaning while restyling the passage to match the student's preferences.", source_text="Gradient descent updates model parameters by moving them in the direction that reduces loss. The learning rate controls how large each update is, and choosing it well affects both stability and speed of convergence."),
            CoreQuestionDraft("adapt_retrieved_text", "Adapt this D2L-style passage about batch normalization to fit the student's preferred way of learning.", "Keep the same content but convert the presentation style to match the learner profile.", source_text="Batch normalization normalizes intermediate activations within a minibatch. This can stabilize training, permit larger learning rates, and reduce sensitivity to initialization."),
            CoreQuestionDraft("adapt_retrieved_text", "Restyle this explanation of self-attention for the current student profile.", "Transform the wording and structure while preserving the core explanation of attention.", source_text="Self-attention computes how strongly each token should attend to other tokens in the same sequence. It uses query, key, and value projections to build context-aware representations."),
            CoreQuestionDraft("adapt_retrieved_text", "Take this short textbook explanation of overfitting and rewrite it in the learner's style.", "Retain the original claim while converting the form of presentation.", source_text="Overfitting occurs when a model learns patterns that are too specific to the training data. Regularization techniques help control model complexity and improve generalization."),
        ],
    ),
    ToolSpec(
        tool_id=9,
        tool_name="FSLSM Styler",
        grounding_mode="style_fixture",
        target_behavior="Transform existing content from one style to another on demand.",
        expected_question_patterns=["explain this differently", "make this more visual", "convert this explanation"],
        disallowed_confounders=["brand-new retrieval task", "chapter summarization"],
        drafts=[
            CoreQuestionDraft("style_transfer", "Transform this explanation of embeddings into a more visual and intuitive teaching style.", "Convert the same explanation into a different FSLSM presentation style.", source_text="Embeddings represent discrete tokens as dense vectors so that similar tokens occupy nearby positions in vector space. These learned representations support downstream prediction tasks."),
            CoreQuestionDraft("style_transfer", "Convert this sequential derivation of convolution output shape into a more global summary style.", "Preserve the meaning but change the pedagogical presentation.", source_text="To compute the output shape of a convolution, start with the input height and width, account for padding, subtract the kernel size, divide by the stride, and then add one."),
            CoreQuestionDraft("style_transfer", "Rewrite this abstract explanation of regularization into a more concrete, sensing-oriented style.", "Retain the content while shifting from abstract to concrete presentation.", source_text="Regularization constrains model flexibility so that training minimizes not only empirical loss but also excessive complexity that harms generalization."),
            CoreQuestionDraft("style_transfer", "Take this prose explanation of attention and make it more structured and stepwise.", "Change the delivery format without changing the substance.", source_text="Attention allows a model to weigh different parts of the input differently when producing each output representation."),
        ],
    ),
    ToolSpec(
        tool_id=10,
        tool_name="Think-Pair-Share Generator",
        grounding_mode="style_fixture",
        target_behavior="Generate reflective think-pair-share prompts.",
        expected_question_patterns=["reflect on", "think-pair-share", "pause and discuss"],
        disallowed_confounders=["coding exercise request", "quiz with fixed answers"],
        drafts=[
            CoreQuestionDraft("reflective_discussion", "Create a think-pair-share activity about vanishing gradients in deep networks.", "Produce a three-phase reflective activity about diagnosing and reasoning through vanishing gradients.", source_text="Deep neural networks can become hard to optimize when gradients shrink as they pass backward through many layers."),
            CoreQuestionDraft("reflective_discussion", "Generate a think-pair-share prompt sequence on choosing between CNNs and transformers for a vision task.", "Encourage reflection, peer comparison, and synthesis about architecture choice.", source_text="Different model architectures trade off inductive bias, data efficiency, and computational cost."),
            CoreQuestionDraft("reflective_discussion", "Design a reflective think-pair-share activity around optimizer selection in deep learning.", "Invite the learner to reason about optimizer tradeoffs before discussion.", source_text="Optimizers such as SGD and Adam differ in adaptivity, stability, and hyperparameter sensitivity."),
            CoreQuestionDraft("reflective_discussion", "Make a think-pair-share sequence for interpreting what attention weights might reveal in a model.", "Prompt internal reflection and discussion about what attention can and cannot tell us.", source_text="Attention weights are sometimes used as interpretability signals, but they do not always map directly to causal importance."),
        ],
    ),
    ToolSpec(
        tool_id=11,
        tool_name="Interactive Exercise Generator",
        grounding_mode="style_fixture",
        target_behavior="Create a hands-on exercise with instructions and hints.",
        expected_question_patterns=["give me an exercise", "hands-on practice", "let me implement"],
        disallowed_confounders=["reflection-only framing", "recent web topic"],
        drafts=[
            CoreQuestionDraft("hands_on_exercise", "Create a hands-on PyTorch exercise for implementing linear regression from scratch.", "Provide setup, task steps, expected outputs, and hints for an implementation task.", source_text="The learner should practice model definition, loss computation, and gradient-based parameter updates."),
            CoreQuestionDraft("hands_on_exercise", "Design an interactive exercise where the student computes attention weights for a tiny example.", "Require the learner to do a concrete calculation rather than just read an explanation.", source_text="Use a toy query-key-value example so the learner can compute similarity scores and normalized weights."),
            CoreQuestionDraft("hands_on_exercise", "Give me a practice task for exploring how stride and padding change CNN output shapes.", "Provide an exercise with multiple parameter settings and expected checks.", source_text="The student should vary input size, stride, and padding and reason about the resulting feature map sizes."),
            CoreQuestionDraft("hands_on_exercise", "Create a coding exercise for checking gradients numerically against backpropagation.", "Ask the learner to implement finite-difference checking and compare results.", source_text="Gradient checking compares analytical gradients from backpropagation with numerical approximations."),
        ],
    ),
    ToolSpec(
        tool_id=12,
        tool_name="Quiz Generator",
        grounding_mode="style_fixture",
        target_behavior="Generate assessment questions with feedback.",
        expected_question_patterns=["quiz me", "test my understanding", "check what I know"],
        disallowed_confounders=["open-ended reflection", "coding implementation task"],
        drafts=[
            CoreQuestionDraft("quiz", "Quiz me on activation functions and when they are used.", "Generate concrete assessment items with brief explanations for the answers.", source_text="Focus on ReLU, sigmoid, tanh, and softmax in common deep learning settings."),
            CoreQuestionDraft("quiz", "Create a short quiz on embeddings and token representations.", "Assess factual knowledge of what embeddings encode and why they are useful.", source_text="Cover discrete tokens, dense vectors, semantic proximity, and downstream modeling."),
            CoreQuestionDraft("quiz", "Test my understanding of common optimizers in deep learning.", "Generate short factual checks on SGD, momentum, and Adam.", source_text="Use a mix of conceptual and concrete optimizer questions."),
            CoreQuestionDraft("quiz", "Make a quiz that checks whether I understand RNNs, LSTMs, and GRUs.", "Probe architectural differences and use cases with direct feedback.", source_text="Focus on sequence modeling, gating, and long-range dependency handling."),
        ],
    ),
    ToolSpec(
        tool_id=13,
        tool_name="Summarizer",
        grounding_mode="style_fixture",
        target_behavior="Concise synthesis of existing content.",
        expected_question_patterns=["summarize", "overview", "key points", "tl;dr"],
        disallowed_confounders=["request for latest external info", "stepwise derivation"],
        drafts=[
            CoreQuestionDraft("summarize", "Summarize this short passage on regularization and model generalization.", "Condense the main takeaways and key terms into a compact synthesis.", source_text="Regularization techniques such as weight decay and dropout help reduce overfitting by constraining model complexity. They aim to improve generalization so that the model performs well not only on training data but also on unseen examples."),
            CoreQuestionDraft("summarize", "Give me a concise summary of this overview of CNN architectures.", "Extract the central ideas and organize them into a short synthesis.", source_text="Convolutional neural networks process images using learned filters, hierarchical feature extraction, pooling, and a classification head. Architectural innovations such as residual connections help deeper networks train effectively."),
            CoreQuestionDraft("summarize", "Summarize this discussion of transformers versus recurrent models.", "Produce a short synthesis of the main differences and implications.", source_text="Transformers rely on attention to model token interactions in parallel, while recurrent networks process sequences step by step. This changes training efficiency, scaling behavior, and the kinds of dependencies the models handle well."),
            CoreQuestionDraft("summarize", "Provide a concise summary of this explanation of the model training loop.", "Highlight the essential training stages and terminology.", source_text="A typical training loop repeatedly computes predictions, evaluates loss, performs backpropagation, updates parameters, and tracks validation metrics to monitor generalization."),
        ],
    ),
    ToolSpec(
        tool_id=14,
        tool_name="Content Retriever",
        grounding_mode="d2l",
        target_behavior="Return the most relevant D2L passages without style transformation.",
        expected_question_patterns=["find the relevant section", "retrieve the passage", "what does D2L say"],
        disallowed_confounders=["asks for explanation instead of retrieval", "requests latest information"],
        drafts=[
            CoreQuestionDraft("locate_d2l_content", "Find the most relevant D2L passage about BLEU score and machine translation evaluation.", "Return the textbook section most directly tied to BLEU and MT evaluation.", retrieval_query="BLEU score machine translation evaluation"),
            CoreQuestionDraft("locate_d2l_content", "Retrieve the D2L section that explains Xavier initialization.", "Locate the chunk that defines or motivates Xavier initialization.", retrieval_query="Xavier initialization parameter initialization"),
            CoreQuestionDraft("locate_d2l_content", "Find the D2L explanation of anchor boxes in object detection.", "Return the most relevant object-detection passage about anchor boxes.", retrieval_query="anchor boxes object detection"),
            CoreQuestionDraft("locate_d2l_content", "Locate the D2L content that introduces Nadaraya-Watson attention pooling.", "Retrieve the section that explains the attention pooling method.", retrieval_query="Nadaraya-Watson attention pooling"),
        ],
    ),
    ToolSpec(
        tool_id=15,
        tool_name="Web Search Tool",
        grounding_mode="search",
        target_behavior="Find recent or beyond-D2L information from the web.",
        expected_question_patterns=["latest", "recent developments", "industry use", "beyond D2L"],
        disallowed_confounders=["purely textbook retrieval", "single-concept definition request"],
        drafts=[
            CoreQuestionDraft("external_search", "What are the latest transformer architecture developments beyond the material covered in D2L?", "Find recent advances or trends in transformer design that are newer than the textbook content.", target_evidence_criteria=["recent transformer architectures", "beyond textbook coverage", "named external sources"]),
            CoreQuestionDraft("external_search", "How are diffusion models being used in current real-world applications?", "Retrieve recent external examples or summaries of diffusion model use beyond course material.", target_evidence_criteria=["recent application examples", "external sources", "non-D2L grounding"]),
            CoreQuestionDraft("external_search", "What are some recent industry practices for fine-tuning large language models?", "Search for current external guidance or examples of LLM fine-tuning approaches.", target_evidence_criteria=["recent fine-tuning practices", "industry or research sources", "post-D2L context"]),
            CoreQuestionDraft("external_search", "What are the latest benchmark trends for graph neural networks outside the D2L textbook?", "Find recent benchmark or survey information about GNN performance and evaluation.", target_evidence_criteria=["recent GNN benchmarks", "external sources", "beyond D2L"]),
        ],
    ),
]


def build_tool_specs_payload() -> list[dict[str, Any]]:
    payload = []
    for spec in TOOL_SPEC_MATRIX:
        payload.append(
            {
                "tool_id": spec.tool_id,
                "tool_name": spec.tool_name,
                "grounding_mode": spec.grounding_mode,
                "target_behavior": spec.target_behavior,
                "expected_question_patterns": spec.expected_question_patterns,
                "disallowed_confounders": spec.disallowed_confounders,
                "n_questions": len(spec.drafts),
            }
        )
    return payload


def build_core_questions() -> list[dict[str, Any]]:
    retriever = D2LRetriever()
    rows: list[dict[str, Any]] = []
    counter = 1
    for spec in TOOL_SPEC_MATRIX:
        for draft in spec.drafts:
            essential_chunk_ids: list[str] = []
            support_chunk_ids: list[str] = []
            if spec.grounding_mode == "d2l":
                retrieval = retriever.retrieve(draft.retrieval_query or draft.question, k=5)
                chunk_ids = retrieval["chunk_ids"]
                essential_chunk_ids = chunk_ids[: min(2, len(chunk_ids))]
                support_chunk_ids = chunk_ids[min(2, len(chunk_ids)) : 5]

            rows.append(
                {
                    "question_id": f"EXP3C_{counter:03d}",
                    "question": draft.question,
                    "question_family": draft.question_family,
                    "question_type": draft.question_family,
                    "grounding_mode": spec.grounding_mode,
                    "gold_answer_brief": draft.gold_answer_brief,
                    "essential_chunk_ids": essential_chunk_ids,
                    "support_chunk_ids": support_chunk_ids,
                    "source_text": draft.source_text,
                    "target_evidence_criteria": draft.target_evidence_criteria or [],
                    "manual_review_status": "draft",
                    "manual_review_notes": "",
                    "needs_corpus": spec.grounding_mode == "d2l",
                    "retrieval_query": draft.retrieval_query,
                    "tool_uniqueness_notes": f"Designed to primarily activate {spec.tool_name}.",
                }
            )
            counter += 1
    return rows


def build_core_answer_key() -> dict[str, dict[str, Any]]:
    answer_key: dict[str, dict[str, Any]] = {}
    profiles = load_canonical_profiles()
    counter = 1
    for spec in TOOL_SPEC_MATRIX:
        for draft in spec.drafts:
            answer_key[f"EXP3C_{counter:03d}"] = {
                "target_tool_id": spec.tool_id,
                "profile_eval_eligible": is_profile_eval_eligible(draft.question_family),
                "profile_target_tool_ids": build_profile_target_tool_ids(
                    question_family=draft.question_family,
                    target_tool_id=spec.tool_id,
                    profiles=profiles,
                ),
            }
            counter += 1
    return answer_key


def serialize_core_questions() -> str:
    return json.dumps(build_core_questions(), indent=2)
