# FSLSM Dimensions — order matters (used in arrays/vectors throughout the codebase)
FSLSM_DIMENSIONS = ["act_ref", "sen_int", "vis_ver", "seq_glo"]

# Human-readable labels per dimension: (negative_pole, positive_pole)
FSLSM_DIM_LABELS = {
    "act_ref": ("Active", "Reflective"),    # -1 = Active,      +1 = Reflective
    "sen_int": ("Sensing", "Intuitive"),    # -1 = Sensing,     +1 = Intuitive
    "vis_ver": ("Visual", "Verbal"),        # -1 = Visual,      +1 = Verbal
    "seq_glo": ("Sequential", "Global"),    # -1 = Sequential,  +1 = Global
}

# Experiment sizing
NUM_PROFILES = 16        # 2^4 binary FSLSM combinations
NUM_INSTANCES = 5        # agent instances per profile
NUM_AGENTS = 80          # total = 16 × 5

# Knowledge levels per agent instance
KNOWLEDGE_LEVELS = ["beginner", "intermediate", "advanced"]
KNOWLEDGE_LEVEL_MAP: dict[int, str | None] = {
    1: "beginner",
    2: "intermediate",
    3: "advanced",
    4: None,   # general
    5: None,   # general
}
NUM_LEVELED_AGENTS = 48  # 16 profiles × 3 leveled instances
NUM_GENERAL_AGENTS = 32  # 16 profiles × 2 general instances

# Non-Personalized Baseline
BASELINE_PROFILE_CODE = "P00_Baseline"
NUM_BASELINE_INSTANCES = 5

# ILS questionnaire
ILS_NUM_QUESTIONS = 44
ILS_SCORE_MIN = -11      # max preference for negative pole
ILS_SCORE_MAX = 11       # max preference for positive pole

# ILS dimension question mapping: which questions map to which dimension
# Each dimension gets 11 questions, interleaved every 4 questions
# act_ref: q1,5,9,13,17,21,25,29,33,37,41
# sen_int: q2,6,10,14,18,22,26,30,34,38,42
# vis_ver: q3,7,11,15,19,23,27,31,35,39,43
# seq_glo: q4,8,12,16,20,24,28,32,36,40,44
ILS_QUESTION_DIM_MAP: dict[int, str] = {
    q: dim
    for dim, start in zip(FSLSM_DIMENSIONS, [1, 2, 3, 4])
    for q in range(start, 45, 4)
}

# Tool registry IDs (used in Exp3)
TOOL_IDS = [
    "concept_explainer",
    "step_by_step_derivator",
    "worked_example_generator",
    "quiz_generator",
    "summarizer",
    "diagrammatic_text_explainer",
    "interactive_exercise",
    "case_study_retriever",
]

# LLM models used across experiments
MODELS = {
    "gpt-4.1-mini": "openai",
    "claude-sonnet-4-20250514": "anthropic",
    "llama3.1:8b": "ollama",
    "qwen2.5:7b": "ollama",
    "gemma2:9b": "ollama",
    "gemma3:12b": "ollama",
}
