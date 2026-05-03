"""
FSLSM Query Augmentor — S1b condition.
Handles both profile formats (±1 bipolar and old binary).
"""
from experiments.exp3_revised.core.profile_decoder import decode_profile

DIM_DIRECTIVES = {
    "Active":     "interactive exercise quiz practice challenge",
    "Reflective": "think-pair-share synthesis key-takeaways condensed reflection summary",
    "Sensing":    "worked-example concrete-factual step-by-step derivation real-world",
    "Intuitive":  "analogy metaphor cross-domain abstract-pattern",
    "Visual":     "diagram flowchart computation-graph visual-figure labeled",
    "Verbal":     "prose comparison narrative text explanation",
    "Sequential": "step-by-step mathematical-derivation algorithm-execution numbered-stages",
    "Global":     "concept-map holistic-overview thematic big-picture chapter-synthesis",
}


def augment_query(query: str, profile: dict) -> str:
    """Prepend compact FSLSM style directives before the raw query.

    Prepending gives directives more embedding weight vs appending (positional
    bias in mean-pooled sentence encoders).
    """
    dims = decode_profile(profile)
    directives = [DIM_DIRECTIVES[dim] for dim in sorted(dims) if dim in DIM_DIRECTIVES]
    if not directives:
        return query
    return f"{'; '.join(directives)}; {query}"
