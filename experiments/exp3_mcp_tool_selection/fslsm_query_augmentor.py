"""FSLSM query augmentor.

Places short tool-vocabulary keyword phrases BEFORE the query so that the
combined embedding is dominated by the FSLSM profile signal rather than the
query content. This is necessary because the ground truth (PROFILE_TOOL_MAP)
is profile-driven: the same query has different optimal tools for different
profiles, so S1b must be able to steer FAISS toward the profile-appropriate tool.

Keyword phrases are drawn directly from the updated tool descriptions in
`tool_registry.py` to maximise cosine overlap with the target tool embedding.

Profile encoding (matches `data/agents/validated_agents.json`):
    act_ref: -1 = Active,    +1 = Reflective
    sen_int: -1 = Sensing,   +1 = Intuitive
    vis_ver: -1 = Visual,    +1 = Verbal
    seq_glo: -1 = Sequential,+1 = Global
"""
from __future__ import annotations


# Short keyword phrases drawn from updated tool descriptions.
# Kept concise so all 4 poles combined are ~25 words — directive-first format
# ensures these keywords carry strong weight in the embedding.
DIM_DIRECTIVES: dict[str, str] = {
    "Active":      "hands-on interactive simulation code practice experiment doing",
    "Reflective":  "structured outline reflection prose review thinking alone write",
    "Sensing":     "concrete numerical worked example calculation procedure explicit facts",
    "Intuitive":   "abstract theory formal derivation analogy conceptual innovation proof",
    "Visual":      "diagram flowchart architecture figure visual chart schematic network",
    "Verbal":      "prose explanation written narrative formal derivation verbal text",
    "Sequential":  "step-by-step walkthrough ordered layer-by-layer linear procedure",
    "Global":      "big-picture overview concept-map real-world context connections holistic",
}


def _profile_to_poles(profile: dict) -> list[str]:
    """Order matters — most pedagogically salient dimensions first.

    vis_ver and seq_glo are the primary PROFILE_TOOL_MAP axes (they split the
    16 profiles into the main tool groups), so place them first in the directive.
    """
    return [
        "Verbal"     if profile["vis_ver"] > 0 else "Visual",
        "Global"     if profile["seq_glo"] > 0 else "Sequential",
        "Intuitive"  if profile["sen_int"] > 0 else "Sensing",
        "Reflective" if profile["act_ref"] > 0 else "Active",
    ]


class FSLSMQueryAugmentor:
    """Prepend FSLSM-pole keyword phrases to a query for FSLSM-aware retrieval.

    Directive-first format ensures the profile signal dominates the embedding,
    enabling FAISS to retrieve the profile-appropriate tool even when query
    vocabulary alone would favour a content-matched tool.
    """

    def augment(self, query: str, profile: dict) -> str:
        poles = _profile_to_poles(profile)
        keywords = " ".join(DIM_DIRECTIVES[p] for p in poles)
        return f"{keywords} — {query.strip()}"
