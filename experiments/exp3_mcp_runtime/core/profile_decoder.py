"""
Decodes ±1 bipolar FSLSM profiles into named dimension sets.
Handles both formats:
  New bipolar: {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1}
  Old binary:  {"act": 1, "ref": 0, "vis": 1, ...}
"""
from typing import Dict, Set


def decode_profile(profile: dict) -> Set[str]:
    """Return set of active pole names, e.g. {"Active", "Sensing", "Visual", "Sequential"}."""
    if "act_ref" in profile:
        dims: Set[str] = set()
        dims.add("Active"     if profile["act_ref"] < 0 else "Reflective")
        dims.add("Sensing"    if profile["sen_int"] < 0 else "Intuitive")
        dims.add("Visual"     if profile["vis_ver"] < 0 else "Verbal")
        dims.add("Sequential" if profile["seq_glo"] < 0 else "Global")
        return dims
    else:
        field_map = {
            "act": "Active", "ref": "Reflective",
            "sns": "Sensing", "int": "Intuitive",
            "vis": "Visual",  "vrb": "Verbal",
            "seq": "Sequential", "glo": "Global",
        }
        return {field_map[f] for f, v in profile.items() if v == 1 and f in field_map}


def profile_to_label(profile: dict) -> str:
    """Return canonical label, e.g. "Active-Sensing-Visual-Sequential"."""
    dims = decode_profile(profile)
    act = "Active"     if "Active"     in dims else "Reflective"
    sns = "Sensing"    if "Sensing"    in dims else "Intuitive"
    vis = "Visual"     if "Visual"     in dims else "Verbal"
    seq = "Sequential" if "Sequential" in dims else "Global"
    return f"{act}-{sns}-{vis}-{seq}"


def profile_to_tuple(profile: dict) -> tuple:
    """Return (act, sns, vis, seq) tuple for GROUND_TRUTH_MAP_FULL lookup."""
    dims = decode_profile(profile)
    return (
        "Active"     if "Active"     in dims else "Reflective",
        "Sensing"    if "Sensing"    in dims else "Intuitive",
        "Visual"     if "Visual"     in dims else "Verbal",
        "Sequential" if "Sequential" in dims else "Global",
    )


def get_primary_dim(profile: dict) -> str:
    """Return the single most pedagogically salient pole.

    Priority: Visual > Sequential > Active > Global >
              Intuitive > Sensing > Reflective > Verbal
    """
    dims = decode_profile(profile)
    priority = ["Visual", "Sequential", "Active", "Global",
                "Intuitive", "Sensing", "Reflective", "Verbal"]
    for dim in priority:
        if dim in dims:
            return dim
    return "Verbal"
