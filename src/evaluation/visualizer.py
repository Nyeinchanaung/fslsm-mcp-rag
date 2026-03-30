"""Visualization functions for Experiment 1 results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.constants import FSLSM_DIMENSIONS, FSLSM_DIM_LABELS

# Readable axis labels for radar charts
DIM_AXIS_LABELS = [
    f"{neg}/{pos}" for neg, pos in FSLSM_DIM_LABELS.values()
]

FIGURES_DIR = Path("results/exp1/figures")


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Radar chart per agent ─────────────────────────────────────────

def radar_chart(
    agent_uid: str,
    assigned: dict[str, int],
    detected_avg: dict[str, float],
    save_path: str | Path | None = None,
):
    """
    Polar plot with 4 axes: assigned profile (blue) vs ILS result (red dashed).

    assigned/detected_avg values should be in [-11, +11] range for raw scores,
    or ±1 for binary.
    """
    _ensure_dir()

    assigned_vals = [assigned[d] for d in FSLSM_DIMENSIONS]
    detected_vals = [detected_avg[d] for d in FSLSM_DIMENSIONS]

    angles = np.linspace(0, 2 * np.pi, len(FSLSM_DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    assigned_vals += assigned_vals[:1]
    detected_vals += detected_vals[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5, 5))
    ax.plot(angles, assigned_vals, "b-o", label="Assigned Profile", linewidth=2)
    ax.plot(angles, detected_vals, "r--o", label="ILS Result", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_AXIS_LABELS, fontsize=9)
    ax.set_title(f"Agent {agent_uid}", pad=15, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()

    path = save_path or FIGURES_DIR / f"radar_{agent_uid}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 2. Heatmap: 16 profiles × 4 dimensions ──────────────────────────

def heatmap_profiles(
    results: list[dict],
    model_name: str,
    save_path: str | Path | None = None,
):
    """
    Heatmap showing per-profile, per-dimension alignment rate (0 → 1).
    """
    _ensure_dir()

    # Group by profile_code
    profile_dim_matches: dict[str, dict[str, list[int]]] = {}
    for r in results:
        code = r["profile_code"]
        if code not in profile_dim_matches:
            profile_dim_matches[code] = {d: [] for d in FSLSM_DIMENSIONS}
        for d in FSLSM_DIMENSIONS:
            detected = r["detected"][d]
            assigned = r["assigned"][d]
            match = int(assigned == detected) if detected != 0 else 0
            profile_dim_matches[code][d].append(match)

    # Build matrix
    profile_codes = sorted(profile_dim_matches.keys())
    matrix = np.array([
        [np.mean(profile_dim_matches[code][d]) for d in FSLSM_DIMENSIONS]
        for code in profile_codes
    ])

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(FSLSM_DIMENSIONS)))
    ax.set_xticklabels(DIM_AXIS_LABELS, fontsize=9)
    ax.set_yticks(range(len(profile_codes)))
    ax.set_yticklabels(profile_codes, fontsize=7)
    ax.set_title(f"Profile × Dimension Alignment — {model_name}", fontsize=11)

    # Annotate cells
    for i in range(len(profile_codes)):
        for j in range(len(FSLSM_DIMENSIONS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Alignment Rate")
    plt.tight_layout()

    path = save_path or FIGURES_DIR / f"heatmap_{model_name.replace('/', '_').replace(':', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 3. Model comparison bar chart ────────────────────────────────────

def model_comparison_bar(
    df_pra: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Grouped bars: 3 models × 4 dimensions, Y = PRA.
    df_pra must have columns: model, dimension, pra, knowledge_level.
    Filters to knowledge_level == 'ALL' and excludes 'overall_4d'.
    """
    _ensure_dir()

    df = df_pra[
        (df_pra["knowledge_level"] == "ALL")
        & (df_pra["dimension"] != "overall_4d")
    ].copy()

    models = df["model"].unique()
    dims = FSLSM_DIMENSIONS
    x = np.arange(len(dims))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        pra_vals = [
            model_data[model_data["dimension"] == d]["pra"].values[0]
            if len(model_data[model_data["dimension"] == d]) > 0
            else 0
            for d in dims
        ]
        bars = ax.bar(x + i * width, pra_vals, width, label=model)
        for bar, val in zip(bars, pra_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(DIM_AXIS_LABELS, fontsize=9)
    ax.set_ylabel("PRA")
    ax.set_ylim(0, 1.1)
    ax.set_title("Profile Recovery Accuracy by Model & Dimension")
    ax.legend(fontsize=8)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    plt.tight_layout()

    path = save_path or FIGURES_DIR / "model_comparison_pra.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 4. Knowledge-level PRA comparison ────────────────────────────────

def knowledge_level_comparison(
    df_pra: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Grouped bars: knowledge levels × overall_4d PRA, one group per model.
    """
    _ensure_dir()

    df = df_pra[
        (df_pra["dimension"] == "overall_4d")
        & (df_pra["knowledge_level"] != "ALL")
    ].copy()

    models = df["model"].unique()
    levels = sorted(df["knowledge_level"].unique())
    x = np.arange(len(levels))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        pra_vals = [
            model_data[model_data["knowledge_level"] == lv]["pra"].values[0]
            if len(model_data[model_data["knowledge_level"] == lv]) > 0
            else 0
            for lv in levels
        ]
        bars = ax.bar(x + i * width, pra_vals, width, label=model)
        for bar, val in zip(bars, pra_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("PRA (overall 4D)")
    ax.set_ylim(0, 1.1)
    ax.set_title("PRA by Knowledge Level")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = save_path or FIGURES_DIR / "knowledge_level_pra.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 5. Cost per model bar chart ──────────────────────────────────────

def cost_per_model_bar(
    df_cost: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Simple bar chart: total USD cost per model for Exp1.
    Uses LiteLLM's per-call cost tracking.
    """
    _ensure_dir()

    # Sum FSLSM + Baseline cost per model
    df_agg = df_cost.groupby("model", sort=False)["total_usd"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    models = df_agg["model"].values
    costs = df_agg["total_usd"].values
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(range(len(models)), costs, color=colors[: len(models)])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [m.split(":")[0].split("-")[0] for m in models],
        rotation=15, fontsize=9,
    )
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Experiment 1 — API Cost per Model (via LiteLLM)")

    for bar, cost in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"${cost:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    path = save_path or FIGURES_DIR / "cost_per_model.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── 6. FSLSM vs Baseline bar chart ─────────────────────────────────

def fslsm_vs_baseline_bar(
    df_pra: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Side-by-side bars: FSLSM PRA vs Baseline PRA per model (overall_4d).
    Demonstrates that FSLSM persona encoding drives high PRA.
    """
    _ensure_dir()

    df = df_pra[
        (df_pra["dimension"] == "overall_4d")
        & (df_pra["knowledge_level"] == "ALL")
    ].copy()

    models = df["model"].unique()
    conditions = ["FSLSM", "Baseline"]
    x = np.arange(len(models))
    width = 0.35
    colors = {"FSLSM": "#2196F3", "Baseline": "#FF9800"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(conditions):
        cond_data = df[df["condition"] == cond]
        pra_vals = [
            cond_data[cond_data["model"] == m]["pra"].values[0]
            if len(cond_data[cond_data["model"] == m]) > 0 else 0
            for m in models
        ]
        bars = ax.bar(
            x + i * width, pra_vals, width,
            label=cond, color=colors[cond],
        )
        for bar, val in zip(bars, pra_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, fontsize=9, rotation=10)
    ax.set_ylabel("PRA (overall 4D)")
    ax.set_ylim(0, 1.15)
    ax.set_title("FSLSM Personalized vs Non-Personalized Baseline PRA")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance level")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = save_path or FIGURES_DIR / "fslsm_vs_baseline_pra.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 7. Baseline bias radar chart ────────────────────────────────────

def baseline_bias_radar(
    df_baseline: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Radar chart showing each model's natural dimension bias (raw scores).
    Overlays all 3 models on one polar plot to compare innate LLM tendencies.
    """
    _ensure_dir()

    dims = FSLSM_DIMENSIONS
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))

    for i, (_, row) in enumerate(df_baseline.iterrows()):
        scores = [row[f"bias_{d}_score"] for d in dims]
        scores += scores[:1]
        ax.plot(
            angles, scores, "o-",
            color=colors[i % len(colors)],
            label=row["model"], linewidth=2,
        )
        ax.fill(angles, scores, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_AXIS_LABELS, fontsize=9)
    ax.set_ylim(-11, 11)
    ax.set_yticks([-11, -5, 0, 5, 11])
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_title("Baseline Natural Dimension Bias by Model", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    plt.tight_layout()

    path = save_path or FIGURES_DIR / "baseline_bias_radar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 8. Baseline bias heatmap ─────────────────────────────────────────

def heatmap_baseline_bias(
    baseline_results: list[dict],
    model_name: str,
    save_path: str | Path | None = None,
):
    """
    Heatmap showing raw ILS scores per baseline agent instance × 4 dimensions.
    Diverging colormap centered at 0, range [-11, +11].
    """
    _ensure_dir()

    # Group by agent_uid, average across trials
    agent_scores: dict[str, dict[str, list[float]]] = {}
    for r in baseline_results:
        uid = r["agent_uid"]
        if uid not in agent_scores:
            agent_scores[uid] = {d: [] for d in FSLSM_DIMENSIONS}
        for d in FSLSM_DIMENSIONS:
            agent_scores[uid][d].append(r["raw_scores"][d])

    agent_uids = sorted(agent_scores.keys())
    matrix = np.array([
        [np.mean(agent_scores[uid][d]) for d in FSLSM_DIMENSIONS]
        for uid in agent_uids
    ])

    fig, ax = plt.subplots(figsize=(8, max(4, len(agent_uids) * 0.6)))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-11, vmax=11)

    ax.set_xticks(range(len(FSLSM_DIMENSIONS)))
    ax.set_xticklabels(DIM_AXIS_LABELS, fontsize=9)
    ax.set_yticks(range(len(agent_uids)))
    ax.set_yticklabels([u.split("_", 1)[1] for u in agent_uids], fontsize=7)
    ax.set_title(f"Baseline Raw ILS Scores — {model_name}", fontsize=11)

    for i in range(len(agent_uids)):
        for j in range(len(FSLSM_DIMENSIONS)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Raw Score (−11 to +11)")
    plt.tight_layout()

    safe = model_name.replace("/", "_").replace(":", "_")
    path = save_path or FIGURES_DIR / f"heatmap_baseline_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 9. DAS FSLSM vs Baseline bar chart ─────────────────────────────

def das_fslsm_vs_baseline_bar(
    df_das: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Side-by-side bars: FSLSM DAS vs Baseline DAS per model (overall_4d).
    """
    _ensure_dir()

    df = df_das[
        (df_das["dimension"] == "overall_4d")
        & (df_das["knowledge_level"] == "ALL")
    ].copy()

    models = df["model"].unique()
    conditions = ["FSLSM", "Baseline"]
    x = np.arange(len(models))
    width = 0.35
    colors = {"FSLSM": "#2196F3", "Baseline": "#FF9800"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(conditions):
        cond_data = df[df["condition"] == cond]
        das_vals = [
            cond_data[cond_data["model"] == m]["das"].values[0]
            if len(cond_data[cond_data["model"] == m]) > 0 else 0
            for m in models
        ]
        bars = ax.bar(
            x + i * width, das_vals, width,
            label=cond, color=colors[cond],
        )
        for bar, val in zip(bars, das_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, fontsize=9, rotation=10)
    ax.set_ylabel("DAS (overall 4D)")
    ax.set_ylim(0, 1.15)
    ax.set_title("FSLSM Personalized vs Non-Personalized Baseline DAS")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance level")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = save_path or FIGURES_DIR / "das_fslsm_vs_baseline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 10. Per-question alignment heatmap ──────────────────────────────

def per_question_alignment_heatmap(
    df_pq: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    44-row × N-model heatmap showing per-question alignment rate.

    Questions are ordered 1-44; dimension group dividers and color-coded
    y-axis labels make it easy to see which dimension drives misalignment.

    Args:
        df_pq: DataFrame with columns model, q_num, dimension, alignment_rate
    """
    _ensure_dir()

    models = list(df_pq["model"].unique())
    q_nums = sorted(df_pq["q_num"].unique())

    # Build matrix (44 × n_models)
    matrix = np.array([
        [
            df_pq[(df_pq["q_num"] == qn) & (df_pq["model"] == m)]["alignment_rate"].values[0]
            if len(df_pq[(df_pq["q_num"] == qn) & (df_pq["model"] == m)]) > 0 else np.nan
            for m in models
        ]
        for qn in q_nums
    ])

    # Dimension boundaries (questions are interleaved every 4)
    dim_order = ["act_ref", "sen_int", "vis_ver", "seq_glo"]
    dim_colors = {"act_ref": "#1f77b4", "sen_int": "#ff7f0e",
                  "vis_ver": "#2ca02c", "seq_glo": "#d62728"}
    # Map q_num → dimension
    q_dim = dict(zip(
        df_pq["q_num"].unique(),
        df_pq.set_index("q_num")["dimension"].to_dict().values()
    ))
    q_dim = df_pq.drop_duplicates("q_num").set_index("q_num")["dimension"].to_dict()

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 18))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    short_names = [m.split(":")[0].split("-")[0] for m in models]
    ax.set_xticklabels(short_names, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(q_nums)))

    # Color-coded y-axis labels by dimension
    ax.set_yticklabels([f"Q{qn}" for qn in q_nums], fontsize=7)
    for tick, qn in zip(ax.get_yticklabels(), q_nums):
        dim = q_dim.get(qn, "act_ref")
        tick.set_color(dim_colors[dim])

    # Annotate each cell
    for i, qn in enumerate(q_nums):
        for j in range(len(models)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black")

    # Horizontal dividers between dimension groups
    prev_dim = None
    for i, qn in enumerate(q_nums):
        curr_dim = q_dim.get(qn)
        if curr_dim != prev_dim and prev_dim is not None:
            ax.axhline(i - 0.5, color="white", linewidth=2)
        prev_dim = curr_dim

    ax.set_title("Per-Question Alignment Rate — All Models", fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.4, label="Alignment Rate (0–1)")

    # Legend for dimension colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=dim_colors[d], label=f"{FSLSM_DIM_LABELS[d][0]}/{FSLSM_DIM_LABELS[d][1]}")
        for d in dim_order
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.25, 1.0), fontsize=8, title="Dimension")

    plt.tight_layout()
    path = save_path or FIGURES_DIR / "per_question_alignment_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 11. DAS model comparison bar chart ──────────────────────────────

def das_comparison_bar(
    df_das: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """
    Grouped bars: 3 models × 4 dimensions, Y = mean DAS.
    Filters to knowledge_level == 'ALL' and excludes 'overall_4d'.
    """
    _ensure_dir()

    df = df_das[
        (df_das["knowledge_level"] == "ALL")
        & (df_das["dimension"] != "overall_4d")
    ].copy()

    models = df["model"].unique()
    dims = FSLSM_DIMENSIONS
    x = np.arange(len(dims))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        das_vals = [
            model_data[model_data["dimension"] == d]["das"].values[0]
            if len(model_data[model_data["dimension"] == d]) > 0 else 0.0
            for d in dims
        ]
        bars = ax.bar(x + i * width, das_vals, width, label=model)
        for bar, val in zip(bars, das_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(DIM_AXIS_LABELS, fontsize=9)
    ax.set_ylabel("DAS (cosine similarity)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Dimension Alignment Score by Model & Dimension")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = save_path or FIGURES_DIR / "das_comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
