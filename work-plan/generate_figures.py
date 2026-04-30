"""
Generate all 9 fresh thesis figures as specified in figure_instructions.md.
Run from repo root: python work-plan/generate_figures.py
Output: figures/  (300 dpi PNG, white background)
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

REPO = os.path.join(os.path.dirname(__file__), "..")

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_BASELINE = "#95A5A6"
C_PROPOSED = "#2980B9"
C_S1A      = "#27AE60"
C_HIGH     = "#1ABC9C"
C_MID      = "#F39C12"
C_FAILED   = "#E74C3C"
C_ACCENT   = "#8E44AD"
C_ORANGE   = "#E67E22"

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


# ══════════════════════════════════════════════════════════════════════════════
# F5-1  exp1_pra_bar.png  — Horizontal bar chart, PRA by model
# ══════════════════════════════════════════════════════════════════════════════
def f5_1():
    models = [
        ("claude-sonnet-4", 1.000, "High"),
        ("gemma3:12b",      1.000, "High"),
        ("gpt-4.1-mini",    0.996, "High"),
        ("qwen2.5:7b",      0.956, "High"),
        ("gemma2:9b",       0.948, "High"),
        ("phi4-mini",       0.915, "High"),
        ("llama3.2:3b",     0.902, "High"),
        ("qwen2.5:3b",      0.890, "High"),
        ("llama3.1:8b",     0.724, "Mid"),
        ("gemma3:4b",       0.718, "Mid"),
        ("mistral:7b",      0.544, "Failed"),
        ("qwen2.5:1.5b",    0.508, "Failed"),
        ("llama3.2:1b",     0.500, "Failed"),
        ("gemma2:2b",       0.500, "Failed"),
        ("gemma3:1b",       0.500, "Failed"),
    ]
    tier_col = {"High": C_HIGH, "Mid": C_MID, "Failed": C_FAILED}

    fig, ax = plt.subplots(figsize=(13/2.54, 9/2.54))
    names  = [m[0] for m in models]
    pras   = [m[1] for m in models]
    tiers  = [m[2] for m in models]
    colors = [tier_col[t] for t in tiers]
    y = np.arange(len(models))

    bars = ax.barh(y, pras, color=colors, height=0.6, zorder=3)

    # Value labels
    for bar, pra in zip(bars, pras):
        ax.text(pra + 0.005, bar.get_y() + bar.get_height()/2,
                f"{pra:.3f}", va="center", ha="left", fontsize=8)

    # Reference lines
    ax.axvline(0.82, color="navy", linestyle="--", linewidth=1.0, zorder=4)
    ax.text(0.821, len(models) - 0.3, "H2 target (0.82)", fontsize=8,
            color="navy", va="top")
    ax.axvline(0.50, color="grey", linestyle="--", linewidth=1.0, zorder=4)
    ax.text(0.501, 0.3, "Random chance", fontsize=8, color="grey", va="bottom")

    # Tier separators + right-margin labels
    tier_bounds = {"High": (0, 7), "Mid": (8, 9), "Failed": (10, 14)}
    tier_labels = {"High": "High (n=8)", "Mid": "Mid (n=2)", "Failed": "Failed (n=5)"}
    prev_tier = models[0][2]
    for i in range(1, len(models)):
        if models[i][2] != prev_tier:
            ax.axhline(i - 0.5, color="black", linewidth=0.6, linestyle="-", zorder=5)
            prev_tier = models[i][2]
    for tier, (lo, hi) in tier_bounds.items():
        mid = (lo + hi) / 2
        ax.text(1.065, mid, tier_labels[tier], va="center", ha="left",
                fontsize=8, color=tier_col[tier], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.40, 1.09)
    ax.set_xticks([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("Profile Reconstruction Accuracy (PRA)")
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.invert_yaxis()
    ax.set_title("PRA by Model", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "exp1_pra_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# F5-2  exp1_radar.png  — Radar chart, FSLSM profile fidelity
# ══════════════════════════════════════════════════════════════════════════════
def f5_2():
    labels = ["Processing\n[act_ref]", "Perception\n[sen_int]",
              "Input\n[vis_ver]", "Understanding\n[seq_glo]"]
    predefined = [-1.0, +1.0, -1.0, +1.0]
    questionnaire = [-0.73, +0.64, -0.55, +0.82]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    pred_vals = predefined + predefined[:1]
    ques_vals = questionnaire + questionnaire[:1]

    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), subplot_kw=dict(polar=True))

    ax.plot(angles, pred_vals, color=C_PROPOSED, linewidth=2, linestyle="-", label="Predefined Index")
    ax.fill(angles, pred_vals, color=C_PROPOSED, alpha=0.20)
    ax.plot(angles, ques_vals, color=C_FAILED, linewidth=2, linestyle="--", label="Questionnaire Result")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "+0.5", "+1.0"], fontsize=7)
    ax.set_ylim(-1.1, 1.1)

    # "Same pole ✓" annotations near each axis tip
    for i, (a, p, q) in enumerate(zip(angles[:-1], predefined, questionnaire)):
        if (p > 0 and q > 0) or (p < 0 and q < 0):
            ax.annotate("Same pole ✓", xy=(a, 1.0), xytext=(a, 1.25),
                        fontsize=7, color="darkgreen", ha="center", va="center")

    ax.set_title("Agent #001 — FSLSM Profile Fidelity", fontsize=10, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    save(fig, "exp1_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
# F5-3  exp1_pra_das_scatter.png  — PRA vs DAS scatter
# ══════════════════════════════════════════════════════════════════════════════
def f5_3():
    points = [
        ("claude-sonnet-4", 1.000, 0.927, "High"),
        ("gemma3:12b",      1.000, 0.882, "High"),
        ("gpt-4.1-mini",    0.996, 0.924, "High"),
        ("qwen2.5:7b",      0.956, 0.785, "High"),
        ("gemma2:9b",       0.948, 0.837, "High"),
        ("phi4-mini",       0.915, 0.722, "High"),
        ("llama3.2:3b",     0.902, 0.708, "High"),
        ("qwen2.5:3b",      0.890, 0.752, "High"),
        ("llama3.1:8b",     0.724, 0.741, "Mid"),
        ("gemma3:4b",       0.718, 0.718, "Mid"),
        ("mistral:7b",      0.544, 0.625, "Failed"),
        ("qwen2.5:1.5b",    0.508, 0.602, "Failed"),
        ("llama3.2:1b",     0.500, 0.500, "Failed"),
        ("gemma2:2b",       0.500, 0.531, "Failed"),
        ("gemma3:1b",       0.500, 0.500, "Failed"),
    ]
    tier_col = {"High": C_HIGH, "Mid": C_MID, "Failed": C_FAILED}

    pras  = np.array([p[1] for p in points])
    das   = np.array([p[2] for p in points])
    r = np.corrcoef(pras, das)[0, 1]

    fig, ax = plt.subplots(figsize=(11/2.54, 9/2.54))

    # H2 target zone
    ax.fill_between([0.82, 1.05], [0.75, 0.75], [0.95, 0.95],
                    color="#27AE60", alpha=0.10, label="H2 Target Zone")
    ax.text(0.90, 0.76, "H2 Target Zone", fontsize=8, color="#27AE60",
            ha="center", va="bottom", style="italic")

    # Reference lines
    ax.axvline(0.82, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0.75, color="grey", linestyle="--", linewidth=0.8)

    # Points
    for name, pra, d, tier in points:
        ax.scatter(pra, d, color=tier_col[tier], s=80, zorder=5,
                   edgecolors="white", linewidth=0.5)
        offset_x = 0.005 if pra < 0.98 else -0.005
        ha = "left" if offset_x > 0 else "right"
        ax.annotate(name, (pra, d), xytext=(pra + offset_x, d + 0.005),
                    fontsize=6.5, ha=ha, va="bottom", color="black")

    ax.text(0.455, 0.90, f"r = {r:.2f}", fontsize=9, color="black",
            fontweight="bold")

    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel("PRA (Profile Reconstruction Accuracy)")
    ax.set_ylabel("DAS (Dimensional Alignment Score)")
    ax.set_title("PRA vs DAS for All 15 Models", fontsize=11, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.4)

    legend_handles = [
        mpatches.Patch(color=C_HIGH,   label="High tier"),
        mpatches.Patch(color=C_MID,    label="Mid tier"),
        mpatches.Patch(color=C_FAILED, label="Failed tier"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "exp1_pra_das_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# F5-4  exp2_metrics_bar.png  — Grouped bar R0 vs R1 for 5 metrics
# ══════════════════════════════════════════════════════════════════════════════
def f5_4():
    # (metric, R0, R1, cohen_d, sig)
    metrics = [
        ("SCS",        0.261, 0.469, "d=2.37", "***"),
        ("Engagement", 0.649/5, 0.778/5, "d=1.49", "***"),
        ("RR",         0.758/5, 0.757/5, "",      "n.s."),
        ("CR@5",       0.159, 0.155, "d=−0.02", "*"),
        ("ER",         0.340, 0.333, "d=−0.01", "*"),
    ]

    fig, ax = plt.subplots(figsize=(13/2.54, 8/2.54))
    x = np.arange(len(metrics))
    w = 0.35

    r0_vals = [m[1] for m in metrics]
    r1_vals = [m[2] for m in metrics]

    ax.bar(x - w/2, r0_vals, w, color=C_BASELINE, label="R0 (Baseline)",    zorder=3)
    ax.bar(x + w/2, r1_vals, w, color=C_PROPOSED, label="R1 (Personalized)", zorder=3)

    # Significance brackets
    for i, (_, r0, r1, cohen, sig) in enumerate(metrics):
        y_top = max(r0, r1) + 0.02
        bracket_h = 0.008
        ax.plot([i - w/2, i - w/2, i + w/2, i + w/2],
                [y_top, y_top + bracket_h, y_top + bracket_h, y_top],
                lw=1.0, color="black")
        label = f"{cohen} {sig}" if cohen else sig
        ax.text(i, y_top + bracket_h + 0.005, label,
                ha="center", va="bottom", fontsize=7.5,
                color="black" if sig != "n.s." else "grey")

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=10)
    ax.set_ylim(0.0, 0.85)
    ax.set_ylabel("Score (0–1 normalised)")
    ax.set_title("Track A: R0 vs R1 Metric Comparison (n = 5,760 pairs)",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    fig.tight_layout()
    save(fig, "exp2_metrics_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# F5-5  exp2_dim_effect.png  — Grouped horizontal bar, ΔSCS & ΔEngagement
# ══════════════════════════════════════════════════════════════════════════════
def f5_5():
    poles = [
        ("Reflective", 0.237, 0.505),
        ("Visual",     0.233, 0.906),
        ("Sequential", 0.216, 0.599),
        ("Sensing",    0.211, 0.682),
        ("Intuitive",  0.205, 0.604),
        ("Global",     0.200, 0.687),
        ("Verbal",     0.183, 0.380),
        ("Active",     0.178, 0.781),
    ]
    names    = [p[0] for p in poles]
    d_scs    = np.array([p[1] for p in poles])
    d_eng    = np.array([p[2] for p in poles])
    y = np.arange(len(poles))

    fig, ax1 = plt.subplots(figsize=(13/2.54, 8/2.54))
    ax2 = ax1.twiny()

    h = 0.35
    ax1.barh(y - h/2, d_scs, h, color=C_PROPOSED, label="ΔSCS", zorder=3)
    ax2.barh(y + h/2, d_eng, h, color=C_ORANGE, label="ΔEngagement (raw)", zorder=3)

    # Mean ΔSCS line
    ax1.axvline(0.208, color=C_PROPOSED, linestyle="--", linewidth=1.0)
    ax1.text(0.209, len(poles) - 0.3, "Mean ΔSCS", fontsize=7.5,
             color=C_PROPOSED, va="top")

    # Top value labels
    ax1.text(d_scs[0] + 0.003, y[0] - h/2,  "Reflective: +0.237",
             va="center", fontsize=7.5, color=C_PROPOSED)
    ax2.text(d_eng[1] + 0.02,  y[1] + h/2,  "Visual: +0.906",
             va="center", fontsize=7.5, color=C_ORANGE)

    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlim(0, 0.32)
    ax1.set_xlabel("ΔSCS (R1 − R0)", color=C_PROPOSED)
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("ΔEngagement (R1 − R0, raw scale)", color=C_ORANGE)
    ax1.set_title("Per-Pole Improvement: ΔSCS and ΔEngagement",
                  fontsize=10, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "exp2_dim_effect.png")


# ══════════════════════════════════════════════════════════════════════════════
# F5-6  exp2_pairwise_wtl.png  — Stacked horizontal bar W/T/L
# ══════════════════════════════════════════════════════════════════════════════
def f5_6():
    pairwise_path = os.path.join(REPO,
        "experiments/exp2_tutor_personalization/results/pairwise/summary_overall.json")
    with open(pairwise_path) as f:
        d = json.load(f)
    win   = d["n_r1_win"]
    tie   = d["n_tie"]
    loss  = d["n_r0_win"]
    total = d["n_total"]

    fig, ax = plt.subplots(figsize=(12/2.54, 4/2.54))

    left = 0
    segs = [
        (win,  C_HIGH,       "R1 Win"),
        (tie,  "#BDC3C7",    "Tie"),
        (loss, C_FAILED,     "R0 Win"),
    ]
    for count, color, _ in segs:
        ax.barh(0, count, left=left, color=color, height=0.5, zorder=3)
        if count > 0:
            pct = count / total * 100
            ax.text(left + count / 2, 0, f"{count:,} ({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
        left += count

    ax.set_xlim(0, total)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel(f"Number of judgements (n = {total:,})")
    ax.set_title("Track B: Pairwise Judgement — R1 vs R0 (n = 5,760)",
                 fontsize=10, fontweight="bold")

    legend_handles = [mpatches.Patch(color=c, label=l) for _, c, l in segs if _ > 0 or l == "Tie"]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.0, -0.25), ncol=3)

    ax.text(0.5, -0.52, "Binomial test (ties excluded): p < 0.001 | Judge: GPT-4o, position-randomised",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="grey")

    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    save(fig, "exp2_pairwise_wtl.png")


# ══════════════════════════════════════════════════════════════════════════════
# F6-1  ana_scs_distribution.png  — Violin plot of SCS distributions
# ══════════════════════════════════════════════════════════════════════════════
def f6_1():
    csv_path = os.path.join(REPO,
        "experiments/exp2_tutor_personalization/results/exp2_session_metrics.csv")
    r0_scs, r1_scs = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            val = float(row["scs"])
            if row["mode"] == "R0":
                r0_scs.append(val)
            else:
                r1_scs.append(val)
    r0 = np.array(r0_scs)
    r1 = np.array(r1_scs)

    fig, ax = plt.subplots(figsize=(11/2.54, 8/2.54))

    vp = ax.violinplot([r0, r1], positions=[1, 2], widths=0.6,
                       showmedians=False, showextrema=False)
    vp["bodies"][0].set_facecolor(C_BASELINE)
    vp["bodies"][0].set_alpha(0.7)
    vp["bodies"][1].set_facecolor(C_PROPOSED)
    vp["bodies"][1].set_alpha(0.7)

    # Inner box plot
    for pos, data, color in [(1, r0, C_BASELINE), (2, r1, C_PROPOSED)]:
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        ax.plot([pos, pos], [q1, q3], color="black", lw=3, zorder=4)
        ax.scatter(pos, med, color="white", s=30, zorder=5, edgecolors="black", lw=1)

    # Jitter overlay (500-pt subsample)
    rng = np.random.default_rng(42)
    for pos, data in [(1, r0), (2, r1)]:
        idx = rng.choice(len(data), 500, replace=False)
        sub = data[idx]
        jitter = rng.uniform(-0.12, 0.12, 500)
        ax.scatter(pos + jitter, sub, alpha=0.05, s=4, color="black", zorder=2)

    # Mean annotations
    for pos, data, color in [(1, r0, C_BASELINE), (2, r1, C_PROPOSED)]:
        m, s = data.mean(), data.std()
        ax.annotate(f"μ={m:.3f}\nσ={s:.3f}", xy=(pos, m),
                    xytext=(pos + 0.28, m), fontsize=8, color=color,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    # R0 mean reference line
    ax.axhline(r0.mean(), color=C_BASELINE, linestyle="--", linewidth=1.0, alpha=0.8)

    # Delta annotation
    ax.text(1.5, max(r1.mean(), r0.mean()) + 0.05,
            "Δ mean = +0.208, d = 2.37",
            ha="center", fontsize=9, color="black", fontweight="bold")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["R0 (Baseline)", "R1 (Personalized)"], fontsize=10)
    ax.set_ylabel("Style Conformance Score (SCS)")
    ax.set_title("SCS Distribution: R0 vs R1 (n = 5,760 each)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save(fig, "ana_scs_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# F6-2  ana_retrieval_style_tradeoff.png  — Scatter CR@5 vs SCS per profile
# ══════════════════════════════════════════════════════════════════════════════
def f6_2():
    csv_path = os.path.join(REPO,
        "experiments/exp2_tutor_personalization/results/exp2_session_metrics.csv")

    from collections import defaultdict
    agg = defaultdict(lambda: {"scs": [], "cr5": []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["profile_label"], row["mode"])
            agg[key]["scs"].append(float(row["scs"]))
            agg[key]["cr5"].append(float(row["cr5"]))

    profiles = sorted(set(k[0] for k in agg))
    r0_pts, r1_pts = [], []
    for p in profiles:
        r0s = agg[(p, "R0")]
        r1s = agg[(p, "R1")]
        r0_pts.append((np.mean(r0s["cr5"]), np.mean(r0s["scs"]), p))
        r1_pts.append((np.mean(r1s["cr5"]), np.mean(r1s["scs"]), p))

    # Reference midpoints
    mid_cr5 = 0.157
    mid_scs  = 0.365

    fig, ax = plt.subplots(figsize=(12/2.54, 9/2.54))

    # Quadrant shading
    xl, xr = 0.12, 0.20
    yl, yr = 0.20, 0.58
    ax.fill_between([mid_cr5, xr], [mid_scs, mid_scs], [yr, yr],
                    color="lightgreen", alpha=0.25, label="Ideal")
    ax.fill_between([xl, mid_cr5], [mid_scs, mid_scs], [yr, yr],
                    color="lightyellow", alpha=0.6, label="Style gain, retrieval cost")
    ax.fill_between([mid_cr5, xr], [yl, yl], [mid_scs, mid_scs],
                    color="lightblue", alpha=0.35, label="Factual, no style")
    ax.fill_between([xl, mid_cr5], [yl, yl], [mid_scs, mid_scs],
                    color="#FDEDEC", alpha=0.6, label="Underperforming")

    ax.text(0.179, yr - 0.01,  "Ideal",                    fontsize=7.5, ha="right", color="darkgreen")
    ax.text(0.122, yr - 0.01,  "Style gain,\nretrieval cost", fontsize=7, ha="left", color="olive")
    ax.text(0.179, yl + 0.005, "Factual, no style",         fontsize=7.5, ha="right", color="steelblue")
    ax.text(0.122, yl + 0.005, "Underperforming",           fontsize=7.5, ha="left", color="red")

    # Reference lines
    ax.axvline(mid_cr5, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(mid_scs,  color="grey", linestyle="--", linewidth=0.8)

    # Arrows R0 → R1 and points
    for (cr5_0, scs_0, name), (cr5_1, scs_1, _) in zip(r0_pts, r1_pts):
        ax.annotate("", xy=(cr5_1, scs_1), xytext=(cr5_0, scs_0),
                    arrowprops=dict(arrowstyle="-|>", color="darkblue",
                                   lw=0.8, mutation_scale=8))
        ax.scatter(cr5_0, scs_0, color=C_BASELINE, s=40, zorder=5,
                   edgecolors="white", lw=0.5)
        ax.scatter(cr5_1, scs_1, color=C_PROPOSED, s=40, zorder=5,
                   edgecolors="white", lw=0.5)

    ax.set_xlim(xl, xr)
    ax.set_ylim(yl, yr)
    ax.set_xlabel("Mean CR@5 (Contextual Relevance)")
    ax.set_ylabel("Mean SCS (Style Conformance Score)")
    ax.set_title("Per-Profile Retrieval–Style Trade-off: R0 → R1",
                 fontsize=10, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=C_BASELINE, label="R0 (Baseline)"),
        mpatches.Patch(color=C_PROPOSED, label="R1 (Personalized)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.grid(linestyle=":", alpha=0.3)
    fig.tight_layout()
    save(fig, "ana_retrieval_style_tradeoff.png")


# ══════════════════════════════════════════════════════════════════════════════
# F7-1  disc_hypothesis_dashboard.png  — 3-col hypothesis summary table
# ══════════════════════════════════════════════════════════════════════════════
def f7_1():
    fig, ax = plt.subplots(figsize=(14/2.54, 7/2.54))
    ax.set_axis_off()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 6)

    col_headers = ["H2 (Exp 1)\nAgent Fidelity", "H1 (Exp 2)\nPersonalization", "H3 (Exp 3)\nMCP Tool Sel."]
    statuses    = ["✅ EXCEEDED",  "⚠️ PARTIAL",     "✅ CONFIRMED"]
    status_cols = [C_HIGH,         C_MID,             C_HIGH]
    metric_rows = [
        ["PRA = 1.000 (top models)",  "SCS d = 2.37 ✅",   "TSA +15.8 pp ✅"],
        ["8/15 exceed target",         "Eng d = 1.49 ✅",   "PTS = 93% ✅"],
        ["",                           "CR@5 d = −0.02 ❌", "S1a ≈ S0 ⚠️"],
        ["",                           "RR: no change ✅",  ""],
    ]
    rq_row = ["RQ2 ✅ Answered", "RQ1 ✅ Answered", "RQ3 ✅ Answered"]

    col_x = [0.5, 1.5, 2.5]

    # Column headers
    for i, (hdr, cx) in enumerate(zip(col_headers, col_x)):
        ax.text(cx, 5.7, hdr, ha="center", va="center", fontsize=9,
                fontweight="bold", color=C_PROPOSED)

    # Divider below headers
    ax.axhline(5.3, color="black", linewidth=0.8)

    # Status badges
    for i, (status, color, cx) in enumerate(zip(statuses, status_cols, col_x)):
        badge = mpatches.FancyBboxPatch((cx - 0.42, 4.88), 0.84, 0.32,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor="none")
        ax.add_patch(badge)
        ax.text(cx, 5.04, status, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")

    ax.axhline(4.8, color="black", linewidth=0.8)

    # Metric rows
    metric_y_start = 4.45
    row_h = 0.55
    for row_i, row in enumerate(metric_rows):
        y = metric_y_start - row_i * row_h
        for col_i, cell in enumerate(row):
            color = "black"
            if "✅" in cell:
                color = "#1A7A4A"
            elif "❌" in cell:
                color = C_FAILED
            elif "⚠️" in cell:
                color = "#B7770D"
            ax.text(col_x[col_i], y, cell, ha="center", va="center",
                    fontsize=8, color=color)
        ax.axhline(y - row_h / 2 + 0.02, color="#D5D8DC", linewidth=0.5)

    # RQ answered row
    rq_y = metric_y_start - len(metric_rows) * row_h + 0.1
    ax.axhline(rq_y + 0.25, color="black", linewidth=0.8)
    for cx, rq in zip(col_x, rq_row):
        bg = mpatches.FancyBboxPatch((cx - 0.45, rq_y - 0.22), 0.9, 0.38,
                                       boxstyle="round,pad=0.04",
                                       facecolor="#ECF0F1", edgecolor="none")
        ax.add_patch(bg)
        ax.text(cx, rq_y - 0.03, rq, ha="center", va="center",
                fontsize=8, color="#2C3E50", fontweight="bold")

    # Column dividers
    ax.axvline(1.0, color="#D5D8DC", linewidth=0.8)
    ax.axvline(2.0, color="#D5D8DC", linewidth=0.8)

    # Outer border
    border = mpatches.FancyBboxPatch((0.02, rq_y - 0.30), 2.96,
                                      6.0 - (rq_y - 0.30),
                                      boxstyle="round,pad=0.05",
                                      facecolor="none", edgecolor="black",
                                      linewidth=1.2)
    ax.add_patch(border)

    ax.set_title("Hypothesis Outcomes Across All Three Experiments",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "disc_hypothesis_dashboard.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    f5_1(); f5_2(); f5_3()
    f5_4(); f5_5(); f5_6()
    f6_1(); f6_2()
    f7_1()
    print("Done. All 9 figures written to figures/")
