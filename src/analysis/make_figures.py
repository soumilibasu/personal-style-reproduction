"""
src/analysis/make_figures.py

WP1: regenerate the paper's key figures (and a few extras) from the saved
analysis outputs, so the reproduction has the same visual evidence the
paper presents.

Figures written to reports/figures/:
    fig_h1a_h1b_before_after.png   - similarity to self / LLM before vs after
    fig_h1c_self_vs_llm.png        - edited text: similarity to self vs LLM
    fig_h1a_prime_own_vs_other.png - edited: own control vs others' control
    fig_h2_homogeneity.png         - between-author similarity by group
    fig_h3_perception_vs_luar.png  - perceived vs measured self-similarity
    fig_effect_sizes.png           - our Hedges' g vs the paper's, with CIs
    fig_llm_markers_by_label.png   - LLM cue rates by text type (WP2)

Run from the repository root (after analyze_hypotheses.py and
build_stylometry.py):
    python src/analysis/make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path("data/processed")
HYP_DIR = DATA_DIR / "hypotheses"
FIG_DIR = Path("reports/figures")

BLUE, ORANGE, GREEN = "#2c6fbb", "#e07b39", "#3a9b7a"


def savefig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_before_after(tasks):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (bx, ay, title, ref) in zip(axes, [
        ("draft_to_llm", "edited_to_llm",
         "Similarity to LLM text (H1b)", "LLM-generated reference"),
        ("draft_to_control", "edited_to_control",
         "Similarity to own control (H1a)", "human-written reference"),
    ]):
        ax.scatter(tasks[bx], tasks[ay], s=14, alpha=0.5, color=BLUE,
                   edgecolor="none")
        lo = min(tasks[bx].min(), tasks[ay].min())
        hi = max(tasks[bx].max(), tasks[ay].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change")
        ax.set_xlabel(f"before post-editing\n({ref})")
        ax.set_ylabel("after post-editing")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Before vs after post-editing", fontweight="bold")
    savefig(fig, "fig_h1a_h1b_before_after.png")


def fig_h1c(tasks):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(tasks["edited_to_llm"], tasks["edited_to_own_control"],
               s=16, alpha=0.55, color=ORANGE, edgecolor="none",
               label="post-edited text")
    lo = min(tasks["edited_to_llm"].min(), tasks["edited_to_own_control"].min())
    hi = max(tasks["edited_to_llm"].max(), tasks["edited_to_own_control"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="equally similar")
    ax.set_xlabel("similarity to LLM-generated text")
    ax.set_ylabel("similarity to own control text")
    ax.set_title("H1c: post-edited text stays closer to LLM than to self")
    ax.legend(fontsize=8)
    savefig(fig, "fig_h1c_self_vs_llm.png")


def fig_h1a_prime(tasks):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(tasks["edited_to_other_controls"],
               tasks["edited_to_own_control"],
               s=16, alpha=0.55, color=GREEN, edgecolor="none")
    lo = min(tasks["edited_to_other_controls"].min(),
             tasks["edited_to_own_control"].min())
    hi = max(tasks["edited_to_other_controls"].max(),
             tasks["edited_to_own_control"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="equally similar")
    ax.set_xlabel("similarity to OTHER participants' control")
    ax.set_ylabel("similarity to OWN control")
    ax.set_title("H1a': post-edited text is closer to the author's own style")
    ax.legend(fontsize=8)
    savefig(fig, "fig_h1a_prime_own_vs_other.png")


def fig_homogeneity(pairs):
    order = [("control", "Human control"), ("edited", "Post-edited"),
             ("llm_draft", "LLM-generated")]
    data = [pairs.loc[pairs["group"] == g, "similarity"].to_numpy()
            for g, _ in order]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, c in zip(parts["bodies"], [GREEN, ORANGE, BLUE]):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([lbl for _, lbl in order])
    ax.set_ylabel("between-author stylistic similarity")
    ax.set_title("H2: stylistic homogeneity by text type\n"
                 "(higher = more homogeneous / less diverse)")
    savefig(fig, "fig_h2_homogeneity.png")


def fig_h3(h3):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(h3["perceived_scaled_0_1"], h3["luar_self_sim"],
               s=14, alpha=0.4, color=BLUE, edgecolor="none")
    # Per-participant within-subject lines (repeated-measures view).
    for _, grp in h3.groupby("participant_id"):
        if len(grp) >= 2 and grp["perceived_scaled_0_1"].nunique() >= 2:
            coeffs = np.polyfit(grp["perceived_scaled_0_1"],
                                grp["luar_self_sim"], 1)
            xs = np.array([grp["perceived_scaled_0_1"].min(),
                           grp["perceived_scaled_0_1"].max()])
            ax.plot(xs, np.polyval(coeffs, xs), color=ORANGE, lw=0.6,
                    alpha=0.3)
    ax.set_xlabel("perceived self-similarity (scaled 0-1)")
    ax.set_ylabel("LUAR self-similarity")
    ax.set_title("H3: perceived vs measured self-similarity\n"
                 "(orange = per-participant repeated-measures lines)")
    savefig(fig, "fig_h3_perception_vs_luar.png")


def fig_effect_sizes(summary):
    summary = summary.dropna(subset=["hedges_g"]).copy()
    labels = summary["hypothesis"].tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.8))

    ours = summary["hedges_g"].to_numpy()
    lo = summary["g_ci_95_low"].to_numpy()
    hi = summary["g_ci_95_high"].to_numpy()
    ax.errorbar(ours, y - 0.12, xerr=[ours - lo, hi - ours], fmt="o",
                color=BLUE, capsize=3, label="ours")

    paper = summary["paper_g"].to_numpy(dtype=float)
    mask = np.isfinite(paper)
    ax.scatter(paper[mask], y[mask] + 0.12, color=ORANGE, marker="s",
               label="paper", zorder=3)

    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Hedges' g (95% CI)")
    ax.set_title("Effect sizes: ours vs the paper")
    ax.legend(fontsize=8)
    savefig(fig, "fig_effect_sizes.png")


def fig_llm_markers():
    path = DATA_DIR / "features" / "stylometry_features.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = ["control", "draft", "edited"]
    means = df.groupby("label")[["llm_marker_rate_per100w", "emdash_rate"]
                                ].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, means["llm_marker_rate_per100w"], w,
           color=BLUE, label="LLM marker words / 100w")
    ax.bar(x + w / 2, means["emdash_rate"], w, color=ORANGE,
           label="em-dashes / 100 chars")
    ax.set_xticks(x)
    ax.set_xticklabels(["Human control", "LLM draft", "Post-edited"])
    ax.set_ylabel("rate")
    ax.set_title("LLM stylistic cues by text type (WP2)")
    ax.legend(fontsize=8)
    savefig(fig, "fig_llm_markers_by_label.png")


def main():
    print("Rendering figures to reports/figures/ ...")
    tasks = pd.read_csv(HYP_DIR / "h1_task_scores.csv")
    fig_before_after(tasks)
    fig_h1c(tasks)
    fig_h1a_prime(tasks)

    pairs = pd.read_csv(HYP_DIR / "h2_homogeneity_pairs.csv")
    fig_homogeneity(pairs)

    h3 = pd.read_csv(HYP_DIR / "h3_perception_vs_measure.csv")
    fig_h3(h3)

    summary = pd.read_csv(HYP_DIR / "all_hypotheses_summary.csv")
    fig_effect_sizes(summary)

    fig_llm_markers()
    print("Done.")


if __name__ == "__main__":
    main()
