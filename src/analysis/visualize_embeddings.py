"""
src/analysis/visualize_embeddings.py

WP5 (part 1): visualise the LUAR embedding space and per-scenario effects.

- 2-D PCA and t-SNE projections of all texts, coloured by type
  (human control / LLM draft / post-edited). This shows visually whether
  post-edited text forms its own region -- the "third variety" of H2.
- Per-scenario mean style recovery (change_after_editing), to show which
  writing tasks were easiest/hardest to personalise.

Outputs:
    reports/figures/fig_embedding_pca.png
    reports/figures/fig_embedding_tsne.png
    reports/figures/fig_recovery_by_scenario.png
    data/processed/analysis/recovery_by_scenario.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/processed")
HYP_DIR = DATA_DIR / "hypotheses"
OUT_DIR = DATA_DIR / "analysis"
FIG_DIR = Path("reports/figures")
SEED = 42

COLORS = {"control": "#3a9b7a", "draft": "#2c6fbb", "edited": "#e07b39"}
NAMES = {"control": "human control", "draft": "LLM draft",
         "edited": "post-edited"}
CLASS_MAP = {"control_reference": "control", "ai_draft": "draft",
             "edited_text": "edited"}


def load_embeddings():
    with np.load(DATA_DIR / "luar_embeddings.npz", allow_pickle=False) as a:
        X = a["embeddings"].astype(np.float64)
        keys = a["embedding_keys"].astype(str)
    index = pd.read_csv(DATA_DIR / "luar_embedding_index.csv",
                        dtype={"embedding_key": str, "participant_id": str})
    assert np.array_equal(keys, index["embedding_key"].to_numpy())
    labels = index["text_type"].map(CLASS_MAP).to_numpy()
    return X, labels


def scatter_projection(coords, labels, title, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for cls in ["control", "draft", "edited"]:
        m = labels == cls
        ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.6,
                   color=COLORS[cls], edgecolor="none", label=NAMES[cls])
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, labels = load_embeddings()
    Xs = StandardScaler().fit_transform(X)

    print("Projecting embeddings ...")
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    scatter_projection(pca, labels, "LUAR embeddings (PCA)",
                       FIG_DIR / "fig_embedding_pca.png")

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30,
                init="pca", learning_rate="auto").fit_transform(Xs)
    scatter_projection(tsne, labels, "LUAR embeddings (t-SNE)",
                       FIG_DIR / "fig_embedding_tsne.png")

    # Per-scenario recovery.
    tasks = pd.read_csv(HYP_DIR / "h1_task_scores.csv")
    by_scen = tasks.groupby("scenario")["change_after_editing"].agg(
        ["mean", "sem", "count"]).sort_values("mean", ascending=False)
    by_scen.to_csv(OUT_DIR / "recovery_by_scenario.csv")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(range(len(by_scen)), by_scen["mean"],
           yerr=by_scen["sem"], color="#2c6fbb", capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(by_scen)))
    ax.set_xticklabels(by_scen.index, rotation=35, ha="right")
    ax.set_ylabel("mean style recovery (edited - draft)")
    ax.set_title("Style recovery by writing scenario")
    fig.savefig(FIG_DIR / "fig_recovery_by_scenario.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR / 'fig_recovery_by_scenario.png'}")

    print("\nRecovery by scenario (mean change_after_editing):")
    print(by_scen.round(4).to_string())


if __name__ == "__main__":
    main()
