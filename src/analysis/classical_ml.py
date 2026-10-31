"""
src/analysis/classical_ml.py

WP3: a classical machine-learning model zoo for two style tasks, run on
two different feature representations.

Tasks:
  3way    : classify text as human control / LLM draft / post-edited
  detect  : binary AI-detection, human control (0) vs LLM draft (1)

Feature sets:
  embeddings : 512-dim LUAR embeddings (control is one combined vector
               per participant, so 81 / 324 / 324 texts)
  stylometry : 120 interpretable features from build_stylometry.py
               (control texts are individual, so 162 / 324 / 324 texts)

Models: logistic regression, linear SVM, RBF SVM, random forest,
gradient boosting, k-NN, Gaussian naive Bayes. Everything is evaluated
out-of-fold with GroupKFold by participant (no author in both train and
test).

Outputs:
    data/processed/analysis/classical_ml_model_comparison.csv
    reports/figures/fig_roc_ai_detection.png
    reports/figures/fig_confusion_best_3way.png
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "analysis"
FIG_DIR = Path("reports/figures")
SEED = 42
N_SPLITS = 5

CLASS_MAP = {"control_reference": "control", "ai_draft": "draft",
             "edited_text": "edited"}


def models():
    def pipe(clf):
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    return {
        "logreg": pipe(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED)),
        "linear_svm": pipe(SVC(
            kernel="linear", class_weight="balanced", random_state=SEED)),
        "rbf_svm": pipe(SVC(
            kernel="rbf", class_weight="balanced", random_state=SEED)),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1)),
        "gradient_boosting": pipe(GradientBoostingClassifier(
            random_state=SEED)),
        "knn": pipe(KNeighborsClassifier(n_neighbors=15)),
        "naive_bayes": pipe(GaussianNB()),
    }


def load_embeddings():
    with np.load(DATA_DIR / "luar_embeddings.npz", allow_pickle=False) as a:
        X = a["embeddings"].astype(np.float64)
        keys = a["embedding_keys"].astype(str)
    index = pd.read_csv(DATA_DIR / "luar_embedding_index.csv",
                        dtype={"embedding_key": str, "participant_id": str})
    assert np.array_equal(keys, index["embedding_key"].to_numpy())
    y = index["text_type"].map(CLASS_MAP).to_numpy()
    groups = index["participant_id"].to_numpy()
    return X, y, groups


def load_stylometry():
    df = pd.read_csv(DATA_DIR / "features" / "stylometry_features.csv",
                     dtype={"participant_id": str})
    feat_cols = [c for c in df.columns
                 if c not in ("participant_id", "task_key", "scenario",
                              "label")]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    groups = df["participant_id"].to_numpy()
    return X, y, groups


def scores_for(model, X, y, groups):
    """OOF predicted probability/score for the positive class (binary)."""
    cv = GroupKFold(n_splits=N_SPLITS)
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = cross_val_predict(model, X, y, groups=groups, cv=cv,
                                  method="predict_proba")
        return proba[:, 1]
    # SVM without probability: use the decision function.
    return cross_val_predict(model, X, y, groups=groups, cv=cv,
                             method="decision_function")


def evaluate(X, y, groups, task):
    cv = GroupKFold(n_splits=N_SPLITS)
    rows = []
    roc_data = {}
    preds_by_model = {}
    for name, model in models().items():
        preds = cross_val_predict(model, X, y, groups=groups, cv=cv)
        row = {
            "task": task,
            "model": name,
            "accuracy": float(accuracy_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "macro_f1": float(f1_score(y, preds, average="macro")),
        }
        preds_by_model[name] = preds
        if task.startswith("detect"):
            score = scores_for(model, X, y, groups)
            y_bin = (y == "draft").astype(int)
            row["roc_auc"] = float(roc_auc_score(y_bin, score))
            roc_data[name] = (y_bin, score)
        rows.append(row)
    return rows, roc_data, preds_by_model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    feature_sets = {
        "embeddings": load_embeddings(),
        "stylometry": load_stylometry(),
    }

    all_rows = []
    roc_for_plot = {}
    best_3way = None  # (feature_set, model, y, preds, balanced_acc)

    for fs_name, (X, y, groups) in feature_sets.items():
        # 3-way task.
        rows, _, preds_by = evaluate(X, y, groups, "3way")
        for r in rows:
            r["feature_set"] = fs_name
        all_rows.extend(rows)
        # Track best 3-way model for a confusion matrix.
        for r in rows:
            if best_3way is None or r["balanced_accuracy"] > best_3way[4]:
                best_3way = (fs_name, r["model"], y,
                             preds_by[r["model"]], r["balanced_accuracy"])

        # Binary detection task (control vs draft only).
        mask = np.isin(y, ["control", "draft"])
        rows_d, roc_d, _ = evaluate(X[mask], y[mask], groups[mask],
                                    f"detect")
        for r in rows_d:
            r["feature_set"] = fs_name
        all_rows.extend(rows_d)
        roc_for_plot[fs_name] = roc_d

    comparison = pd.DataFrame(all_rows)[
        ["task", "feature_set", "model", "accuracy", "balanced_accuracy",
         "macro_f1", "roc_auc"]]
    comparison = comparison.sort_values(
        ["task", "feature_set", "balanced_accuracy"],
        ascending=[True, True, False])
    comparison.to_csv(OUT_DIR / "classical_ml_model_comparison.csv",
                      index=False)

    # ---- ROC curves for AI detection (stylometry feature set) -----------
    fig, ax = plt.subplots(figsize=(6, 5.4))
    for name, (y_bin, score) in roc_for_plot["stylometry"].items():
        fpr, tpr, _ = roc_curve(y_bin, score)
        auc = roc_auc_score(y_bin, score)
        ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("AI detection (human vs LLM) — stylometry features")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(FIG_DIR / "fig_roc_ai_detection.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Confusion matrix for best 3-way model --------------------------
    fs_name, model_name, y_true, y_pred, bacc = best_3way
    labels = ["control", "draft", "edited"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Best 3-way model: {model_name} ({fs_name})\n"
                 f"balanced acc = {bacc:.3f}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.savefig(FIG_DIR / "fig_confusion_best_3way.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Console --------------------------------------------------------
    print("=== Classical ML model zoo (GroupKFold by participant) ===\n")
    for task in ["3way", "detect"]:
        print(f"-- task: {task} --")
        sub = comparison[comparison["task"] == task]
        cols = ["feature_set", "model", "balanced_accuracy", "macro_f1"]
        if task == "detect":
            cols.append("roc_auc")
        print(sub[cols].to_string(index=False))
        print()
    print(f"Best 3-way: {model_name} on {fs_name} "
          f"(balanced acc {bacc:.3f}, chance 0.333)")
    print("\nSaved:")
    print("  data/processed/analysis/classical_ml_model_comparison.csv")
    print("  reports/figures/fig_roc_ai_detection.png")
    print("  reports/figures/fig_confusion_best_3way.png")


if __name__ == "__main__":
    main()
