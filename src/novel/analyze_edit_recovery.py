"""
src/novel/analyze_edit_recovery.py

NOVEL CONTRIBUTION (step 2 of 2): do editing behaviours predict how much
an LLM draft recovers the author's personal style?

Target
------
`change_after_editing` = cosine(edited, own control) - cosine(draft, own
control), i.e. how far post-editing moved the text toward the author's own
unassisted style in LUAR space. This is exactly the quantity H1a aggregates;
here we predict it per task from editing features.

What we test
------------
We compare nested models under participant-grouped cross-validation
(GroupKFold, so no participant appears in both train and test):

    M0  controls only        : scenario + draft length
    M1  controls + QUANTITY   : how MUCH was edited (edit count, chars, time)
    M2  controls + KIND       : what KIND of edit (words/sentences/punct/voice)
    M3  controls + QUANTITY + KIND (full)

The headline question is whether KIND (M2) explains style recovery better
than QUANTITY (M1). The paper's framing predicts that raw editing volume
is a weak predictor while specific edit types matter more.

We also report:
  * Spearman correlation of every feature with the target.
  * Standardised Ridge coefficients and permutation importance from the
    full model, to rank which behaviours matter.

Run from the repository root:
    python src/novel/analyze_edit_recovery.py

Inputs:
    data/processed/novel/edit_features.csv            (from step 1)
    data/processed/hypotheses/h1_task_scores.csv      (preferred target)
    data/processed/h1a_task_similarities.csv          (fallback target)

Outputs:
    data/processed/novel/feature_target_correlations.csv
    data/processed/novel/model_comparison.csv
    data/processed/novel/feature_importance.csv
    data/processed/novel/novel_report.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "novel"

TARGET = "change_after_editing"
SEED = 42
N_SPLITS = 5

# Ridge penalty candidates. The strength is tuned by RidgeCV *inside* each
# training fold, so a model with many collinear features (M3) can regularise
# them away instead of over-fitting; this keeps the nested-model comparison
# fair and stops the full model from collapsing purely for lack of shrinkage.
ALPHAS = np.logspace(-2, 3, 30)

# Feature groups ------------------------------------------------------------
CONTROL_NUMERIC = ["len_draft_chars"]
CONTROL_CATEGORICAL = ["scenario"]

QUANTITY_FEATURES = [
    "n_edits",
    "chars_inserted",
    "chars_deleted",
    "net_char_change_log",
    "editing_time_s",
    "edits_per_100_draft_chars",
]

KIND_FEATURES = [
    "words_inserted",
    "words_deleted",
    "words_replaced",
    "frac_words_changed",
    "draft_final_similarity",
    "sent_count_change",
    "avg_sentlen_change",
    "ttr_change",
    "punct_count_change",
    "comma_change",
    "contraction_change",
    "first_person_change",
    "uppercase_ratio_change",
    "len_change_ratio",
]


def load_data():
    feats_path = OUT_DIR / "edit_features.csv"
    if not feats_path.exists():
        raise FileNotFoundError(
            f"Missing {feats_path}. Run src/novel/extract_edit_features.py first."
        )
    feats = pd.read_csv(feats_path, dtype={"participant_id": str,
                                           "task_key": str})

    preferred = DATA_DIR / "hypotheses" / "h1_task_scores.csv"
    fallback = DATA_DIR / "h1a_task_similarities.csv"
    target_path = preferred if preferred.exists() else fallback
    if not target_path.exists():
        raise FileNotFoundError(
            "No target file found. Run src/analysis/analyze_hypotheses.py "
            "(or analyze_h1a.py) first."
        )

    target = pd.read_csv(target_path, dtype={"participant_id": str,
                                             "task_key": str})
    target = target[["participant_id", "task_key", TARGET]]

    df = feats.merge(target, on=["participant_id", "task_key"], how="inner")
    if df.empty:
        raise ValueError("Features and target did not merge on any task.")

    # Impute the only feature that can be missing (editing time) with its
    # median; everything else is defined for every task.
    if df["editing_time_s"].isna().any():
        df["editing_time_s"] = df["editing_time_s"].fillna(
            df["editing_time_s"].median()
        )

    print(f"Merged tasks: {len(df)} "
          f"({df['participant_id'].nunique()} participants, target from "
          f"{target_path.name})")
    return df


def make_pipeline(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ))
    pre = ColumnTransformer(transformers)
    return Pipeline([("pre", pre), ("model", RidgeCV(alphas=ALPHAS))])


def grouped_r2(df, numeric_features, categorical_features):
    """Out-of-fold R^2/MAE, per-fold R^2, and per-fold chosen alpha.

    GroupKFold keeps every participant in a single fold. Within each training
    fold RidgeCV selects the penalty by internal cross-validation, so the
    regularisation strength is chosen without ever seeing the held-out
    participants. Per-fold R^2 is reported so the stability of each model
    (not just the pooled number) can be read directly.
    """
    X = df[numeric_features + categorical_features]
    y = df[TARGET].to_numpy()
    groups = df["participant_id"].to_numpy()
    cv = GroupKFold(n_splits=N_SPLITS)

    oof = np.full(len(y), np.nan)
    fold_r2, fold_alphas = [], []
    for train_idx, test_idx in cv.split(X, y, groups):
        pipe = make_pipeline(numeric_features, categorical_features)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        # Per-fold R^2 uses the TRAIN mean as the baseline (honest OOF).
        ss_res = float(np.sum((y[test_idx] - pred) ** 2))
        ss_tot = float(np.sum((y[test_idx] - np.mean(y[train_idx])) ** 2))
        fold_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        fold_alphas.append(float(pipe.named_steps["model"].alpha_))

    ss_res = float(np.sum((y - oof) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y - oof)))
    return r2, mae, oof, fold_r2, fold_alphas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = load_data()

    all_numeric = CONTROL_NUMERIC + QUANTITY_FEATURES + KIND_FEATURES

    # ----- 1. Univariate Spearman correlations with the target -----------
    corr_rows = []
    for feat in QUANTITY_FEATURES + KIND_FEATURES:
        r, p = stats.spearmanr(df[feat], df[TARGET])
        group = "quantity" if feat in QUANTITY_FEATURES else "kind"
        corr_rows.append({
            "feature": feat,
            "group": group,
            "spearman_r": float(r),
            "spearman_p": float(p),
            "abs_r": abs(float(r)),
        })
    corr = pd.DataFrame(corr_rows).sort_values("abs_r", ascending=False)
    corr.to_csv(OUT_DIR / "feature_target_correlations.csv", index=False)

    # ----- 2. Nested model comparison (grouped CV) -----------------------
    models = {
        "M0_controls": (CONTROL_NUMERIC, CONTROL_CATEGORICAL),
        "M1_controls+quantity": (
            CONTROL_NUMERIC + QUANTITY_FEATURES, CONTROL_CATEGORICAL),
        "M2_controls+kind": (
            CONTROL_NUMERIC + KIND_FEATURES, CONTROL_CATEGORICAL),
        "M3_full": (all_numeric, CONTROL_CATEGORICAL),
    }

    model_rows = []
    fold_r2_by_model = {}
    for name, (num_f, cat_f) in models.items():
        r2, mae, _, fold_r2, fold_alphas = grouped_r2(df, num_f, cat_f)
        fold_r2_by_model[name] = fold_r2
        model_rows.append({
            "model": name,
            "n_numeric_features": len(num_f),
            "cv_r2": r2,
            "cv_mae": mae,
            "fold_r2_min": float(np.nanmin(fold_r2)),
            "fold_r2_max": float(np.nanmax(fold_r2)),
            "fold_r2_std": float(np.nanstd(fold_r2)),
            "median_alpha": float(np.median(fold_alphas)),
        })
    model_comparison = pd.DataFrame(model_rows)
    model_comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # ----- 3. Importance from the full model -----------------------------
    # Standardised Ridge coefficients (numeric features only, so the
    # coefficients are directly comparable).
    X = df[all_numeric + CONTROL_CATEGORICAL]
    y = df[TARGET].to_numpy()
    groups = df["participant_id"].to_numpy()

    full_pipe = make_pipeline(all_numeric, CONTROL_CATEGORICAL)
    full_pipe.fit(X, y)

    ridge = full_pipe.named_steps["model"]
    full_model_alpha = float(ridge.alpha_)
    feature_names = full_pipe.named_steps["pre"].get_feature_names_out()
    coef_by_name = dict(zip(feature_names, ridge.coef_))

    # Permutation importance via out-of-fold predictions (grouped).
    cv = GroupKFold(n_splits=N_SPLITS)
    perm_scores = {f: [] for f in all_numeric}
    for train_idx, test_idx in cv.split(X, y, groups):
        pipe = make_pipeline(all_numeric, CONTROL_CATEGORICAL)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        result = permutation_importance(
            pipe, X.iloc[test_idx], y[test_idx],
            n_repeats=20, random_state=SEED, scoring="r2",
        )
        for i, col in enumerate(X.columns):
            if col in perm_scores:
                perm_scores[col].append(result.importances_mean[i])

    imp_rows = []
    for feat in all_numeric:
        group = (
            "control" if feat in CONTROL_NUMERIC
            else "quantity" if feat in QUANTITY_FEATURES
            else "kind"
        )
        std_coef = float(coef_by_name.get(f"num__{feat}", np.nan))
        perm = float(np.mean(perm_scores[feat])) if perm_scores[feat] else np.nan
        imp_rows.append({
            "feature": feat,
            "group": group,
            "std_ridge_coef": std_coef,
            "abs_std_coef": abs(std_coef),
            "perm_importance_r2": perm,
        })
    importance = pd.DataFrame(imp_rows).sort_values(
        "abs_std_coef", ascending=False
    )
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # ----- 4. Report -----------------------------------------------------
    m = model_comparison.set_index("model")["cv_r2"].to_dict()
    quantity_gain = m["M1_controls+quantity"] - m["M0_controls"]
    kind_gain = m["M2_controls+kind"] - m["M0_controls"]

    top_corr = corr.head(8)[["feature", "group", "spearman_r",
                             "spearman_p"]].to_dict("records")
    top_imp = importance.head(8)[["feature", "group", "std_ridge_coef",
                                  "perm_importance_r2"]].to_dict("records")

    report = {
        "target": TARGET,
        "n_tasks": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "cv": f"GroupKFold(n_splits={N_SPLITS}) grouped by participant",
        "model_cv_r2": {k: float(v) for k, v in m.items()},
        "quantity_r2_gain_over_controls": float(quantity_gain),
        "kind_r2_gain_over_controls": float(kind_gain),
        "kind_beats_quantity": bool(kind_gain > quantity_gain),
        "model_fold_r2": {k: [float(v) for v in vs]
                          for k, vs in fold_r2_by_model.items()},
        "full_model_alpha": full_model_alpha,
        "regularisation": (
            "Ridge penalty tuned per training fold by RidgeCV over "
            f"alphas in [{ALPHAS.min():.3g}, {ALPHAS.max():.3g}]."
        ),
        "top_correlations": top_corr,
        "top_importance": top_imp,
        "interpretation": (
            "If kind_r2_gain_over_controls exceeds "
            "quantity_r2_gain_over_controls, the KIND of editing predicts "
            "personal-style recovery better than the AMOUNT of editing, "
            "which is the novel claim this analysis sets out to test. The "
            "full model (M3) stacks quantity and kind features that carry "
            "overlapping information; with the penalty tuned per fold its "
            "cross-validated R^2 reflects that redundancy rather than a lack "
            "of shrinkage."
        ),
        "seed": SEED,
    }
    with (OUT_DIR / "novel_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ----- Console summary ----------------------------------------------
    print("\n=== Model comparison (grouped CV R^2) ===")
    print(model_comparison.to_string(index=False))
    print(f"\n  Quantity adds R^2 = {quantity_gain:+.4f} over controls")
    print(f"  Kind     adds R^2 = {kind_gain:+.4f} over controls")
    print(f"  Kind beats quantity: {report['kind_beats_quantity']}")

    print("\n=== Strongest feature-target correlations (Spearman) ===")
    print(corr.head(8).to_string(index=False))

    print("\n=== Full-model importance (standardised Ridge coef) ===")
    print(importance.head(8).to_string(index=False))

    print("\nSaved to data/processed/novel/:")
    for name in ["feature_target_correlations.csv", "model_comparison.csv",
                 "feature_importance.csv", "novel_report.json"]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
