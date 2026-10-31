"""
src/novel/own_vs_generic_recovery.py

NOVEL CONTRIBUTION (WP6): is the style gained through post-editing
specific to the author's OWN voice, or is it generic "humanisation"?

The paper shows post-editing moves text toward the author's control style
(H1a) and, separately, more toward the author's own style than other
people's (H1a'). We go further and ask the question a skeptic raises about
our Stage-3 result: does editing *extent* predict movement toward the
author's OWN style specifically, or just generic movement toward any human
style? If heavy editing only moved text "away from the draft" it would
raise similarity to everyone equally.

For every treatment task we compute, in LUAR space:
    own_gain   = sim(edited, own control)   - sim(draft, own control)
    other_gain = sim(edited, others' ctrl)  - sim(draft, others' ctrl)
    specificity = own_gain - other_gain

Then:
  1. Test specificity > 0 (editing moves toward the author's own style
     more than toward other authors' styles) -- permutation test + g.
  2. Predict own_gain and other_gain from the edit features
     (GroupKFold by participant). If editing features predict own_gain
     better than other_gain, the restored style is personal, not generic.
  3. Predict own_gain while controlling for other_gain, to isolate the
     author-specific component.

Run from the repository root:
    python src/novel/own_vs_generic_recovery.py

Inputs:
    data/processed/luar_embeddings.npz
    data/processed/luar_embedding_index.csv
    data/processed/novel/edit_features.csv

Outputs (data/processed/novel/):
    own_vs_generic_task_scores.csv
    own_vs_generic_summary.csv
    own_vs_generic_models.csv
    own_vs_generic_report.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "novel"
N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 1_000
SEED = 42
N_SPLITS = 5
ALPHAS = np.logspace(-2, 3, 30)

EDIT_FEATURES = [
    "n_edits", "chars_inserted", "chars_deleted", "editing_time_s",
    "frac_words_changed", "words_replaced", "words_inserted", "words_deleted",
    "sent_count_change", "avg_sentlen_change", "ttr_change",
    "punct_count_change", "contraction_change", "first_person_change",
    "len_change_ratio", "draft_final_similarity",
]


def unit_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero-norm embedding.")
    return matrix / norms


def hedges_g(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    df = nx + ny - 2
    pooled = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / df
    if pooled <= 0:
        return np.nan
    d = (np.mean(y) - np.mean(x)) / np.sqrt(pooled)
    return float((1 - 3 / (4 * df - 1)) * d)


def paired_permutation(diff, rng):
    """Two-sided permutation test that mean(diff) != 0 by sign-flipping."""
    diff = np.asarray(diff, float)
    observed = np.mean(diff)
    count = 0
    for _ in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        if abs(np.mean(signs * diff)) >= abs(observed):
            count += 1
    return float(observed), float((count + 1) / (N_PERMUTATIONS + 1))


def load_task_scores():
    with np.load(DATA_DIR / "luar_embeddings.npz", allow_pickle=False) as a:
        vectors = a["embeddings"].astype(np.float64)
        keys = a["embedding_keys"].astype(str)
    index = pd.read_csv(
        DATA_DIR / "luar_embedding_index.csv",
        dtype={c: str for c in ["embedding_key", "participant_id",
                                "task_key", "scenario", "text_type"]})
    vec = {k: vectors[i] for i, k in enumerate(index["embedding_key"])}

    control = index[index["text_type"] == "control_reference"]
    control_pids = control["participant_id"].to_numpy()
    control_mat = unit_rows(np.stack([vec[k] for k in control["embedding_key"]]))

    def sim_to_controls(mat, pids):
        # Control texts are embedded individually (two per participant), so
        # own = mean similarity to the author's own control texts and
        # other = mean similarity to every other participant's control texts.
        u = unit_rows(mat)
        sim = u @ control_mat.T                       # (n, n_control_texts)
        same = pids[:, None] == control_pids[None, :]
        own = np.where(same, sim, np.nan)
        other = np.where(same, np.nan, sim)
        return np.nanmean(own, axis=1), np.nanmean(other, axis=1)

    draft = index[index["text_type"] == "ai_draft"].sort_values(
        ["participant_id", "task_key"]).reset_index(drop=True)
    edited = index[index["text_type"] == "edited_text"].sort_values(
        ["participant_id", "task_key"]).reset_index(drop=True)

    d_mat = np.stack([vec[k] for k in draft["embedding_key"]])
    e_mat = np.stack([vec[k] for k in edited["embedding_key"]])
    d_pids = draft["participant_id"].to_numpy()
    e_pids = edited["participant_id"].to_numpy()

    d_own, d_other = sim_to_controls(d_mat, d_pids)
    e_own, e_other = sim_to_controls(e_mat, e_pids)

    tbl = draft[["participant_id", "task_key", "scenario"]].copy()
    tbl["draft_to_own"] = d_own
    tbl["edited_to_own"] = e_own
    tbl["draft_to_other"] = d_other
    tbl["edited_to_other"] = e_other
    tbl["own_gain"] = e_own - d_own
    tbl["other_gain"] = e_other - d_other
    tbl["specificity"] = tbl["own_gain"] - tbl["other_gain"]
    return tbl


def grouped_r2(df, features, target):
    data = df.dropna(subset=features + [target])
    X = data[features]
    y = data[target].to_numpy()
    groups = data["participant_id"].to_numpy()
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", RidgeCV(alphas=ALPHAS))])
    preds = cross_val_predict(pipe, X, y, groups=groups,
                              cv=GroupKFold(n_splits=N_SPLITS))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    tbl = load_task_scores()
    feats_path = OUT_DIR / "edit_features.csv"
    if feats_path.exists():
        feats = pd.read_csv(feats_path, dtype={"participant_id": str,
                                               "task_key": str})
        tbl = tbl.merge(feats, on=["participant_id", "task_key"],
                        how="left", suffixes=("", "_f"))
        if "scenario_f" in tbl.columns:
            tbl = tbl.drop(columns=["scenario_f"])
        if "editing_time_s" in tbl.columns:
            tbl["editing_time_s"] = tbl["editing_time_s"].fillna(
                tbl["editing_time_s"].median())

    tbl.to_csv(OUT_DIR / "own_vs_generic_task_scores.csv", index=False)

    # ----- 1. Specificity test -------------------------------------------
    obs, p = paired_permutation(tbl["specificity"].to_numpy(), rng)
    g = hedges_g(tbl["other_gain"].to_numpy(), tbl["own_gain"].to_numpy())
    boots = []
    spec = tbl["specificity"].to_numpy()
    for _ in range(N_BOOTSTRAPS):
        boots.append(np.mean(rng.choice(spec, size=len(spec), replace=True)))
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    summary = pd.DataFrame([{
        "n_tasks": int(len(tbl)),
        "mean_own_gain": float(tbl["own_gain"].mean()),
        "mean_other_gain": float(tbl["other_gain"].mean()),
        "mean_specificity_own_minus_other": obs,
        "permutation_p": p,
        "hedges_g_own_vs_other": g,
        "specificity_ci_95_low": float(ci_low),
        "specificity_ci_95_high": float(ci_high),
    }])
    summary.to_csv(OUT_DIR / "own_vs_generic_summary.csv", index=False)

    # ----- 2 & 3. Do edit features predict own vs generic recovery? ------
    model_rows = []
    have_feats = feats_path.exists() and all(
        f in tbl.columns for f in EDIT_FEATURES)
    if have_feats:
        r2_own = grouped_r2(tbl, EDIT_FEATURES, "own_gain")
        r2_other = grouped_r2(tbl, EDIT_FEATURES, "other_gain")
        r2_own_ctrl = grouped_r2(
            tbl, EDIT_FEATURES + ["other_gain"], "own_gain")
        model_rows = [
            {"target": "own_gain", "features": "edit features",
             "cv_r2": r2_own},
            {"target": "other_gain", "features": "edit features",
             "cv_r2": r2_other},
            {"target": "own_gain", "features": "edit features + other_gain",
             "cv_r2": r2_own_ctrl},
        ]
        pd.DataFrame(model_rows).to_csv(
            OUT_DIR / "own_vs_generic_models.csv", index=False)

    report = {
        "n_tasks": int(len(tbl)),
        "specificity": {
            "mean_own_gain": float(tbl["own_gain"].mean()),
            "mean_other_gain": float(tbl["other_gain"].mean()),
            "mean_specificity": obs,
            "permutation_p": p,
            "hedges_g_own_vs_other": g,
            "ci_95": [float(ci_low), float(ci_high)],
        },
        "models": model_rows,
        "interpretation": (
            "Positive specificity means post-editing moves text toward the "
            "author's OWN control style more than toward other authors' "
            "styles (not generic humanisation). If edit features predict "
            "own_gain better than other_gain, the restored style is "
            "author-specific. NOTE: the 'edit features + other_gain' model "
            "includes other_gain as a predictor, so its R^2 answers a "
            "different question (how much author-specific signal is left once "
            "the generic component is accounted for) and is NOT directly "
            "comparable to the edit-features-only R^2 values above."
        ),
        "seed": SEED,
    }
    with (OUT_DIR / "own_vs_generic_report.json").open("w",
                                                       encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ----- Console -------------------------------------------------------
    print("=== Own vs generic style recovery ===")
    print(f"  tasks                 : {len(tbl)}")
    print(f"  mean own_gain         : {tbl['own_gain'].mean():+.4f}")
    print(f"  mean other_gain       : {tbl['other_gain'].mean():+.4f}")
    print(f"  specificity (own-other): {obs:+.4f}  "
          f"(perm p = {p:.5f}, g = {g:.3f}, "
          f"95% CI [{ci_low:.4f}, {ci_high:.4f}])")
    verdict = ("author-SPECIFIC (not generic humanisation)" if obs > 0 and
               p < 0.05 else "not clearly author-specific")
    print(f"  => {verdict}")
    if model_rows:
        print("\n=== Do edit features predict recovery? (grouped CV R^2) ===")
        for r in model_rows:
            print(f"  {r['target']:<11} ~ {r['features']:<28}: "
                  f"R^2 = {r['cv_r2']:.4f}")
        print("  (edit features predicting own_gain > other_gain would show "
              "editing restores personal, not generic, style)")

    print("\nSaved to data/processed/novel/:")
    for n in ["own_vs_generic_task_scores.csv", "own_vs_generic_summary.csv",
              "own_vs_generic_models.csv", "own_vs_generic_report.json"]:
        print(f"  {n}")


if __name__ == "__main__":
    main()
