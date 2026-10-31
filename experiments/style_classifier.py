"""
src/novel/style_classifier.py

NOVEL CONTRIBUTION (ML upgrade): instead of only using LUAR cosine
*similarity scores*, we train machine-learning models directly on the
512-dimensional LUAR *embeddings* to study post-edited style.

Three analyses, all cross-validated with GroupKFold so that no
participant appears in both training and test (the models are judged on
unseen authors):

  A. THREE-WAY STYLE CLASSIFIER
     Classify each text as human control / LLM draft / post-edited.
     This turns the paper's H2 claim -- that post-edited text is a
     "distinct, detectable third variety" -- into a trained model.
     The headline is the confusion matrix: when post-edited text is
     misclassified, is it mistaken for human or for LLM?

  B. LLM-ORIGIN DETECTOR
     Train a binary detector on control (human) vs draft (LLM), then run
     it on the post-edited texts. The fraction of edited texts still
     flagged as LLM, and the drop in predicted P(LLM) from draft to
     edited, is a model-based measure of how much LLM style survives
     post-editing -- our analogue of the paper's PANGRAM analysis.

  C. FUSION WITH EDIT TRACES
     Correlate the detector's P(LLM) on each edited text with the
     interpretable edit features from extract_edit_features.py, to see
     which editing behaviours actually make a text look less LLM-like to
     the model.

Models compared: multinomial/logistic regression and random forest.

Run from the repository root:
    python src/novel/style_classifier.py

Inputs:
    data/processed/luar_embeddings.npz
    data/processed/luar_embedding_index.csv
    data/processed/novel/edit_features.csv        (for analysis C)

Outputs (data/processed/novel/):
    classifier_metrics.csv
    classifier_3way_confusion.csv
    detector_on_edited.csv
    edit_vs_detector_correlations.csv
    novel_classifier_report.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "novel"

SEED = 42
N_SPLITS = 5

CLASS_LABELS = {
    "control_reference": "human_control",
    "ai_draft": "llm_draft",
    "edited_text": "post_edited",
}


def load_embeddings():
    with np.load(DATA_DIR / "luar_embeddings.npz", allow_pickle=False) as a:
        vectors = a["embeddings"].astype(np.float64)
        keys = a["embedding_keys"].astype(str)
    index = pd.read_csv(
        DATA_DIR / "luar_embedding_index.csv",
        dtype={"embedding_key": str, "participant_id": str,
               "task_key": str, "scenario": str, "text_type": str},
    )
    if not np.array_equal(keys, index["embedding_key"].to_numpy()):
        raise ValueError("Embedding keys and index rows are misaligned.")
    index = index.copy()
    index["label"] = index["text_type"].map(CLASS_LABELS)
    return vectors, index


def make_models():
    """Two classifiers to compare. Logistic regression is scaled; the
    forest does not need scaling but is wrapped identically for symmetry."""
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("clf", RandomForestClassifier(
                n_estimators=400, class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1)),
        ]),
    }


def participant_folds(participants):
    """Assign each participant to one of N_SPLITS folds (shuffled), so the
    same folds can be reused across the three text subsets."""
    parts = sorted(participants)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = {}
    for fold, (_, test_idx) in enumerate(kf.split(parts)):
        for i in test_idx:
            fold_of[parts[i]] = fold
    return fold_of


# ---------------------------------------------------------------------------
# A. Three-way style classifier
# ---------------------------------------------------------------------------
def three_way_classifier(vectors, index):
    X = vectors
    y = index["label"].to_numpy()
    groups = index["participant_id"].to_numpy()
    cv = GroupKFold(n_splits=N_SPLITS)

    label_order = ["human_control", "llm_draft", "post_edited"]
    results = {}
    confusions = {}

    for name, model in make_models().items():
        preds = cross_val_predict(model, X, y, groups=groups, cv=cv)
        acc = accuracy_score(y, preds)
        bal_acc = balanced_accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average="macro")
        report = classification_report(
            y, preds, labels=label_order, output_dict=True, zero_division=0)

        cm = confusion_matrix(y, preds, labels=label_order)
        cm_df = pd.DataFrame(
            cm, index=[f"true_{l}" for l in label_order],
            columns=[f"pred_{l}" for l in label_order])
        confusions[name] = cm_df

        # Where do post-edited texts land?
        pe_row = cm_df.loc["true_post_edited"]
        pe_total = pe_row.sum()
        results[name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "post_edited_pred_human_frac": float(
                pe_row["pred_human_control"] / pe_total),
            "post_edited_pred_llm_frac": float(
                pe_row["pred_llm_draft"] / pe_total),
            "post_edited_pred_edited_frac": float(
                pe_row["pred_post_edited"] / pe_total),
            "precision_post_edited": float(
                report["post_edited"]["precision"]),
            "recall_post_edited": float(report["post_edited"]["recall"]),
        }

    return results, confusions, label_order


# ---------------------------------------------------------------------------
# B. LLM-origin detector (control vs draft) applied to edited texts
# ---------------------------------------------------------------------------
def llm_origin_detector(vectors, index, model_name="logreg"):
    is_control = index["text_type"] == "control_reference"
    is_draft = index["text_type"] == "ai_draft"
    is_edited = index["text_type"] == "edited_text"

    fold_of = participant_folds(index["participant_id"].unique())
    fold = index["participant_id"].map(fold_of).to_numpy()

    # Held-out evaluation on control-vs-draft, and OOF P(LLM) for edited.
    heldout_true, heldout_prob = [], []
    edited_rows, edited_prob = [], []

    for f in range(N_SPLITS):
        train_mask = (fold != f) & (is_control | is_draft).to_numpy()
        test_cd_mask = (fold == f) & (is_control | is_draft).to_numpy()
        test_ed_mask = (fold == f) & is_edited.to_numpy()

        y_train = is_draft[train_mask].astype(int).to_numpy()  # 1 = LLM
        model = make_models()[model_name]
        model.fit(vectors[train_mask], y_train)

        # Evaluate on held-out human/LLM texts.
        if test_cd_mask.any():
            heldout_true.extend(is_draft[test_cd_mask].astype(int).tolist())
            heldout_prob.extend(
                model.predict_proba(vectors[test_cd_mask])[:, 1].tolist())

        # Score the held-out participants' edited texts.
        if test_ed_mask.any():
            idx = np.where(test_ed_mask)[0]
            edited_rows.extend(idx.tolist())
            edited_prob.extend(
                model.predict_proba(vectors[test_ed_mask])[:, 1].tolist())

    heldout_true = np.array(heldout_true)
    heldout_prob = np.array(heldout_prob)
    detector_auc = float(roc_auc_score(heldout_true, heldout_prob))
    detector_acc = float(accuracy_score(heldout_true, heldout_prob > 0.5))

    # Mean predicted P(LLM): drafts (held-out) vs edited (OOF).
    draft_prob = heldout_prob[heldout_true == 1]
    edited_prob = np.array(edited_prob)
    edited_index = index.iloc[edited_rows].copy()
    edited_index["p_llm"] = edited_prob
    edited_index["predicted_llm"] = (edited_prob > 0.5).astype(int)

    summary = {
        "model": model_name,
        "detector_auc_control_vs_draft": detector_auc,
        "detector_accuracy_control_vs_draft": detector_acc,
        "mean_p_llm_drafts": float(np.mean(draft_prob)),
        "mean_p_llm_edited": float(np.mean(edited_prob)),
        "frac_edited_flagged_llm": float(np.mean(edited_prob > 0.5)),
        "n_edited": int(len(edited_prob)),
    }
    return summary, edited_index[[
        "participant_id", "task_key", "scenario", "p_llm", "predicted_llm"]]


# ---------------------------------------------------------------------------
# C. Fusion: which edits make a text look less LLM-like to the detector?
# ---------------------------------------------------------------------------
def fuse_with_edit_features(edited_scores):
    feats_path = OUT_DIR / "edit_features.csv"
    if not feats_path.exists():
        return None
    feats = pd.read_csv(feats_path, dtype={"participant_id": str,
                                           "task_key": str})
    merged = edited_scores.merge(
        feats, on=["participant_id", "task_key"], how="inner",
        suffixes=("", "_feat"))

    candidate = [
        "frac_words_changed", "words_replaced", "words_inserted",
        "words_deleted", "n_edits", "chars_inserted", "contraction_change",
        "punct_count_change", "first_person_change", "sent_count_change",
        "ttr_change", "len_change_ratio", "draft_final_similarity",
        "editing_time_s",
    ]
    candidate = [c for c in candidate if c in merged.columns]

    rows = []
    for feat in candidate:
        sub = merged[[feat, "p_llm"]].dropna()
        if len(sub) < 10:
            continue
        r, p = stats.spearmanr(sub[feat], sub["p_llm"])
        rows.append({
            "feature": feat,
            "spearman_r_with_p_llm": float(r),
            "spearman_p": float(p),
            "abs_r": abs(float(r)),
            # negative r => more of this edit -> lower detected LLM-ness
            "direction": "reduces LLM-ness" if r < 0 else "increases LLM-ness",
        })
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors, index = load_embeddings()

    print(f"Texts: {len(index)}  "
          f"(human_control={int((index['label']=='human_control').sum())}, "
          f"llm_draft={int((index['label']=='llm_draft').sum())}, "
          f"post_edited={int((index['label']=='post_edited').sum())})")
    print(f"Participants: {index['participant_id'].nunique()}  "
          f"CV: GroupKFold({N_SPLITS}) by participant\n")

    # ----- A. Three-way classifier --------------------------------------
    three_way, confusions, label_order = three_way_classifier(vectors, index)

    metrics_rows = []
    for name, r in three_way.items():
        metrics_rows.append({"task": "3way_style_classifier", "model": name,
                             **r})

    # Save confusion matrices (both models).
    conf_frames = []
    for name, cm_df in confusions.items():
        tmp = cm_df.reset_index().rename(columns={"index": "row"})
        tmp.insert(0, "model", name)
        conf_frames.append(tmp)
    pd.concat(conf_frames, ignore_index=True).to_csv(
        OUT_DIR / "classifier_3way_confusion.csv", index=False)

    # ----- B. LLM-origin detector ---------------------------------------
    detector_logreg, edited_scores = llm_origin_detector(
        vectors, index, "logreg")
    detector_rf, _ = llm_origin_detector(vectors, index, "random_forest")
    for d in (detector_logreg, detector_rf):
        metrics_rows.append({"task": "llm_origin_detector", **d})

    edited_scores.to_csv(OUT_DIR / "detector_on_edited.csv", index=False)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUT_DIR / "classifier_metrics.csv", index=False)

    # ----- C. Fusion with edit features ---------------------------------
    fusion = fuse_with_edit_features(edited_scores)
    if fusion is not None:
        fusion.to_csv(OUT_DIR / "edit_vs_detector_correlations.csv",
                      index=False)

    # ----- Report + console ---------------------------------------------
    report = {
        "n_texts": int(len(index)),
        "n_participants": int(index["participant_id"].nunique()),
        "cv": f"GroupKFold({N_SPLITS}) by participant",
        "three_way_classifier": three_way,
        "llm_origin_detector": {"logreg": detector_logreg,
                                "random_forest": detector_rf},
        "chance_accuracy_3way": 1 / 3,
        "interpretation": (
            "Analysis A tests whether post-edited text is a learnable "
            "distinct class and where it is confused; B measures residual "
            "LLM style via a trained detector; C links detected LLM-ness to "
            "specific editing behaviours."
        ),
        "caveats": [
            "The three-way classifier uses class-weight balancing for the "
            "81/324/324 imbalance; balanced accuracy is the fair metric "
            "(chance = 0.333). All scores are out-of-fold under GroupKFold "
            "by participant, i.e. on unseen authors.",
            "The LLM-origin detector reaches AUC ~1.0 partly as an artifact: "
            "control texts are a COMBINED embedding of two writings while "
            "drafts and edited texts are single-text embeddings, so the "
            "detector also separates 'combined vs single'. Combined with the "
            "fact that an edited text is derived from its own draft, this "
            "inflates how LLM-like edited text looks. Read the detector as "
            "corroborating H1c (edited stays close to LLM) directionally, "
            "not as an exact residual-LLM rate; a clean test needs "
            "individually embedded control texts (re-embedding).",
        ],
        "seed": SEED,
    }
    if fusion is not None:
        report["top_edit_correlations_with_detected_llm"] = (
            fusion.head(6).to_dict("records"))
    with (OUT_DIR / "novel_classifier_report.json").open(
            "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("=== A. Three-way style classifier (OOF, GroupKFold) ===")
    for name, r in three_way.items():
        print(f"  [{name}] acc={r['accuracy']:.3f}  "
              f"balanced_acc={r['balanced_accuracy']:.3f} "
              f"(chance 0.333)  macro_F1={r['macro_f1']:.3f}")
        print(f"      post-edited texts predicted -> "
              f"human {r['post_edited_pred_human_frac']:.2f} | "
              f"LLM {r['post_edited_pred_llm_frac']:.2f} | "
              f"edited {r['post_edited_pred_edited_frac']:.2f}")

    print("\n=== B. LLM-origin detector (trained on human vs LLM) ===")
    for name, d in (("logreg", detector_logreg),
                    ("random_forest", detector_rf)):
        print(f"  [{name}] AUC={d['detector_auc_control_vs_draft']:.3f}  "
              f"acc={d['detector_accuracy_control_vs_draft']:.3f}")
        print(f"      mean P(LLM): drafts={d['mean_p_llm_drafts']:.3f} -> "
              f"edited={d['mean_p_llm_edited']:.3f};  "
              f"{d['frac_edited_flagged_llm']*100:.1f}% of edited still "
              f"flagged LLM")

    if fusion is not None:
        print("\n=== C. Edits most associated with lower detected LLM-ness ===")
        print(fusion.head(6).to_string(index=False))

    print("\nCaveat: the detector's AUC ~1.0 is partly an embedding "
          "artifact (combined control vs single-text draft/edited); read it "
          "directionally. See novel_classifier_report.json for details.")

    print("\nSaved to data/processed/novel/:")
    for n in ["classifier_metrics.csv", "classifier_3way_confusion.csv",
              "detector_on_edited.csv", "edit_vs_detector_correlations.csv",
              "novel_classifier_report.json"]:
        print(f"  {n}")


if __name__ == "__main__":
    main()
