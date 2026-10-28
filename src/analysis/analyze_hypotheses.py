"""
src/analysis/analyze_hypotheses.py

Consolidated analysis for every pre-registered hypothesis in
"Can You Make It Sound Like You? Post-Editing LLM-Generated Text for
Personal Style" (Baumler et al., 2026).

This single file reproduces H1a (already run separately in
analyze_h1a.py) and adds H1a', H1b, H1c, H2a, H2b, H2c, and H3, so the
whole hypothesis set lives in one place and uses one set of statistical
helpers.

Style measure
-------------
All stylistic-similarity hypotheses use the cosine similarity between
LUAR-MUD embeddings, exactly as in the paper (Section 5 / Appendix B.2).
We reuse the embeddings produced by src/embed_luar.py:

    control_reference : one combined embedding of each participant's two
                        unassisted (control) writings. The paper also
                        represents each participant's control writing as
                        a single point, so this matches its homogeneity
                        analysis (one control point per participant).
    ai_draft          : the LLM-generated draft shown for a treatment
                        task (our stand-in for "LLM-generated text").
    edited_text       : the participant's post-edited final text.

"Similarity to LLM-generated text" (H1b, H1c) and "similarity to other
participants' writing" (H1a', H2c) are computed as the mean cosine
similarity to the relevant reference texts of OTHER participants
(leave-own-participant-out), which mirrors the paper's between-author
framing ("stylistic similarity between texts from two random
participants").

Statistics
----------
Following the paper (Appendix B.3) and the existing analyze_h1a.py, each
hypothesis compares two lists of task- or pair-level scores with:
  * a two-sided permutation test on the difference in means
    (10,000 permutations, plus-one corrected), and
  * Hedges' g with a percentile bootstrap 95% CI (1,000 resamples).
H3 instead reports the correlation between perceived and measured
self-similarity, which is how the paper frames it (Figure 8).

IMPORTANT CAVEATS (printed at the end as well):
  * Tests treat task-level (or pair-level) scores as independent. Several
    tasks come from each participant, and each homogeneity pair reuses
    participants, so these are not clustered/exact reproductions of the
    authors' inferential procedure. Effect-size *directions* are the
    primary thing to read.
  * We do not have PANGRAM AI-detector scores, so H1b is reported with
    the LUAR style measure only (the paper reports both and they agree).

Run from the repository root:
    python src/analysis/analyze_hypotheses.py

Inputs:
    data/processed/luar_embeddings.npz
    data/processed/luar_embedding_index.csv
    data/processed/task_records.jsonl        (for H3 self-ratings)

Outputs:
    data/processed/hypotheses/all_hypotheses_summary.csv
    data/processed/hypotheses/h1_task_scores.csv
    data/processed/hypotheses/h2_homogeneity_pairs.csv
    data/processed/hypotheses/h3_perception_vs_measure.csv
    data/processed/hypotheses/hypotheses_report.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "hypotheses"

N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 1_000
SEED = 42

# Effect sizes / p-values reported in the paper, for side-by-side
# comparison. None where the paper does not give a single g (e.g. H2b/H2c
# are read from figures) or where the hypothesis is not an effect size.
PAPER_REFERENCE = {
    "H1a": {"g": 0.55, "p": 0.0002},
    "H1a_prime": {"g": -0.56, "p": 0.0002},
    "H1b": {"g": -0.41, "p": 0.0002},
    "H1c": {"g": -1.43, "p": 0.0002},
    "H2a": {"g": 1.42, "p": 0.0002},
    "H2b": {"g": None, "p": None},
    "H2c": {"g": None, "p": None},
}


# ---------------------------------------------------------------------------
# Statistical helpers (shared conventions with analyze_h1a.py)
# ---------------------------------------------------------------------------
def hedges_g_independent(x, y):
    """Hedges' g for two independent groups (positive => mean(y) > mean(x))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    df = nx + ny - 2
    pooled_variance = ((nx - 1) * vx + (ny - 1) * vy) / df
    if pooled_variance <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_variance)
    cohens_d = (np.mean(y) - np.mean(x)) / pooled_sd

    correction = 1 - (3 / (4 * df - 1))
    return float(correction * cohens_d)


def permutation_test_mean_difference(x, y, rng):
    """Two-sided permutation test for mean(y) - mean(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    observed = np.mean(y) - np.mean(x)
    pooled = np.concatenate([x, y])
    nx = len(x)

    count = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(pooled)
        statistic = np.mean(shuffled[nx:]) - np.mean(shuffled[:nx])
        if abs(statistic) >= abs(observed):
            count += 1

    p_value = (count + 1) / (N_PERMUTATIONS + 1)
    return float(observed), float(p_value)


def bootstrap_hedges_g_ci(x, y, rng):
    """Percentile 95% CI for independent-groups Hedges' g."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    estimates = []
    for _ in range(N_BOOTSTRAPS):
        sample_x = rng.choice(x, size=len(x), replace=True)
        sample_y = rng.choice(y, size=len(y), replace=True)
        estimate = hedges_g_independent(sample_x, sample_y)
        if np.isfinite(estimate):
            estimates.append(estimate)

    if not estimates:
        return np.nan, np.nan

    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def compare_two_lists(name, description, comparison, x, y, rng):
    """Run the standard permutation + Hedges' g comparison and package it.

    x and y are paired-or-pooled score lists. g and the mean difference are
    oriented as (y - x): a positive value means y scored higher than x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    observed_difference, p_value = permutation_test_mean_difference(x, y, rng)
    g = hedges_g_independent(x, y)
    ci_low, ci_high = bootstrap_hedges_g_ci(x, y, rng)

    return {
        "hypothesis": name,
        "description": description,
        "comparison": comparison,
        "n_x": int(len(x)),
        "n_y": int(len(y)),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "mean_difference_y_minus_x": observed_difference,
        "permutation_p_two_sided": p_value,
        "hedges_g": g,
        "g_ci_95_low": ci_low,
        "g_ci_95_high": ci_high,
    }


# ---------------------------------------------------------------------------
# Embedding loading + vectorised cosine utilities
# ---------------------------------------------------------------------------
def load_embeddings():
    embeddings_path = DATA_DIR / "luar_embeddings.npz"
    index_path = DATA_DIR / "luar_embedding_index.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing {embeddings_path}. Run src/embed_luar.py first."
        )
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}.")

    with np.load(embeddings_path, allow_pickle=False) as archive:
        vectors = archive["embeddings"].astype(np.float64)
        keys = archive["embedding_keys"].astype(str)

    index = pd.read_csv(
        index_path,
        dtype={
            "embedding_key": str,
            "participant_id": str,
            "task_key": str,
            "scenario": str,
            "text_type": str,
        },
    )

    if len(vectors) != len(keys) or len(keys) != len(index):
        raise ValueError("Embedding matrix, keys, and index are misaligned.")
    if not np.array_equal(keys, index["embedding_key"].to_numpy()):
        raise ValueError("Embedding keys and index rows are misaligned.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    return vectors, index


def unit_rows(matrix):
    """Return L2-normalised rows (so dot products give cosine similarity)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Encountered a zero-norm embedding.")
    return matrix / norms


def mean_sim_to_other_participants(target_matrix, target_pids,
                                   ref_matrix, ref_pids):
    """Mean cosine of each target text to all reference texts from OTHER
    participants (leave-own-participant-out)."""
    t = unit_rows(target_matrix)
    r = unit_rows(ref_matrix)

    sim = t @ r.T  # (n_target, n_ref) cosine similarities

    target_pids = np.asarray(target_pids)
    ref_pids = np.asarray(ref_pids)
    same = target_pids[:, None] == ref_pids[None, :]

    masked = np.where(same, np.nan, sim)
    return np.nanmean(masked, axis=1)


def pairwise_similarities_between_participants(rep_matrix):
    """All unique off-diagonal pairwise cosine similarities among a set of
    per-participant representation vectors (upper triangle)."""
    r = unit_rows(rep_matrix)
    sim = r @ r.T
    iu = np.triu_indices(len(r), k=1)
    return sim[iu]


# ---------------------------------------------------------------------------
# Assemble per-text views of the embeddings
# ---------------------------------------------------------------------------
def build_views(vectors, index):
    vector_by_key = {k: vectors[i] for i, k in
                     enumerate(index["embedding_key"])}

    def subset(text_type):
        rows = index[index["text_type"] == text_type].copy()
        rows = rows.sort_values(["participant_id", "task_key"])
        mat = np.stack([vector_by_key[k] for k in rows["embedding_key"]])
        return rows.reset_index(drop=True), mat

    control_rows, control_mat = subset("control_reference")
    draft_rows, draft_mat = subset("ai_draft")
    edited_rows, edited_mat = subset("edited_text")

    # One control reference per participant -> dict and aligned matrix.
    if control_rows["participant_id"].duplicated().any():
        raise ValueError("More than one control reference per participant.")
    control_pids = control_rows["participant_id"].to_numpy()
    control_vec_by_pid = {
        pid: control_mat[i] for i, pid in enumerate(control_pids)
    }

    # Draft and edited aligned per task; also participant-level centroids.
    def participant_centroids(rows, mat):
        pids, reps = [], []
        for pid, idx in rows.groupby("participant_id").groups.items():
            reps.append(mat[list(idx)].mean(axis=0))
            pids.append(pid)
        order = np.argsort(pids)
        pids = [pids[i] for i in order]
        reps = np.stack([reps[i] for i in order])
        return np.array(pids), reps

    draft_pids_c, draft_centroids = participant_centroids(draft_rows, draft_mat)
    edited_pids_c, edited_centroids = participant_centroids(edited_rows, edited_mat)

    return {
        "control_rows": control_rows,
        "control_mat": control_mat,
        "control_pids": control_pids,
        "control_vec_by_pid": control_vec_by_pid,
        "draft_rows": draft_rows,
        "draft_mat": draft_mat,
        "edited_rows": edited_rows,
        "edited_mat": edited_mat,
        "draft_centroids": draft_centroids,
        "draft_centroid_pids": draft_pids_c,
        "edited_centroids": edited_centroids,
        "edited_centroid_pids": edited_pids_c,
    }


def task_table(views):
    """Per treatment-task table with the similarities the H1 family needs.

    Columns: participant_id, task_key, scenario,
             draft_to_control, edited_to_control, change_after_editing,
             draft_to_llm, edited_to_llm,
             edited_to_own_control, edited_to_other_controls,
             edited_to_other_edited.
    """
    draft_rows = views["draft_rows"]
    edited_rows = views["edited_rows"]

    # Align draft and edited rows on (participant_id, task_key).
    key = ["participant_id", "task_key", "scenario"]
    merged = draft_rows[key].merge(edited_rows[key], on=key, how="inner")
    if len(merged) != len(draft_rows) or len(merged) != len(edited_rows):
        raise ValueError("Draft and edited tasks do not align one-to-one.")

    draft_mat = views["draft_mat"]
    edited_mat = views["edited_mat"]
    draft_pids = draft_rows["participant_id"].to_numpy()
    edited_pids = edited_rows["participant_id"].to_numpy()

    # Similarity to own combined control reference.
    control_vec_by_pid = views["control_vec_by_pid"]
    control_aligned_for_draft = unit_rows(
        np.stack([control_vec_by_pid[p] for p in draft_pids])
    )
    control_aligned_for_edited = unit_rows(
        np.stack([control_vec_by_pid[p] for p in edited_pids])
    )
    draft_u = unit_rows(draft_mat)
    edited_u = unit_rows(edited_mat)

    draft_to_control = np.sum(draft_u * control_aligned_for_draft, axis=1)
    edited_to_control = np.sum(edited_u * control_aligned_for_edited, axis=1)

    # Similarity to the LLM style class (other participants' drafts).
    draft_to_llm = mean_sim_to_other_participants(
        draft_mat, draft_pids, draft_mat, draft_pids
    )
    edited_to_llm = mean_sim_to_other_participants(
        edited_mat, edited_pids, draft_mat, draft_pids
    )

    # Similarity of edited text to OTHER participants' control writing.
    control_rows = views["control_rows"]
    edited_to_other_controls = mean_sim_to_other_participants(
        edited_mat, edited_pids,
        views["control_mat"], control_rows["participant_id"].to_numpy(),
    )

    # Similarity of edited text to OTHER participants' edited writing.
    edited_to_other_edited = mean_sim_to_other_participants(
        edited_mat, edited_pids, edited_mat, edited_pids
    )

    table = draft_rows[key].copy()
    table["draft_to_control"] = draft_to_control
    table["edited_to_control"] = edited_to_control
    table["change_after_editing"] = edited_to_control - draft_to_control
    table["draft_to_llm"] = draft_to_llm
    table["edited_to_llm"] = edited_to_llm
    table["edited_to_own_control"] = edited_to_control
    table["edited_to_other_controls"] = edited_to_other_controls
    table["edited_to_other_edited"] = edited_to_other_edited
    return table.sort_values(["participant_id", "task_key"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# H3 inputs (perceived vs measured self-similarity)
# ---------------------------------------------------------------------------
def load_self_ratings():
    path = DATA_DIR / "task_records.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model_generation_shown") != 1:
                continue
            likert = r.get("likert") or {}

            def num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return np.nan

            rows.append({
                "participant_id": str(r["participant_id"]),
                "task_key": str(r["task_key"]),
                "perceived_edited_capture": num(likert.get("edited-capture")),
                "perceived_original_capture": num(likert.get("original-capture")),
                "voice_important": num(likert.get("voice-important")),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vectors, index = load_embeddings()
    views = build_views(vectors, index)
    tasks = task_table(views)

    tasks.to_csv(OUT_DIR / "h1_task_scores.csv", index=False)

    results = []

    # ----- H1a: edited vs draft, similarity to own control ----------------
    results.append(compare_two_lists(
        "H1a",
        "AI-assisted writing sounds more like own unassisted writing "
        "after post-editing than before.",
        "draft->own control  vs  edited->own control",
        tasks["draft_to_control"], tasks["edited_to_control"], rng,
    ))

    # ----- H1a': own control vs other participants' control --------------
    # Negative g => edited is more similar to own control than to others'.
    results.append(compare_two_lists(
        "H1a_prime",
        "Post-edited text sounds more like the author's OWN unassisted "
        "writing than like OTHER participants' unassisted writing.",
        "edited->own control  vs  edited->other participants' control",
        tasks["edited_to_own_control"], tasks["edited_to_other_controls"], rng,
    ))

    # ----- H1b: draft vs edited, similarity to LLM class -----------------
    # Negative g => edited is less similar to LLM text than the draft was.
    results.append(compare_two_lists(
        "H1b",
        "AI-assisted writing sounds LESS like LLM-generated writing after "
        "post-editing than before (LUAR style measure; paper also uses PANGRAM).",
        "draft->LLM class  vs  edited->LLM class",
        tasks["draft_to_llm"], tasks["edited_to_llm"], rng,
    ))

    # ----- H1c: edited->LLM vs edited->own control -----------------------
    # Negative g => post-edited text stays closer to LLM text than to the
    # author's own control writing (paper g = -1.43).
    results.append(compare_two_lists(
        "H1c",
        "Post-edited writing still sounds MORE like LLM-generated writing "
        "than like the author's own unassisted writing.",
        "edited->LLM class  vs  edited->own control",
        tasks["edited_to_llm"], tasks["edited_to_own_control"], rng,
    ))

    # ----- H2a: homogeneity edited vs control ----------------------------
    control_pairs = pairwise_similarities_between_participants(
        np.stack([views["control_vec_by_pid"][p]
                  for p in sorted(views["control_vec_by_pid"])])
    )
    edited_pairs = pairwise_similarities_between_participants(
        views["edited_centroids"]
    )
    draft_pairs = pairwise_similarities_between_participants(
        views["draft_centroids"]
    )

    results.append(compare_two_lists(
        "H2a",
        "Post-edited writing is MORE stylistically homogeneous (higher "
        "between-author similarity) than unassisted human writing.",
        "between-author similarity: control pairs  vs  edited pairs",
        control_pairs, edited_pairs, rng,
    ))

    # ----- H2b: homogeneity edited vs LLM --------------------------------
    results.append(compare_two_lists(
        "H2b",
        "Post-edited writing is LESS stylistically homogeneous than "
        "LLM-generated writing.",
        "between-author similarity: draft pairs  vs  edited pairs",
        draft_pairs, edited_pairs, rng,
    ))

    # ----- H2c: own control vs other participants' edited ----------------
    results.append(compare_two_lists(
        "H2c",
        "After post-editing, participants sound LESS like their own "
        "unassisted writing than like OTHER participants' post-edited writing.",
        "edited->own control  vs  edited->other participants' edited",
        tasks["edited_to_own_control"], tasks["edited_to_other_edited"], rng,
    ))

    # Save H2 pairwise distributions for inspection / plotting.
    h2_pairs = pd.concat([
        pd.DataFrame({"group": "control", "similarity": control_pairs}),
        pd.DataFrame({"group": "edited", "similarity": edited_pairs}),
        pd.DataFrame({"group": "llm_draft", "similarity": draft_pairs}),
    ], ignore_index=True)
    h2_pairs.to_csv(OUT_DIR / "h2_homogeneity_pairs.csv", index=False)

    # ----- H3: perceived vs measured self-similarity ---------------------
    ratings = load_self_ratings()
    h3 = tasks.merge(ratings, on=["participant_id", "task_key"], how="inner")

    perceived = h3["perceived_edited_capture"].to_numpy()
    measured = h3["edited_to_control"].to_numpy()
    valid = np.isfinite(perceived) & np.isfinite(measured)
    perceived, measured = perceived[valid], measured[valid]
    h3_valid = h3[valid].copy()

    spearman_r, spearman_p = stats.spearmanr(perceived, measured)
    pearson_r, pearson_p = stats.pearsonr(perceived, measured)

    # Binary agreement: did the participant feel editing improved capture of
    # their style, and did LUAR agree that editing moved it toward control?
    perceived_improved = (
        h3_valid["perceived_edited_capture"]
        >= h3_valid["perceived_original_capture"]
    ).to_numpy()
    measured_improved = (h3_valid["change_after_editing"] > 0).to_numpy()
    agreement_rate = float(np.mean(perceived_improved == measured_improved))

    h3_valid.to_csv(OUT_DIR / "h3_perception_vs_measure.csv", index=False)

    h3_result = {
        "hypothesis": "H3",
        "description": (
            "Participants agree with the automatic style metric about "
            "whether post-editing made the text sound like them."
        ),
        "comparison": "perceived self-similarity (edited-capture, 1-5)  vs  "
                      "LUAR edited->own control",
        "n": int(len(perceived)),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "binary_agreement_rate": agreement_rate,
    }

    # ----- Assemble summary table ----------------------------------------
    summary = pd.DataFrame(results)
    summary["paper_g"] = summary["hypothesis"].map(
        lambda h: PAPER_REFERENCE.get(h, {}).get("g")
    )
    summary["paper_p"] = summary["hypothesis"].map(
        lambda h: PAPER_REFERENCE.get(h, {}).get("p")
    )
    summary["significant_p_lt_.05"] = summary["permutation_p_two_sided"] < 0.05
    summary["direction_matches_paper"] = summary.apply(
        lambda row: (
            None if PAPER_REFERENCE.get(row["hypothesis"], {}).get("g") is None
            else bool(np.sign(row["hedges_g"])
                      == np.sign(PAPER_REFERENCE[row["hypothesis"]]["g"]))
        ),
        axis=1,
    )

    summary.to_csv(OUT_DIR / "all_hypotheses_summary.csv", index=False)

    report = {
        "n_participants": int(tasks["participant_id"].nunique()),
        "n_treatment_tasks": int(len(tasks)),
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap_samples": N_BOOTSTRAPS,
        "random_seed": SEED,
        "hypotheses": results,
        "H3": h3_result,
        "caveats": [
            "Scores are treated as independent; tasks are nested within "
            "participants and homogeneity pairs reuse participants.",
            "LLM style class = other participants' drafts "
            "(leave-own-participant-out).",
            "H1b uses the LUAR style measure only; PANGRAM scores are not "
            "available in this dataset.",
            "Control writing is represented by one combined embedding per "
            "participant, matching the paper's one-point-per-participant "
            "homogeneity analysis.",
        ],
    }
    with (OUT_DIR / "hypotheses_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ----- Console report -------------------------------------------------
    # Print a full block per hypothesis showing every reported value.
    def fmt(v, nd=4):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    print("\n" + "=" * 72)
    print("HYPOTHESIS RESULTS (LUAR cosine similarity; "
          "10,000-perm test, bootstrap 95% CI)")
    print("=" * 72)

    for r in results:
        hyp = r["hypothesis"]
        ref = PAPER_REFERENCE.get(hyp, {})
        p = r["permutation_p_two_sided"]
        sig = "yes" if p < 0.05 else "no"
        direction = (
            "n/a" if ref.get("g") is None
            else ("matches paper"
                  if np.sign(r["hedges_g"]) == np.sign(ref["g"])
                  else "DIFFERS from paper")
        )
        is_homog = hyp in ("H2a", "H2b")
        sim_label = ("mean between-author similarity"
                     if is_homog else "mean similarity")

        print(f"\n[{hyp}] {r['description']}")
        print(f"   comparison        : {r['comparison']}")
        print(f"   n (x, y)          : {r['n_x']}, {r['n_y']}")
        print(f"   {sim_label} x    : {fmt(r['mean_x'])}")
        print(f"   {sim_label} y    : {fmt(r['mean_y'])}")
        print(f"   mean difference   : {fmt(r['mean_difference_y_minus_x'])} "
              f"(y - x)")
        print(f"   permutation p     : {fmt(p, 5)}  (two-sided; "
              f"significant: {sig})")
        print(f"   Hedges' g         : {fmt(r['hedges_g'])}")
        print(f"   95% CI for g      : [{fmt(r['g_ci_95_low'])}, "
              f"{fmt(r['g_ci_95_high'])}]")
        print(f"   paper g / p       : {fmt(ref.get('g'))} / "
              f"{fmt(ref.get('p'), 5)}")
        print(f"   direction vs paper: {direction}")

    print("\n" + "-" * 72)
    print("[H3] " + h3_result["description"])
    print(f"   comparison        : {h3_result['comparison']}")
    print(f"   n tasks           : {h3_result['n']}")
    print(f"   Spearman r / p    : {fmt(spearman_r, 3)} / "
          f"{fmt(spearman_p, 5)}")
    print(f"   Pearson r / p     : {fmt(pearson_r, 3)} / "
          f"{fmt(pearson_p, 5)}")
    print(f"   binary agreement  : {fmt(agreement_rate, 3)} "
          f"(perceived vs measured 'editing helped')")

    print("\n" + "=" * 72)
    print("COMPACT TABLE")
    print("=" * 72)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    show_cols = [
        "hypothesis", "mean_x", "mean_y", "mean_difference_y_minus_x",
        "permutation_p_two_sided", "hedges_g", "g_ci_95_low", "g_ci_95_high",
        "paper_g", "significant_p_lt_.05", "direction_matches_paper",
    ]
    print(summary[show_cols].to_string(index=False))

    print("\nSaved to data/processed/hypotheses/:")
    for name in [
        "all_hypotheses_summary.csv", "h1_task_scores.csv",
        "h2_homogeneity_pairs.csv", "h3_perception_vs_measure.csv",
        "hypotheses_report.json",
    ]:
        print(f"  {name}")

    print("\nCaveats:")
    for c in report["caveats"]:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
