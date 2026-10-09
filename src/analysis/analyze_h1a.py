
"""
src/analysis/analyze_h1a.py

H1a: Compare LUAR-MUD cosine similarity to a participant's
combined control reference for AI drafts versus edited texts.

Run from repository root:
    python src/analysis/analyze_h1a.py

Inputs:
    data/processed/luar_embeddings.npz
    data/processed/luar_embedding_index.csv

Outputs:
    data/processed/h1a_task_similarities.csv
    data/processed/h1a_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")
N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 1_000
SEED = 42


def cosine_similarity(a, b):
    """Cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        raise ValueError("Encountered a zero-norm embedding.")

    return float(np.dot(a, b) / denominator)


def hedges_g_independent(x, y):
    """
    Hedges' g for two independent groups, using pooled SD.

    Positive g means the mean of y is greater than the mean of x.
    This is the conventional independent-groups effect size.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx, ny = len(x), len(y)

    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    df = nx + ny - 2
    pooled_variance = (
        ((nx - 1) * vx + (ny - 1) * vy) / df
    )

    if pooled_variance <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_variance)
    cohens_d = (np.mean(y) - np.mean(x)) / pooled_sd

    # Small-sample correction for Hedges' g.
    correction = 1 - (3 / (4 * df - 1))

    return float(correction * cohens_d)


def permutation_test_mean_difference(x, y, rng):
    """
    Two-sided permutation test for difference in means.

    Observed statistic = mean(y) - mean(x).
    Labels are shuffled across the pooled task-level observations.

    This is an independent-observation permutation procedure.
    It does not preserve participant clusters.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    observed = np.mean(y) - np.mean(x)
    pooled = np.concatenate([x, y])
    nx = len(x)

    count = 0

    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(pooled)

        permuted_x = shuffled[:nx]
        permuted_y = shuffled[nx:]

        statistic = np.mean(permuted_y) - np.mean(permuted_x)

        if abs(statistic) >= abs(observed):
            count += 1

    # Plus-one correction prevents reporting a zero p-value.
    p_value = (count + 1) / (N_PERMUTATIONS + 1)

    return float(observed), float(p_value)


def bootstrap_hedges_g_ci(x, y, rng):
    """
    Percentile 95% CI for independent-groups Hedges' g.

    Resamples each group independently. This does not preserve
    participant-level clustering.
    """
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


def main():
    rng = np.random.default_rng(SEED)

    embeddings_path = DATA_DIR / "luar_embeddings.npz"
    index_path = DATA_DIR / "luar_embedding_index.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing {embeddings_path}. Run src/embed_luar.py first."
        )

    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}.")

    # Keys are saved as strings by the updated embed_luar.py.
    with np.load(embeddings_path, allow_pickle=False) as archive:
        vectors = archive["embeddings"]
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

    if vectors.ndim != 2 or vectors.shape[1] != 512:
        raise ValueError(
            f"Expected an (N, 512) matrix; got {vectors.shape}."
        )

    if len(vectors) != len(keys) or len(keys) != len(index):
        raise ValueError(
            "Embedding matrix, key array, and index row counts differ."
        )

    if not np.array_equal(
        keys, index["embedding_key"].to_numpy()
    ):
        raise ValueError(
            "Embedding keys and index rows are misaligned."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    if index["embedding_key"].duplicated().any():
        raise ValueError("Duplicate embedding keys found.")

    vector_by_key = dict(zip(keys, vectors))
    results = []

    # Find one combined control reference for each participant.
    for participant_id, group in index.groupby(
        "participant_id", sort=True
    ):
        control_rows = group[
            group["text_type"] == "control_reference"
        ]

        if len(control_rows) != 1:
            raise ValueError(
                f"{participant_id}: expected one control reference; "
                f"found {len(control_rows)}."
            )

        control_key = control_rows.iloc[0]["embedding_key"]
        control_vector = vector_by_key[control_key]

        treatment = group[
            group["text_type"].isin(["ai_draft", "edited_text"])
        ]

        for task_key, task_group in treatment.groupby(
            "task_key", sort=True
        ):
            draft_rows = task_group[
                task_group["text_type"] == "ai_draft"
            ]

            edited_rows = task_group[
                task_group["text_type"] == "edited_text"
            ]

            if len(draft_rows) != 1 or len(edited_rows) != 1:
                raise ValueError(
                    f"{participant_id}/{task_key}: expected one draft "
                    "and one edited text."
                )

            draft = draft_rows.iloc[0]
            edited = edited_rows.iloc[0]

            draft_similarity = cosine_similarity(
                vector_by_key[draft["embedding_key"]],
                control_vector,
            )

            edited_similarity = cosine_similarity(
                vector_by_key[edited["embedding_key"]],
                control_vector,
            )

            results.append({
                "participant_id": participant_id,
                "task_key": task_key,
                "scenario": draft["scenario"],
                "draft_to_control": draft_similarity,
                "edited_to_control": edited_similarity,
                "change_after_editing": (
                    edited_similarity - draft_similarity
                ),
            })

    scores = pd.DataFrame(results)

    if scores.empty:
        raise ValueError("No treatment-task comparisons were generated.")

    if scores["task_key"].duplicated().any():
        # Task keys may repeat across participants, so check the pair.
        if scores.duplicated(
            ["participant_id", "task_key"]
        ).any():
            raise ValueError("Duplicate participant/task comparisons.")

    scores.to_csv(
        DATA_DIR / "h1a_task_similarities.csv",
        index=False,
    )

    # Paper-style task-level group comparison:
    # compare the 324 draft similarities with the 324 edited similarities.
    draft_scores = scores["draft_to_control"].to_numpy()
    edited_scores = scores["edited_to_control"].to_numpy()

    observed_difference, p_value = permutation_test_mean_difference(
        draft_scores,
        edited_scores,
        rng,
    )

    effect_size = hedges_g_independent(
        draft_scores,
        edited_scores,
    )

    ci_low, ci_high = bootstrap_hedges_g_ci(
        draft_scores,
        edited_scores,
        rng,
    )

    summary = pd.DataFrame([{
        "n_participants": scores["participant_id"].nunique(),
        "n_treatment_tasks": len(scores),
        "mean_draft_to_control": float(np.mean(draft_scores)),
        "mean_edited_to_control": float(np.mean(edited_scores)),
        "mean_change_after_editing": float(observed_difference),
        "permutation_p_two_sided": p_value,
        "hedges_g_edited_vs_draft": effect_size,
        "g_ci_95_low": ci_low,
        "g_ci_95_high": ci_high,
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap_samples": N_BOOTSTRAPS,
        "random_seed": SEED,
    }])

    summary.to_csv(
        DATA_DIR / "h1a_summary.csv",
        index=False,
    )

    print("\nH1a analysis complete")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print("data/processed/h1a_task_similarities.csv")
    print("data/processed/h1a_summary.csv")

    print(
        "\nInterpretation note: the test and bootstrap treat task scores "
        "as independent. Multiple tasks come from each participant, so "
        "this is not necessarily the paper's exact inferential procedure."
    )


if __name__ == "__main__":
    main()