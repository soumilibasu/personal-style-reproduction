
from pathlib import Path

import numpy as np
import pandas as pd


# Existing LUAR embedding files
EMBEDDINGS_FILE = Path("data/processed/luar_embeddings.npz")
INDEX_FILE = Path("data/processed/luar_embedding_index.csv")

OUTPUT_DIR = Path("data/processed")
N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 1_000
SEED = 42


def load_embeddings():
    data = np.load(EMBEDDINGS_FILE, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float64)
    embedding_keys = data["embedding_keys"].astype(str)

    index = pd.read_csv(INDEX_FILE)
    index["embedding_key"] = index["embedding_key"].astype(str)
    index["participant_id"] = index["participant_id"].astype(str)
    index["text_type"] = index["text_type"].astype(str)

    if len(embeddings) != len(embedding_keys):
        raise ValueError("Embedding matrix and embedding keys do not match.")

    if len(index) != len(embedding_keys):
        raise ValueError("Embedding index and embedding keys do not match.")

    if index["embedding_key"].duplicated().any():
        raise ValueError("Duplicate embedding keys found.")

    if set(index["embedding_key"]) != set(embedding_keys):
        raise ValueError("Embedding keys differ between the index and NPZ file.")

    # Align the index with the exact order of the embedding matrix.
    index = (
        index.set_index("embedding_key")
        .loc[embedding_keys]
        .reset_index()
    )

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite values found in embeddings.")

    return embeddings, index


def cosine_similarity_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms == 0):
        raise ValueError("A zero-length embedding was found.")

    normalized = embeddings / norms[:, None]
    return normalized @ normalized.T


def calculate_h1a_dash(embeddings, index):
    similarities = cosine_similarity_matrix(embeddings)

    key_to_position = {
        key: position
        for position, key in enumerate(index["embedding_key"])
    }

    controls = index[index["text_type"] == "control_reference"]
    edited = index[index["text_type"] == "edited_text"]

    control_by_participant = {}

    for _, row in controls.iterrows():
        participant = row["participant_id"]

        if participant in control_by_participant:
            raise ValueError(
                f"More than one control reference for participant {participant}."
            )

        control_by_participant[participant] = key_to_position[
            row["embedding_key"]
        ]

    if not control_by_participant:
        raise ValueError("No control-reference embeddings found.")

    if len(control_by_participant) < 2:
        raise ValueError("At least two participants are needed.")

    control_participants = set(control_by_participant)
    rows = []

    for _, row in edited.iterrows():
        participant = row["participant_id"]

        if participant not in control_by_participant:
            raise ValueError(
                f"No control reference found for participant {participant}."
            )

        edited_position = key_to_position[row["embedding_key"]]
        own_control_position = control_by_participant[participant]

        # Similarity to the participant's own combined control writing.
        own_similarity = similarities[
            edited_position, own_control_position
        ]

        # Similarity to each OTHER participant's combined control writing.
        other_positions = [
            position
            for other_participant, position in control_by_participant.items()
            if other_participant != participant
        ]

        other_similarities = similarities[
            edited_position, other_positions
        ]

        rows.append(
            {
                "participant_id": participant,
                "task_key": row["task_key"],
                "scenario": row["scenario"],
                "own_control_similarity": float(own_similarity),
                "other_controls_mean_similarity": float(
                    np.mean(other_similarities)
                ),
                "difference_other_minus_own": float(
                    np.mean(other_similarities) - own_similarity
                ),
            }
        )

    result = pd.DataFrame(rows)

    expected_tasks = edited.groupby("participant_id").size()
    if not expected_tasks.eq(4).all():
        raise ValueError(
            "Expected four edited treatment texts per participant."
        )

    return result


def paired_permutation_test(own, other, rng):
    """
    Two-tailed paired permutation test.

    For each task, randomly swap the own-control and other-control
    similarity values. The statistic is mean(other - own).
    """
    differences = other - own
    observed = differences.mean()

    exceedances = 0

    for _ in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=len(differences))
        permuted_statistic = np.mean(differences * signs)

        if abs(permuted_statistic) >= abs(observed):
            exceedances += 1

    return observed, (exceedances + 1) / (N_PERMUTATIONS + 1)


def hedges_g(other, own):
    """
    Hedges' g using other-control similarity minus own-control
    similarity. A negative value means lower similarity to others.
    """
    other = np.asarray(other, dtype=float)
    own = np.asarray(own, dtype=float)

    n1 = len(other)
    n2 = len(own)

    pooled_variance = (
        (n1 - 1) * np.var(other, ddof=1)
        + (n2 - 1) * np.var(own, ddof=1)
    ) / (n1 + n2 - 2)

    if pooled_variance <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_variance)
    cohen_d = (np.mean(other) - np.mean(own)) / pooled_sd

    # Small-sample correction.
    correction = 1 - 3 / (4 * (n1 + n2) - 9)

    return correction * cohen_d


def bootstrap_hedges_g_ci(other, own, rng):
    other = np.asarray(other, dtype=float)
    own = np.asarray(own, dtype=float)

    bootstrap_values = []

    for _ in range(N_BOOTSTRAPS):
        sampled_other = rng.choice(
            other, size=len(other), replace=True
        )
        sampled_own = rng.choice(
            own, size=len(own), replace=True
        )

        value = hedges_g(sampled_other, sampled_own)

        if np.isfinite(value):
            bootstrap_values.append(value)

    if not bootstrap_values:
        return np.nan, np.nan

    lower, upper = np.percentile(bootstrap_values, [2.5, 97.5])
    return float(lower), float(upper)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    embeddings, index = load_embeddings()
    task_results = calculate_h1a_dash(embeddings, index)

    # Each edited task contributes two paired similarity scores.
    own = task_results["own_control_similarity"].to_numpy()
    other = task_results["other_controls_mean_similarity"].to_numpy()

    mean_difference, p_value = paired_permutation_test(
        own, other, rng
    )

    # The paper's H1a' reported direction is represented here as
    # other-control similarity minus own-control similarity.
    effect_size = hedges_g(other, own)
    ci_lower, ci_upper = bootstrap_hedges_g_ci(other, own, rng)

    task_results.to_csv(
        OUTPUT_DIR / "h1a_dash_task_similarities.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "n_participants": task_results["participant_id"].nunique(),
                "n_treatment_tasks": len(task_results),
                "mean_own_control_similarity": float(np.mean(own)),
                "mean_other_controls_similarity": float(np.mean(other)),
                "mean_difference_other_minus_own": float(mean_difference),
                "permutation_p_value": float(p_value),
                "hedges_g_other_minus_own": float(effect_size),
                "hedges_g_ci_95_lower": ci_lower,
                "hedges_g_ci_95_upper": ci_upper,
                "n_permutations": N_PERMUTATIONS,
                "n_bootstraps": N_BOOTSTRAPS,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "h1a_dash_summary.csv",
        index=False,
    )

    print("\nH1a' results")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(OUTPUT_DIR / "h1a_dash_task_similarities.csv")
    print(OUTPUT_DIR / "h1a_dash_summary.csv")


if __name__ == "__main__":
    main()