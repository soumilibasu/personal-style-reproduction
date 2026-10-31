"""
src/analysis/demographics.py

WP5 (part 2): does post-editing effort or success differ across
demographic groups? This mirrors the paper's H5 ("inequities in effort to
post-edit to capture individual style across demographic groups").

We build a participant-level table from the post-survey demographics and
join it to per-participant post-editing effort and success:
    effort  : mean editing time, mean number of edits, mean fraction of
              words changed
    success : mean style recovery toward the author's own control (own_gain)

Groups compared (where sample sizes allow): gender (woman vs man) and
education (4-year college degree or higher vs not). Two-group comparisons
use the Mann-Whitney U test (small, non-normal samples).

Outputs:
    data/processed/analysis/demographics_participant.csv
    data/processed/analysis/demographics_group_tests.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RAW_DIR = Path("data/raw")
DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "analysis"


def load_demographics():
    rows = []
    for f in sorted(RAW_DIR.glob("*.json")):
        with f.open(encoding="utf-8") as fh:
            d = json.load(fh)
        post = d.get("post_survey") or {}
        pid = str(d.get("user_info", {}).get("id", f.stem))
        rows.append({
            "participant_id": pid,
            "gender": post.get("gender"),
            "race": post.get("race"),
            "age": post.get("age"),
            "education": post.get("loe"),
        })
    return pd.DataFrame(rows)


def effort_success_table():
    feats = pd.read_csv(DATA_DIR / "novel" / "edit_features.csv",
                        dtype={"participant_id": str})
    own = pd.read_csv(DATA_DIR / "novel" / "own_vs_generic_task_scores.csv",
                      dtype={"participant_id": str})
    per = feats.groupby("participant_id").agg(
        mean_editing_time_s=("editing_time_s", "mean"),
        mean_n_edits=("n_edits", "mean"),
        mean_frac_words_changed=("frac_words_changed", "mean"),
    )
    per_own = own.groupby("participant_id")["own_gain"].mean().rename(
        "mean_own_gain")
    return per.join(per_own).reset_index()


def mann_whitney(df, group_col, groups, metrics):
    rows = []
    a = df[df[group_col] == groups[0]]
    b = df[df[group_col] == groups[1]]
    for m in metrics:
        xa = a[m].dropna()
        xb = b[m].dropna()
        if len(xa) < 3 or len(xb) < 3:
            continue
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        rows.append({
            "comparison": f"{group_col}: {groups[0]} vs {groups[1]}",
            "metric": m,
            f"n_{groups[0]}": int(len(xa)),
            f"n_{groups[1]}": int(len(xb)),
            f"median_{groups[0]}": float(np.median(xa)),
            f"median_{groups[1]}": float(np.median(xb)),
            "mannwhitney_u": float(u),
            "p_value": float(p),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    demo = load_demographics()
    perf = effort_success_table()
    df = demo.merge(perf, on="participant_id", how="inner")

    # Binarise education into "has 4-year degree or higher".
    def has_degree(loe):
        if not isinstance(loe, str):
            return None
        s = loe.lower()
        if any(k in s for k in ["college graduate", "master", "doctor",
                                "professional", "bachelor"]):
            return "degree"
        return "no_degree"
    df["education_bin"] = df["education"].map(has_degree)

    df.to_csv(OUT_DIR / "demographics_participant.csv", index=False)

    metrics = ["mean_editing_time_s", "mean_n_edits",
               "mean_frac_words_changed", "mean_own_gain"]
    tests = []
    tests += mann_whitney(df, "gender", ("woman", "man"), metrics)
    tests += mann_whitney(df, "education_bin", ("degree", "no_degree"),
                          metrics)
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(OUT_DIR / "demographics_group_tests.csv", index=False)

    print("=== Participant demographics summary ===")
    print(f"  participants: {len(df)}")
    print("  gender:", df["gender"].value_counts().to_dict())
    print("  education:", df["education_bin"].value_counts().to_dict())
    print("\n=== Group comparisons (Mann-Whitney U, two-sided) ===")
    if tests_df.empty:
        print("  (no groups large enough to compare)")
    else:
        print(tests_df.to_string(index=False))
    print("\nNote: small per-group samples; treat as exploratory, as the "
          "paper does for H5.")
    print("\nSaved:")
    print("  data/processed/analysis/demographics_participant.csv")
    print("  data/processed/analysis/demographics_group_tests.csv")


if __name__ == "__main__":
    main()
