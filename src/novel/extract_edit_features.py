"""
src/novel/extract_edit_features.py

NOVEL CONTRIBUTION (step 1 of 2): turn each participant's raw editing
trace into interpretable, per-task editing features.

Research question
-----------------
The original paper shows THAT post-editing moves an LLM draft toward the
author's own style. It never asks WHICH editing behaviours do that. We
extract features that describe *how* a draft was edited so that
analyze_edit_recovery.py can test whether the *kind* of editing predicts
style recovery better than the sheer *amount* of editing.

Two complementary sources are used for each treatment task:

  1. The keystroke-level edit log (`edits`): a list of
     {type: insert|delete|replace, position, removed, added, timestamp}
     operations. This gives *process* features (how much, how fast, how
     it was done).

  2. A diff between the original LLM draft (`model_generation`) and the
     participant's final text (`final_version`). This gives *linguistic*
     features (what changed: words, sentences, punctuation, voice).

Only treatment tasks (model_generation_shown == 1) with a non-empty draft
and final text are kept, which matches the 324 tasks analysed for H1.

Run from the repository root:
    python src/novel/extract_edit_features.py

Input:
    data/processed/task_records.jsonl

Output:
    data/processed/novel/edit_features.csv
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "novel"

PAUSE_THRESHOLD_S = 2.0  # an inter-keystroke gap counted as a "pause"

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENT_RE = re.compile(r"[.!?]+")
PUNCT_CHARS = set(".,;:!?\"'()-—…")
CONTRACTION_RE = re.compile(
    r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE
)
FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|we|us|our|ours|myself)\b",
                             re.IGNORECASE)


def words(text):
    return WORD_RE.findall(text or "")


def sentences(text):
    parts = [p.strip() for p in SENT_RE.split(text or "") if p.strip()]
    return parts


def type_token_ratio(tokens):
    if not tokens:
        return 0.0
    lowered = [t.lower() for t in tokens]
    return len(set(lowered)) / len(lowered)


def count_punct(text):
    return sum(1 for ch in (text or "") if ch in PUNCT_CHARS)


def uppercase_ratio(text):
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def word_diff_counts(draft, final):
    """Word-level insert/delete/replace counts via difflib."""
    a = [w.lower() for w in words(draft)]
    b = [w.lower() for w in words(final)]
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)

    inserted = deleted = replaced = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            inserted += (j2 - j1)
        elif tag == "delete":
            deleted += (i2 - i1)
        elif tag == "replace":
            replaced += max(i2 - i1, j2 - j1)

    denom = max(len(a), 1)
    frac_changed = (inserted + deleted + replaced) / denom
    similarity = matcher.ratio()  # 1.0 == final identical to draft
    return inserted, deleted, replaced, frac_changed, similarity


def process_features(edits, details_time, final_time, draft_len):
    """Features derived from the keystroke-level edit log."""
    f = {
        "n_edits": 0,
        "n_insert": 0,
        "n_delete": 0,
        "n_replace": 0,
        "chars_inserted": 0,
        "chars_deleted": 0,
        "net_char_change_log": 0,
        "insert_delete_ratio": 0.0,
        "edit_log_duration_s": 0.0,
        "n_pauses_gt_2s": 0,
        "mean_inter_edit_s": 0.0,
        "max_pause_s": 0.0,
        "edits_per_100_draft_chars": 0.0,
        "frac_draft_deleted_log": 0.0,
    }

    if not isinstance(edits, list) or not edits:
        # No logged keystrokes: fall back to the self-report timing only.
        if details_time and final_time and final_time > details_time:
            f["edit_log_duration_s"] = 0.0
        return f

    f["n_edits"] = len(edits)
    timestamps = []
    for e in edits:
        etype = e.get("type")
        if etype == "insert":
            f["n_insert"] += 1
        elif etype == "delete":
            f["n_delete"] += 1
        elif etype == "replace":
            f["n_replace"] += 1
        f["chars_inserted"] += len(e.get("added") or "")
        f["chars_deleted"] += len(e.get("removed") or "")
        try:
            timestamps.append(float(e.get("timestamp")))
        except (TypeError, ValueError):
            pass

    f["net_char_change_log"] = f["chars_inserted"] - f["chars_deleted"]
    f["insert_delete_ratio"] = (
        f["n_insert"] / f["n_delete"] if f["n_delete"] else float(f["n_insert"])
    )
    f["edits_per_100_draft_chars"] = (
        f["n_edits"] / (draft_len / 100.0) if draft_len else 0.0
    )
    f["frac_draft_deleted_log"] = (
        f["chars_deleted"] / draft_len if draft_len else 0.0
    )

    if len(timestamps) >= 2:
        timestamps.sort()
        gaps = np.diff(timestamps)
        f["edit_log_duration_s"] = float(timestamps[-1] - timestamps[0])
        f["n_pauses_gt_2s"] = int(np.sum(gaps > PAUSE_THRESHOLD_S))
        f["mean_inter_edit_s"] = float(np.mean(gaps))
        f["max_pause_s"] = float(np.max(gaps))

    return f


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "task_records.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/load_dataset.py first."
        )

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model_generation_shown") != 1:
                continue
            draft = (r.get("model_generation") or "").strip()
            final = (r.get("final_version") or "").strip()
            if not draft or not final:
                continue

            draft_words = words(draft)
            final_words = words(final)
            draft_sents = sentences(draft)
            final_sents = sentences(final)

            ins, dele, rep, frac_changed, diff_similarity = word_diff_counts(
                draft, final
            )

            proc = process_features(
                r.get("edits"),
                r.get("submit_details_time"),
                r.get("submit_final_text_time"),
                draft_len=len(draft),
            )

            # Self-reported editing time (details submitted -> final submitted).
            dt = r.get("submit_details_time")
            ft = r.get("submit_final_text_time")
            editing_time_s = (
                float(ft - dt) if (dt and ft and ft > dt) else np.nan
            )

            feat = {
                "participant_id": str(r["participant_id"]),
                "task_key": str(r["task_key"]),
                "scenario": r.get("scenario"),

                # --- Process / "how much" (from the keystroke log) ---
                **proc,
                "editing_time_s": editing_time_s,

                # --- Length ---
                "len_draft_chars": len(draft),
                "len_final_chars": len(final),
                "len_change_chars": len(final) - len(draft),
                "len_change_ratio": (
                    (len(final) - len(draft)) / len(draft) if draft else 0.0
                ),
                "n_words_draft": len(draft_words),
                "n_words_final": len(final_words),
                "word_count_change": len(final_words) - len(draft_words),

                # --- "What kind" of edit (draft vs final diff) ---
                "words_inserted": ins,
                "words_deleted": dele,
                "words_replaced": rep,
                "frac_words_changed": frac_changed,
                "draft_final_similarity": diff_similarity,

                # --- Sentence restructuring ---
                "n_sent_draft": len(draft_sents),
                "n_sent_final": len(final_sents),
                "sent_count_change": len(final_sents) - len(draft_sents),
                "avg_sentlen_change": (
                    (len(final_words) / max(len(final_sents), 1))
                    - (len(draft_words) / max(len(draft_sents), 1))
                ),

                # --- Lexical diversity ---
                "ttr_draft": type_token_ratio(draft_words),
                "ttr_final": type_token_ratio(final_words),
                "ttr_change": (
                    type_token_ratio(final_words)
                    - type_token_ratio(draft_words)
                ),

                # --- Surface style: punctuation, contractions, voice, case ---
                "punct_count_change": count_punct(final) - count_punct(draft),
                "comma_change": final.count(",") - draft.count(","),
                "period_change": final.count(".") - draft.count("."),
                "exclaim_change": final.count("!") - draft.count("!"),
                "question_change": final.count("?") - draft.count("?"),
                "contraction_change": (
                    len(CONTRACTION_RE.findall(final))
                    - len(CONTRACTION_RE.findall(draft))
                ),
                "first_person_change": (
                    len(FIRST_PERSON_RE.findall(final))
                    - len(FIRST_PERSON_RE.findall(draft))
                ),
                "uppercase_ratio_change": (
                    uppercase_ratio(final) - uppercase_ratio(draft)
                ),
            }
            rows.append(feat)

    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No treatment tasks produced features.")

    if features.duplicated(["participant_id", "task_key"]).any():
        raise ValueError("Duplicate participant/task feature rows.")

    out_path = OUT_DIR / "edit_features.csv"
    features.to_csv(out_path, index=False)

    print("Edit-feature extraction complete.")
    print(f"  Tasks with features : {len(features)}")
    print(f"  Participants        : {features['participant_id'].nunique()}")
    print(f"  Features per task   : {features.shape[1] - 3}")  # minus id cols
    print(f"  Saved               : {out_path}")
    print("\nMedian values for a few headline features:")
    headline = [
        "n_edits", "frac_words_changed", "words_replaced",
        "sent_count_change", "editing_time_s", "draft_final_similarity",
    ]
    print(features[headline].median(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
