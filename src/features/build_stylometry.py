"""
src/features/build_stylometry.py

WP2: interpretable stylometric features for every text in the study.

The paper measures style only through LUAR embeddings. Classic stylometry
represents style with transparent, human-readable features (function-word
rates, part-of-speech mix, punctuation habits, readability, lexical
diversity) and with markers of LLM phrasing that the paper mentions
qualitatively (em dashes, words like "delve"/"tapestry"). This script
builds that feature set so the downstream model zoo (WP3) can classify
style from features a person can actually inspect, alongside the
embeddings.

One row per text, labelled by its role:
    control : a participant's unassisted writing (control task final text)
    draft   : the LLM-generated draft of a treatment task
    edited  : the participant's post-edited final text

Run from the repository root:
    python src/features/build_stylometry.py

Input:
    data/processed/task_records.jsonl

Output:
    data/processed/features/stylometry_features.csv
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import nltk


DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "features"

# Classic function-word list (closed-class words that carry style, not
# topic). A compact but standard set.
FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "to", "from", "in", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "i", "me", "my", "we", "us", "our", "you", "your", "he", "she",
    "it", "they", "them", "this", "that", "these", "those", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "would", "should", "could", "may", "might", "must",
]

# Words/markers commonly associated with LLM output (the paper names em
# dashes and "delve"; these are the widely cited ones).
LLM_MARKER_WORDS = [
    "delve", "tapestry", "testament", "realm", "navigate", "navigating",
    "boast", "boasts", "crucial", "moreover", "furthermore", "additionally",
    "nuanced", "intricate", "pivotal", "underscore", "underscores",
    "vibrant", "bustling", "showcase", "elevate", "foster", "fostering",
    "landscape", "leverage", "seamless", "robust", "holistic", "myriad",
]

UNIVERSAL_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
                  "CONJ", "PRT", "NUM", "X", ".", "PRON"]
POS_KEEP = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
            "PRT", "NUM"]

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")


def ensure_nltk():
    """Download the tokeniser and POS tagger data if missing (quietly)."""
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng",
                "universal_tagset"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def safe_pos(tokens):
    try:
        return nltk.pos_tag(tokens, tagset="universal")
    except Exception:
        return [(t, "X") for t in tokens]


def features_for_text(text):
    text = text or ""
    words = WORD_RE.findall(text.lower())
    n_words = len(words)
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    n_sents = max(len(sents), 1)
    n_chars = len(text)

    counts = Counter(words)
    unique = len(counts)
    hapax = sum(1 for w, c in counts.items() if c == 1)

    f = {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": len(sents),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sentence_len_words": n_words / n_sents,
        "type_token_ratio": unique / n_words if n_words else 0.0,
        "hapax_ratio": hapax / n_words if n_words else 0.0,
    }

    # Punctuation rates (per 100 characters).
    per100 = 100.0 / n_chars if n_chars else 0.0
    f["comma_rate"] = text.count(",") * per100
    f["period_rate"] = text.count(".") * per100
    f["exclaim_rate"] = text.count("!") * per100
    f["question_rate"] = text.count("?") * per100
    f["semicolon_rate"] = text.count(";") * per100
    f["colon_rate"] = text.count(":") * per100
    f["emdash_rate"] = (text.count("—") + text.count("--")) * per100
    f["ellipsis_rate"] = (text.count("…") + text.count("...")) * per100
    f["curly_quote_rate"] = sum(
        text.count(c) for c in "‘’“”") * per100
    f["uppercase_ratio"] = (
        sum(1 for c in text if c.isupper())
        / sum(1 for c in text if c.isalpha()) if any(c.isalpha() for c in text)
        else 0.0)

    # Function-word rates (per word).
    per_word = 1.0 / n_words if n_words else 0.0
    for fw in FUNCTION_WORDS:
        f[f"fw_{fw}"] = counts.get(fw, 0) * per_word

    # LLM marker-word rate (per 100 words) and em dash already above.
    marker_hits = sum(counts.get(w, 0) for w in LLM_MARKER_WORDS)
    f["llm_marker_rate_per100w"] = marker_hits * (100.0 / n_words) if n_words else 0.0

    # POS mix (proportions).
    pos = safe_pos(WORD_RE.findall(text))
    pos_counts = Counter(tag for _, tag in pos)
    total_pos = sum(pos_counts.values()) or 1
    for tag in POS_KEEP:
        f[f"pos_{tag}"] = pos_counts.get(tag, 0) / total_pos

    return f


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_nltk()

    path = DATA_DIR / "task_records.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r["participant_id"])
            tk = str(r["task_key"])
            scen = r.get("scenario")
            shown = r.get("model_generation_shown")
            draft = (r.get("model_generation") or "").strip()
            final = (r.get("final_version") or "").strip()

            if shown == 1 and draft and final:
                rows.append({"participant_id": pid, "task_key": tk,
                             "scenario": scen, "label": "draft",
                             **features_for_text(draft)})
                rows.append({"participant_id": pid, "task_key": tk,
                             "scenario": scen, "label": "edited",
                             **features_for_text(final)})
            elif shown == 0 and final:
                rows.append({"participant_id": pid, "task_key": tk,
                             "scenario": scen, "label": "control",
                             **features_for_text(final)})

    df = pd.DataFrame(rows)
    out = OUT_DIR / "stylometry_features.csv"
    df.to_csv(out, index=False)

    n_feat = df.shape[1] - 4  # minus id/scenario/label
    print("Stylometry feature extraction complete.")
    print(f"  texts        : {len(df)}")
    print(f"  by label     : {df['label'].value_counts().to_dict()}")
    print(f"  participants : {df['participant_id'].nunique()}")
    print(f"  features     : {n_feat}")
    print(f"  saved        : {out}")
    print("\nMean LLM-marker rate and em-dash rate by label "
          "(stylistic AI cues):")
    print(df.groupby("label")[["llm_marker_rate_per100w", "emdash_rate"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
