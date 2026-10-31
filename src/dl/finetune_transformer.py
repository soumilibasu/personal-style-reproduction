"""
src/dl/finetune_transformer.py

WP4 (part c): fine-tune a small pretrained transformer (DistilBERT) on the
RAW TEXT to classify style as human control / LLM draft / post-edited.

Where the other models use LUAR embeddings or hand-crafted features, this
learns directly from tokens. Because fine-tuning on CPU is slow, we use a
single participant-held-out split (not 5-fold) and a few epochs; this is a
feasibility demonstration, not a tuned system. Downloads
`distilbert-base-uncased` on first run (needs internet).

Outputs:
    data/processed/dl/transformer_results.json
Run:
    python src/dl/finetune_transformer.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)

DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "dl"
MODEL_NAME = "distilbert-base-uncased"
SEED = 42
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
TEST_FRAC = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")
LABELS = {"control": 0, "draft": 1, "edited": 2}
LABEL_NAMES = ["control", "draft", "edited"]


def load_texts():
    rows = []
    with (DATA_DIR / "task_records.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r["participant_id"])
            draft = (r.get("model_generation") or "").strip()
            final = (r.get("final_version") or "").strip()
            if r.get("model_generation_shown") == 1 and draft and final:
                rows.append((pid, draft, "draft"))
                rows.append((pid, final, "edited"))
            elif r.get("model_generation_shown") == 0 and final:
                rows.append((pid, final, "control"))
    return pd.DataFrame(rows, columns=["participant_id", "text", "label"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
    except Exception as exc:  # pragma: no cover
        print(f"transformers unavailable: {exc}")
        return

    df = load_texts()

    # Participant-held-out split.
    rng = np.random.default_rng(SEED)
    parts = df["participant_id"].unique()
    rng.shuffle(parts)
    n_test = int(len(parts) * TEST_FRAC)
    test_parts = set(parts[:n_test])
    is_test = df["participant_id"].isin(test_parts)
    train_df, test_df = df[~is_test], df[is_test]
    print(f"train texts {len(train_df)} / test texts {len(test_df)} "
          f"({len(test_parts)} held-out participants)")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3).to(DEVICE)

    def encode(texts):
        return tok(list(texts), truncation=True, padding="max_length",
                   max_length=MAX_LEN, return_tensors="pt")

    y_train = torch.tensor(train_df["label"].map(LABELS).to_numpy())
    counts = np.bincount(y_train.numpy(), minlength=3)
    weight = torch.tensor(len(y_train) / (3 * counts), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

    train_texts = train_df["text"].tolist()
    model.train()
    for epoch in range(EPOCHS):
        perm = rng.permutation(len(train_texts))
        total = 0.0
        for start in range(0, len(perm), BATCH):
            idx = perm[start:start + BATCH]
            enc = encode([train_texts[i] for i in idx])
            yb = y_train[idx]
            opt.zero_grad()
            out = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{EPOCHS} mean loss "
              f"{total / (len(perm) / BATCH):.3f}")

    # Evaluate on held-out participants.
    model.eval()
    preds = []
    test_texts = test_df["text"].tolist()
    with torch.no_grad():
        for start in range(0, len(test_texts), BATCH):
            enc = encode(test_texts[start:start + BATCH])
            logits = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
            preds.extend(logits.argmax(1).tolist())
    y_test = test_df["label"].map(LABELS).to_numpy()

    results = {
        "model": f"fine-tuned {MODEL_NAME} on raw text",
        "task": "3way_style",
        "split": f"{len(test_parts)} held-out participants",
        "epochs": EPOCHS,
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "chance": 1 / 3,
    }
    with (OUT_DIR / "transformer_results.json").open("w",
                                                     encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nDistilBERT 3-way: balanced_acc={results['balanced_accuracy']:.3f}"
          f" macro_f1={results['macro_f1']:.3f}")
    print(f"Saved {OUT_DIR / 'transformer_results.json'}")


if __name__ == "__main__":
    main()
