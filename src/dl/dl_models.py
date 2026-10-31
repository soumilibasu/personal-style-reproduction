"""
src/dl/dl_models.py

WP4 (parts a & b): deep-learning models.

(a) MLP on LUAR embeddings for the three-way style task (control / draft /
    edited), a neural counterpart to the classical model zoo.

(b) A recurrent model over the RAW KEYSTROKE EDIT STREAM. Each treatment
    task is a sequence of edit operations; an LSTM reads that sequence and
    predicts whether the edit successfully moved the text toward the
    author's own style (own_gain > 0). This uses the logs as a time series,
    which neither the paper nor our other analyses do.

Both are evaluated out-of-fold with GroupKFold by participant.

Outputs:
    data/processed/dl/dl_results.json
Run:
    python src/dl/dl_models.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "dl"
SEED = 42
N_SPLITS = 5
MAX_STEPS = 200  # keystroke sequence length cap

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")
CLASS_MAP = {"control_reference": 0, "ai_draft": 1, "edited_text": 2}
CLASS_NAMES = ["control", "draft", "edited"]


# --------------------------------------------------------------------------
# (a) MLP on embeddings
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def train_eval_mlp():
    with np.load(DATA_DIR / "luar_embeddings.npz", allow_pickle=False) as a:
        X = a["embeddings"].astype(np.float32)
        keys = a["embedding_keys"].astype(str)
    index = pd.read_csv(DATA_DIR / "luar_embedding_index.csv",
                        dtype={"embedding_key": str, "participant_id": str})
    assert np.array_equal(keys, index["embedding_key"].to_numpy())
    y = index["text_type"].map(CLASS_MAP).to_numpy()
    groups = index["participant_id"].to_numpy()

    oof = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(N_SPLITS).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32)
        Xte = torch.tensor(scaler.transform(X[te]), dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.long)

        # Class weights for imbalance.
        counts = np.bincount(y[tr], minlength=3)
        w = torch.tensor(len(y[tr]) / (3 * counts), dtype=torch.float32)

        model = MLP(X.shape[1], 3).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=w)
        model.train()
        for _ in range(80):
            opt.zero_grad()
            out = model(Xtr)
            loss = lossf(out, ytr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            oof[te] = model(Xte).argmax(1).numpy()

    return {
        "model": "MLP(256-128) on LUAR embeddings",
        "task": "3way_style",
        "accuracy": float(accuracy_score(y, oof)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "macro_f1": float(f1_score(y, oof, average="macro")),
        "chance": 1 / 3,
    }


# --------------------------------------------------------------------------
# (b) LSTM over the keystroke edit stream
# --------------------------------------------------------------------------
def build_sequences():
    """One padded (MAX_STEPS, 7) tensor per treatment task + binary target."""
    own = pd.read_csv(DATA_DIR / "novel" / "own_vs_generic_task_scores.csv",
                      dtype={"participant_id": str, "task_key": str})
    target = {(r.participant_id, r.task_key): int(r.own_gain > 0)
              for r in own.itertuples()}

    seqs, ys, groups = [], [], []
    with (DATA_DIR / "task_records.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model_generation_shown") != 1:
                continue
            pid, tk = str(r["participant_id"]), str(r["task_key"])
            key = (pid, tk)
            edits = r.get("edits")
            if key not in target or not isinstance(edits, list) or not edits:
                continue
            draft_len = max(len(r.get("model_generation") or ""), 1)

            # Uniformly subsample long sequences, keeping overall shape.
            if len(edits) > MAX_STEPS:
                idx = np.linspace(0, len(edits) - 1, MAX_STEPS).round().astype(int)
                sel = [edits[i] for i in idx]
            else:
                sel = edits

            ts_prev = None
            feats = []
            for e in sel:
                etype = e.get("type")
                try:
                    ts = float(e.get("timestamp"))
                except (TypeError, ValueError):
                    ts = ts_prev if ts_prev is not None else 0.0
                dt = 0.0 if ts_prev is None else min(ts - ts_prev, 10.0)
                ts_prev = ts
                feats.append([
                    1.0 if etype == "insert" else 0.0,
                    1.0 if etype == "delete" else 0.0,
                    1.0 if etype == "replace" else 0.0,
                    min(len(e.get("added") or "") / 5.0, 5.0),
                    min(len(e.get("removed") or "") / 5.0, 5.0),
                    dt,
                    min(float(e.get("position", 0)) / draft_len, 3.0),
                ])
            arr = np.zeros((MAX_STEPS, 7), dtype=np.float32)
            arr[:len(feats)] = np.array(feats, dtype=np.float32)
            seqs.append(arr)
            ys.append(target[key])
            groups.append(pid)

    return (np.stack(seqs), np.array(ys, dtype=np.float32),
            np.array(groups))


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim=7, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


def train_eval_lstm():
    X, y, groups = build_sequences()
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(X, y, groups):
        Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
        Xte = torch.tensor(X[te])
        pos = max(ytr.sum().item(), 1)
        neg = max(len(ytr) - ytr.sum().item(), 1)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

        model = LSTMClassifier().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model.train()
        for _ in range(30):
            opt.zero_grad()
            loss = lossf(model(Xtr), ytr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(Xte)).numpy()

    pred = (oof > 0.5).astype(int)
    majority = max(y.mean(), 1 - y.mean())
    return {
        "model": "LSTM over keystroke edit stream",
        "task": "predict successful edit (own_gain > 0)",
        "n_tasks": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, oof)),
        "majority_baseline_accuracy": float(majority),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Training MLP on embeddings (3-way) ...")
    mlp = train_eval_mlp()
    print(f"  balanced_acc={mlp['balanced_accuracy']:.3f} "
          f"macro_f1={mlp['macro_f1']:.3f} (chance 0.333)")

    print("Training LSTM over keystroke stream (successful-edit) ...")
    lstm = train_eval_lstm()
    print(f"  AUC={lstm['roc_auc']:.3f} acc={lstm['accuracy']:.3f} "
          f"(majority {lstm['majority_baseline_accuracy']:.3f})")

    results = {"mlp_embeddings": mlp, "lstm_keystrokes": lstm, "seed": SEED}
    with (OUT_DIR / "dl_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {OUT_DIR / 'dl_results.json'}")


if __name__ == "__main__":
    main()
