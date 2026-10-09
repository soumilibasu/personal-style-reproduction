
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


INPUT_FILE = Path("data/processed/analysis_ready.jsonl")
OUTPUT_DIR = Path("data/processed")
MODEL_NAME = "rrivera1849/LUAR-MUD"

CHUNKS_PER_DOCUMENT = 16
TOKENS_PER_CHUNK = 30
MAX_TOKENS = 32
BATCH_SIZE = 4


def load_data():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    df = pd.DataFrame(rows)

    required = {
        "participant_id",
        "task_key",
        "scenario",
        "condition",
        "final_version",
        "model_generation",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["participant_id"] = df["participant_id"].astype(str)
    df["task_key"] = df["task_key"].astype(str)
    df["condition"] = pd.to_numeric(df["condition"], errors="raise")

    if not df["condition"].isin([0, 1]).all():
        raise ValueError("condition must contain only 0 and 1.")

    if df.duplicated(["participant_id", "task_key"]).any():
        raise ValueError("Duplicate participant_id/task_key pairs found.")

    return df


def split_document(text, tokenizer):
    """
    Represent one document as an episode of 16 short text snippets.

    IMPORTANT:
    This is a practical chunking strategy, not a verified reproduction
    of the paper authors' exact preprocessing.
    """
    text = str(text).strip()

    if not text:
        raise ValueError("Found an empty document.")

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    if not token_ids:
        raise ValueError("Document produced no tokens.")

    chunks = [
        token_ids[i:i + TOKENS_PER_CHUNK]
        for i in range(0, len(token_ids), TOKENS_PER_CHUNK)
    ]

    # Sample evenly across long documents rather than using only
    # their beginning.
    if len(chunks) > CHUNKS_PER_DOCUMENT:
        positions = np.linspace(
            0,
            len(chunks) - 1,
            CHUNKS_PER_DOCUMENT,
        ).round().astype(int)

        chunks = [chunks[i] for i in positions]

    text_chunks = [
        tokenizer.decode(
            chunk,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for chunk in chunks
    ]

    # LUAR-MUD requires the same episode length within a batch.
    # Empty snippets are used to pad shorter documents to 16.
    text_chunks += [""] * (
        CHUNKS_PER_DOCUMENT - len(text_chunks)
    )

    return text_chunks


def build_documents(df):
    documents = []

    for participant_id, group in df.groupby(
        "participant_id", sort=True
    ):
        controls = group[group["condition"] == 0]
        treatments = group[group["condition"] == 1]

        if len(controls) != 2 or len(treatments) != 4:
            raise ValueError(
                f"{participant_id}: expected 2 control and 4 treatment "
                f"tasks; found {len(controls)} control and "
                f"{len(treatments)} treatment tasks."
            )

        # One combined control reference per participant.
        control_texts = (
            controls.sort_values("task_key")["final_version"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if any(not text.strip() for text in control_texts):
            raise ValueError(
                f"{participant_id}: an empty control text was found."
            )

        documents.append({
            "participant_id": participant_id,
            "task_key": "combined_controls",
            "scenario": "multiple",
            "text_type": "control_reference",
            "text": "\n\n".join(control_texts),
        })

        # One AI draft and one edited text per treatment task.
        for _, row in treatments.sort_values("task_key").iterrows():
            task_key = str(row["task_key"])

            for text_type, column in [
                ("ai_draft", "model_generation"),
                ("edited_text", "final_version"),
            ]:
                value = row[column]

                if pd.isna(value) or not str(value).strip():
                    raise ValueError(
                        f"Empty {text_type}: "
                        f"participant={participant_id}, task={task_key}"
                    )

                documents.append({
                    "participant_id": participant_id,
                    "task_key": task_key,
                    "scenario": str(row["scenario"]),
                    "text_type": text_type,
                    "text": str(value),
                })

    keys = [
        f"{d['participant_id']}::{d['task_key']}::{d['text_type']}"
        for d in documents
    ]

    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate embedding keys were generated.")

    return documents


def embed_batch(batch, tokenizer, model, device):
    all_chunks = []

    for document in batch:
        chunks = split_document(document["text"], tokenizer)

        if len(chunks) != CHUNKS_PER_DOCUMENT:
            raise ValueError("A document does not have 16 snippets.")

        all_chunks.extend(chunks)

    encoded = tokenizer(
        all_chunks,
        max_length=MAX_TOKENS,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        "input_ids": encoded["input_ids"].reshape(
            len(batch), CHUNKS_PER_DOCUMENT, MAX_TOKENS
        ).to(device),
        "attention_mask": encoded["attention_mask"].reshape(
            len(batch), CHUNKS_PER_DOCUMENT, MAX_TOKENS
        ).to(device),
    }

    with torch.inference_mode():
        output = model(**inputs)

    # The model card documents a tensor of shape (batch_size, 512).
    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Unexpected model output type: {type(output)}"
        )

    expected_shape = (len(batch), 512)

    if tuple(output.shape) != expected_shape:
        raise ValueError(
            f"Expected model output {expected_shape}; "
            f"received {tuple(output.shape)}."
        )

    vectors = output.detach().float().cpu().numpy()

    if not np.isfinite(vectors).all():
        raise ValueError("Model produced non-finite embedding values.")

    return vectors


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    documents = build_documents(df)

    print(f"Tasks loaded: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Documents to embed: {len(documents)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Loading LUAR-MUD model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    vectors = []

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]

        batch_vectors = embed_batch(
            batch,
            tokenizer,
            model,
            device,
        )

        vectors.extend(batch_vectors)

        print(
            f"Embedded {len(vectors)}/{len(documents)} documents"
        )

    matrix = np.stack(vectors).astype(np.float32)

    index_df = pd.DataFrame([
        {
            "embedding_key": (
                f"{document['participant_id']}::"
                f"{document['task_key']}::"
                f"{document['text_type']}"
            ),
            "participant_id": document["participant_id"],
            "task_key": document["task_key"],
            "scenario": document["scenario"],
            "text_type": document["text_type"],
        }
        for document in documents
    ])

    # Save keys as strings, not object arrays.
    embedding_keys = index_df["embedding_key"].to_numpy(dtype=str)

    npz_path = OUTPUT_DIR / "luar_embeddings.npz"
    csv_path = OUTPUT_DIR / "luar_embedding_index.csv"

    np.savez_compressed(
        npz_path,
        embeddings=matrix,
        embedding_keys=embedding_keys,
    )

    index_df.to_csv(csv_path, index=False)

    # Verify the saved files can be read without pickle.
    with np.load(npz_path, allow_pickle=False) as saved:
        saved_vectors = saved["embeddings"]
        saved_keys = saved["embedding_keys"]

        assert saved_vectors.shape == matrix.shape
        assert np.array_equal(saved_keys, embedding_keys)

    print("\nEmbedding generation complete.")
    print(f"Embedding matrix shape: {matrix.shape}")
    print(f"Embedding key dtype: {embedding_keys.dtype}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {csv_path}")
    print("Verified: archive loads with allow_pickle=False.")
    print("No similarity or hypothesis tests were performed.")


if __name__ == "__main__":
    main()