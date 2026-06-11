"""Train a lightweight MLP probe on frozen embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.features.extract_embeddings import load_all_embeddings_aligned
from vocaptest.utils.logging import setup_logging

logger = setup_logging()


class MLPProbe(nn.Module):
    """Simple MLP for producer classification from frozen embeddings."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(
    records: list[EmbeddingRecord],
    train_song_ids: list[str],
    val_song_ids: list[str],
    output_dir: str | Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
    patience: int = 10,
) -> dict:
    """Train an MLP probe and return evaluation metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings, loaded_records = load_all_embeddings_aligned(records)
    if not loaded_records:
        raise ValueError("No readable embeddings were supplied")

    le = LabelEncoder()
    y = le.fit_transform([record.producer_slug for record in loaded_records])
    num_classes = len(le.classes_)

    # Split by song
    train_ids = set(train_song_ids)
    val_ids = set(val_song_ids)
    train_mask = np.array([r.song_id in train_ids for r in loaded_records])
    val_mask = np.array([r.song_id in val_ids for r in loaded_records])

    X_train = torch.tensor(embeddings[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.long)
    X_val = torch.tensor(embeddings[val_mask], dtype=torch.float32)
    y_val = torch.tensor(y[val_mask], dtype=torch.long)
    val_song_ids_aligned = [
        record.song_id for record, selected in zip(loaded_records, val_mask)
        if selected
    ]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = MLPProbe(embeddings.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_song_acc": [], "val_song_macro_f1": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_logits.append(model(xb).cpu())

        logits = torch.cat(val_logits, dim=0).numpy()
        song_logits: dict[str, list[np.ndarray]] = {}
        song_labels: dict[str, int] = {}
        for song_id, logit, label in zip(val_song_ids_aligned, logits, y[val_mask]):
            song_logits.setdefault(song_id, []).append(logit)
            previous = song_labels.setdefault(song_id, int(label))
            if previous != int(label):
                raise ValueError(f"Song {song_id} has conflicting validation labels")

        ordered_songs = sorted(song_logits)
        true_labels = np.array([song_labels[song_id] for song_id in ordered_songs])
        predictions = np.array([
            np.mean(song_logits[song_id], axis=0).argmax()
            for song_id in ordered_songs
        ])
        val_acc = float(np.mean(predictions == true_labels))
        val_macro_f1 = float(f1_score(
            true_labels,
            predictions,
            labels=np.arange(num_classes),
            average="macro",
            zero_division=0,
        ))

        history["train_loss"].append(total_loss / len(train_dl))
        history["val_song_acc"].append(val_acc)
        history["val_song_macro_f1"].append(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d | loss=%.4f | song_acc=%.4f | song_macro_f1=%.4f",
                epoch,
                total_loss / len(train_dl),
                val_acc,
                val_macro_f1,
            )
        if epochs_without_improvement >= patience:
            logger.info("Early stopping after %d epochs without improvement", patience)
            break

    logger.info("Best song-level validation macro-F1: %.4f", best_val_macro_f1)

    # Save label encoder
    import pickle
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return {
        "best_val_song_macro_f1": best_val_macro_f1,
        "num_classes": num_classes,
        "history": history,
    }
