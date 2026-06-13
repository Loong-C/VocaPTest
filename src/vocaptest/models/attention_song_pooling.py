"""Small supervised attention pooling head for frozen segment embeddings."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class AttentionSongClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, class_count: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(input_dim, class_count)

    def pool(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        *,
        segment_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_mask = mask
        if self.training and segment_dropout > 0:
            keep = mask & (
                torch.rand(mask.shape, device=mask.device) >= segment_dropout
            )
            empty = ~keep.any(dim=1)
            keep[empty] = mask[empty]
            active_mask = keep

        scores = self.attention(inputs).squeeze(-1)
        scores = scores.masked_fill(~active_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(inputs * weights.unsqueeze(-1), dim=1)
        pooled = F.normalize(pooled, dim=1)
        return pooled, weights

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        *,
        segment_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, weights = self.pool(
            inputs,
            mask,
            segment_dropout=segment_dropout,
        )
        return self.classifier(pooled), weights


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=max(len(labels), batch_size * 2),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.bool),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        sampler=sampler,
    )


def _attention_uniformity_penalty(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    segment_counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    log_uniform = -torch.log(segment_counts)
    log_weights = torch.log(weights.clamp_min(1e-8))
    return (
        weights * (log_weights - log_uniform) * mask
    ).sum(dim=1).mean()


def _train_epochs(
    model: AttentionSongClassifier,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    device: str,
    seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    segment_dropout: float,
    uniformity_weight: float,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_masks, batch_labels in _loader(
            features,
            masks,
            labels,
            batch_size=batch_size,
            seed=seed + epoch,
        ):
            batch_features = batch_features.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits, attention = model(
                batch_features,
                batch_masks,
                segment_dropout=segment_dropout,
            )
            loss = F.cross_entropy(logits, batch_labels)
            loss = loss + uniformity_weight * _attention_uniformity_penalty(
                attention,
                batch_masks,
            )
            loss.backward()
            optimizer.step()


def fit_attention_song_classifier(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    class_count: int,
    seed: int,
    device: str = "cuda",
    hidden_dim: int = 32,
    max_epochs: int = 120,
    patience: int = 15,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 2e-3,
    segment_dropout: float = 0.15,
    uniformity_weight: float = 0.02,
) -> tuple[AttentionSongClassifier, int]:
    """Select epoch count internally, then retrain on the full outer train set."""
    _set_seed(seed)
    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed,
    )
    inner_train, inner_validation = next(
        splitter.split(features, labels, groups)
    )
    model = AttentionSongClassifier(
        features.shape[2],
        hidden_dim,
        class_count,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    validation_features = torch.tensor(
        features[inner_validation],
        dtype=torch.float32,
        device=device,
    )
    validation_masks = torch.tensor(
        masks[inner_validation],
        dtype=torch.bool,
        device=device,
    )
    best_f1 = -1.0
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_features, batch_masks, batch_labels in _loader(
            features[inner_train],
            masks[inner_train],
            labels[inner_train],
            batch_size=batch_size,
            seed=seed + epoch,
        ):
            batch_features = batch_features.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits, attention = model(
                batch_features,
                batch_masks,
                segment_dropout=segment_dropout,
            )
            loss = F.cross_entropy(logits, batch_labels)
            loss = loss + uniformity_weight * _attention_uniformity_penalty(
                attention,
                batch_masks,
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(
                validation_features,
                validation_masks,
            )[0].argmax(dim=1).cpu().numpy()
        score = f1_score(
            labels[inner_validation],
            predictions,
            labels=np.arange(class_count),
            average="macro",
            zero_division=0,
        )
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    _set_seed(seed + 10000)
    final_model = AttentionSongClassifier(
        features.shape[2],
        hidden_dim,
        class_count,
    ).to(device)
    _train_epochs(
        final_model,
        features,
        masks,
        labels,
        epochs=best_epoch,
        device=device,
        seed=seed + 10000,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        segment_dropout=segment_dropout,
        uniformity_weight=uniformity_weight,
    )
    final_model.eval()
    return final_model, best_epoch


def predict_attention_song_classifier(
    model: AttentionSongClassifier,
    features: np.ndarray,
    masks: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        logits, attention = model(feature_tensor, mask_tensor)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.cpu().numpy(), attention.cpu().numpy()


def pool_attention_song_features(
    model: AttentionSongClassifier,
    features: np.ndarray,
    masks: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        pooled, attention = model.pool(feature_tensor, mask_tensor)
    return pooled.cpu().numpy(), attention.cpu().numpy()
