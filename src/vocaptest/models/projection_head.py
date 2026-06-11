"""Regularized song-level projection head for frozen MERT features."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, class_count: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.30),
        )
        self.classifier = nn.Linear(projection_dim, class_count)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(inputs)
        return self.classifier(projected), projected


def supervised_contrastive_loss(
    projected: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    features = F.normalize(projected, dim=1)
    logits = features @ features.T / temperature
    identity = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
    logits_mask = ~identity
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return projected.sum() * 0.0
    mean_log_prob = (
        (positive_mask * log_prob).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return -mean_log_prob[valid].mean()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balanced_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    counts = np.bincount(labels)
    sample_weights = 1.0 / counts[labels]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=max(len(labels), batch_size * 2),
        replacement=True,
        generator=generator,
    )
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _train_epochs(
    model: ProjectionHead,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    contrastive_weight: float,
    device: str,
    seed: int,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model.train()
    for epoch in range(epochs):
        loader = _balanced_loader(features, labels, batch_size, seed + epoch)
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits, projected = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss = loss + contrastive_weight * supervised_contrastive_loss(
                projected,
                batch_labels,
            )
            loss.backward()
            optimizer.step()


def fit_projection_head(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    class_count: int,
    seed: int,
    device: str = "cuda",
    projection_dim: int = 128,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    contrastive_weight: float = 0.1,
) -> tuple[ProjectionHead, StandardScaler, int]:
    """Select epoch count internally, then retrain on the full outer train set."""
    set_deterministic_seed(seed)
    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed,
    )
    inner_train, inner_val = next(splitter.split(features, labels, groups))
    scaler = StandardScaler().fit(features[inner_train])
    train_features = scaler.transform(features[inner_train]).astype(np.float32)
    val_features = scaler.transform(features[inner_val]).astype(np.float32)

    model = ProjectionHead(features.shape[1], projection_dim, class_count).to(device)
    best_f1 = -1.0
    best_epoch = 1
    stale_epochs = 0
    val_tensor = torch.tensor(val_features, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for epoch in range(1, max_epochs + 1):
        loader = _balanced_loader(
            train_features,
            labels[inner_train],
            batch_size,
            seed + epoch,
        )
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits, projected = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss = loss + contrastive_weight * supervised_contrastive_loss(
                projected,
                batch_labels,
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(val_tensor)[0].argmax(dim=1).cpu().numpy()
        macro_f1 = f1_score(
            labels[inner_val],
            predictions,
            labels=np.arange(class_count),
            average="macro",
            zero_division=0,
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    full_scaler = StandardScaler().fit(features)
    full_features = full_scaler.transform(features).astype(np.float32)
    set_deterministic_seed(seed + 10000)
    final_model = ProjectionHead(
        features.shape[1],
        projection_dim,
        class_count,
    ).to(device)
    _train_epochs(
        final_model,
        full_features,
        labels,
        epochs=best_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        contrastive_weight=contrastive_weight,
        device=device,
        seed=seed + 10000,
    )
    final_model.eval()
    return final_model, full_scaler, best_epoch


def predict_projection_head(
    model: ProjectionHead,
    scaler: StandardScaler,
    features: np.ndarray,
    device: str,
) -> np.ndarray:
    transformed = scaler.transform(features).astype(np.float32)
    tensor = torch.tensor(transformed, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(tensor)[0], dim=1).cpu().numpy()
