"""Experimental regularized discriminant and multi-prototype classifiers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


@dataclass
class RegularizedDiscriminantClassifier:
    """Gaussian classifier with pooled/class and isotropic covariance mixing."""

    classes: np.ndarray
    means: np.ndarray
    precisions: np.ndarray
    log_determinants: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        class_covariance_weight: float,
        isotropic_weight: float,
    ) -> "RegularizedDiscriminantClassifier":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2 or len(features) != len(labels):
            raise ValueError("Expected aligned two-dimensional features and labels")
        if not 0.0 <= class_covariance_weight <= 1.0:
            raise ValueError("class_covariance_weight must be in [0, 1]")
        if not 0.0 <= isotropic_weight <= 1.0:
            raise ValueError("isotropic_weight must be in [0, 1]")

        classes = np.unique(labels)
        dimension = features.shape[1]
        means = []
        scatters = []
        sample_counts = []
        pooled_scatter = np.zeros((dimension, dimension), dtype=np.float64)
        for label in classes:
            class_features = features[labels == label]
            if len(class_features) < 2:
                raise ValueError(f"Class {label!r} needs at least two samples")
            mean = class_features.mean(axis=0)
            centered = class_features - mean
            scatter = centered.T @ centered
            means.append(mean)
            scatters.append(scatter)
            sample_counts.append(len(class_features))
            pooled_scatter += scatter

        pooled_covariance = pooled_scatter / max(len(features) - len(classes), 1)
        identity = np.eye(dimension, dtype=np.float64)
        precisions = []
        log_determinants = []
        for scatter, sample_count in zip(scatters, sample_counts):
            class_covariance = scatter / max(sample_count - 1, 1)
            covariance = (
                (1.0 - class_covariance_weight) * pooled_covariance
                + class_covariance_weight * class_covariance
            )
            average_variance = max(float(np.trace(covariance) / dimension), 1e-8)
            covariance = (
                (1.0 - isotropic_weight) * covariance
                + isotropic_weight * average_variance * identity
            )
            covariance = covariance + average_variance * 1e-6 * identity
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("Regularized covariance is not positive definite")
            precisions.append(np.linalg.inv(covariance))
            log_determinants.append(log_determinant)

        return cls(
            classes=classes,
            means=np.stack(means),
            precisions=np.stack(precisions),
            log_determinants=np.asarray(log_determinants),
        )

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        logits = []
        for mean, precision, log_determinant in zip(
            self.means,
            self.precisions,
            self.log_determinants,
        ):
            centered = features - mean
            squared_distance = np.einsum(
                "ni,ij,nj->n",
                centered,
                precision,
                centered,
            )
            logits.append(-0.5 * (squared_distance + log_determinant))
        return np.stack(logits, axis=1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.decision_function(features))


def _enforce_minimum_cluster_size(
    assignments: np.ndarray,
    features: np.ndarray,
    minimum_size: int,
) -> np.ndarray:
    assignments = assignments.copy()
    while np.bincount(assignments, minlength=2).min() < minimum_size:
        counts = np.bincount(assignments, minlength=2)
        small = int(np.argmin(counts))
        large = 1 - small
        large_indices = np.flatnonzero(assignments == large)
        centroid = features[large_indices].mean(axis=0)
        distances = np.sum((features[large_indices] - centroid) ** 2, axis=1)
        move_index = large_indices[int(np.argmax(distances))]
        assignments[move_index] = small
    return assignments


@dataclass
class DualPrototypeClassifier:
    """Two prototypes per class with shared residual variance."""

    classes: np.ndarray
    prototypes: np.ndarray
    prototype_class_indices: np.ndarray
    log_mixture_weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_variance: float
    distance_scale: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        standardize: bool,
        distance_scale: float,
        seed: int,
        minimum_cluster_size: int = 2,
    ) -> "DualPrototypeClassifier":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        feature_mean = features.mean(axis=0) if standardize else np.zeros(features.shape[1])
        feature_scale = features.std(axis=0) if standardize else np.ones(features.shape[1])
        feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
        transformed = (features - feature_mean) / feature_scale

        prototypes = []
        prototype_class_indices = []
        log_mixture_weights = []
        residual_sum = 0.0
        for class_index, label in enumerate(classes):
            class_features = transformed[labels == label]
            if len(class_features) < minimum_cluster_size * 2:
                raise ValueError(
                    f"Class {label!r} needs at least "
                    f"{minimum_cluster_size * 2} samples"
                )
            assignments = KMeans(
                n_clusters=2,
                n_init=20,
                random_state=seed + class_index,
            ).fit_predict(class_features)
            assignments = _enforce_minimum_cluster_size(
                assignments,
                class_features,
                minimum_cluster_size,
            )
            for prototype_index in range(2):
                cluster = class_features[assignments == prototype_index]
                prototype = cluster.mean(axis=0)
                prototypes.append(prototype)
                prototype_class_indices.append(class_index)
                log_mixture_weights.append(
                    np.log((len(cluster) + 1.0) / (len(class_features) + 2.0))
                )
                residual_sum += float(np.sum((cluster - prototype) ** 2))

        residual_variance = residual_sum / max(transformed.size, 1)
        residual_variance = max(residual_variance, 1e-6)
        return cls(
            classes=classes,
            prototypes=np.stack(prototypes),
            prototype_class_indices=np.asarray(prototype_class_indices),
            log_mixture_weights=np.asarray(log_mixture_weights),
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            residual_variance=residual_variance,
            distance_scale=float(distance_scale),
        )

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        transformed = (
            np.asarray(features, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        distances = np.sum(
            (transformed[:, None, :] - self.prototypes[None, :, :]) ** 2,
            axis=2,
        )
        prototype_logits = (
            -distances
            / (2.0 * self.residual_variance * self.distance_scale)
            + self.log_mixture_weights[None, :]
        )
        return np.stack([
            logsumexp(
                prototype_logits[:, self.prototype_class_indices == class_index],
                axis=1,
            )
            for class_index in range(len(self.classes))
        ], axis=1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.decision_function(features))
