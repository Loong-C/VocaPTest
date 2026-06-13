"""Tests for experimental discriminant classifiers."""
import numpy as np

from vocaptest.models.discriminant_variants import (
    DualPrototypeClassifier,
    RegularizedDiscriminantClassifier,
)


def test_regularized_discriminant_probabilities_are_normalized():
    rng = np.random.default_rng(3)
    features = np.vstack([
        rng.normal(-1.0, 0.3, size=(8, 4)),
        rng.normal(1.0, 0.3, size=(8, 4)),
    ])
    labels = np.array(["a"] * 8 + ["b"] * 8)

    model = RegularizedDiscriminantClassifier.fit(
        features,
        labels,
        class_covariance_weight=0.5,
        isotropic_weight=0.5,
    )
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (16, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(model.classes[probabilities.argmax(axis=1)] == labels) > 0.9


def test_diagonal_regularized_discriminant_handles_wide_features():
    rng = np.random.default_rng(5)
    features = np.vstack([
        rng.normal(-0.5, 0.4, size=(6, 20)),
        rng.normal(0.5, 0.4, size=(6, 20)),
    ])
    labels = np.array(["a"] * 6 + ["b"] * 6)

    model = RegularizedDiscriminantClassifier.fit(
        features,
        labels,
        class_covariance_weight=0.5,
        isotropic_weight=0.1,
        diagonal=True,
    )

    assert model.precisions.shape == (2, 20)
    assert np.mean(
        model.classes[model.predict_proba(features).argmax(axis=1)] == labels
    ) > 0.9


def test_dual_prototype_classifier_models_bimodal_classes():
    rng = np.random.default_rng(7)
    features = np.vstack([
        rng.normal([-2.0, 0.0], 0.15, size=(5, 2)),
        rng.normal([2.0, 0.0], 0.15, size=(5, 2)),
        rng.normal([0.0, -2.0], 0.15, size=(5, 2)),
        rng.normal([0.0, 2.0], 0.15, size=(5, 2)),
    ])
    labels = np.array(["a"] * 10 + ["b"] * 10)

    model = DualPrototypeClassifier.fit(
        features,
        labels,
        standardize=True,
        distance_scale=1.0,
        seed=11,
    )
    probabilities = model.predict_proba(features)

    assert model.prototypes.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(model.classes[probabilities.argmax(axis=1)] == labels) > 0.9
