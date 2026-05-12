"""Tests for the anomaly detector models."""

from __future__ import annotations

import numpy as np
import pytest

from log_detector.models import build_detector, load_detector


def _make_toy_data(rng: np.random.Generator, n_normal: int = 80, n_anom: int = 20):
    """Count-style features: low counts for normal sessions, much higher for anomalies.
    Mirrors the shape of real per-session template-count vectors (always >= 0)."""
    normal = rng.integers(0, 3, size=(n_normal, 4))
    anomalies = rng.integers(15, 25, size=(n_anom, 4))
    X = np.vstack([normal, anomalies]).astype(np.int32)
    y = np.concatenate([np.zeros(n_normal, dtype=np.int8), np.ones(n_anom, dtype=np.int8)])
    return X, y


# ---------------------------------------------------------------------------
# IForest
# ---------------------------------------------------------------------------

def test_iforest_fit_score_ranks_anomalies_higher():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X, y = _make_toy_data(rng)

    det = build_detector("iforest", n_estimators=50, random_state=0)
    det.fit(X)
    scores = det.score(X)
    assert scores.shape == (len(X),)
    # Anomalies should have noticeably higher mean score than normals.
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_iforest_save_load_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X, _ = _make_toy_data(rng)

    det = build_detector("iforest", n_estimators=50, random_state=0)
    det.fit(X)
    expected = det.score(X)

    path = tmp_path / "iforest.bin"
    det.save(path)
    reloaded = load_detector(path)
    np.testing.assert_allclose(reloaded.score(X), expected)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

def test_autoencoder_fit_score_ranks_anomalies_higher():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    X, y = _make_toy_data(rng)

    # Canonical AE-based AD: train on (mostly) normal data, score on everything.
    X_normal = X[y == 0]
    det = build_detector("autoencoder", epochs=80, batch_size=16, lr=5e-3, seed=0)
    det.fit(X_normal)
    scores = det.score(X)
    assert scores.shape == (len(X),)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_autoencoder_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    X, _ = _make_toy_data(rng)

    det = build_detector("autoencoder", epochs=10, batch_size=16, lr=5e-3, seed=0)
    det.fit(X)
    expected = det.score(X)

    path = tmp_path / "ae.pt"
    det.save(path)
    reloaded = load_detector(path)
    np.testing.assert_allclose(reloaded.score(X), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_build_detector_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector("not_a_real_model")
