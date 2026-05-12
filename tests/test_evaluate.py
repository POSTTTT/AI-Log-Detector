"""Tests for evaluation metrics."""

from __future__ import annotations

import json

import numpy as np

from log_detector.evaluate import (
    best_f1_threshold,
    evaluate,
    pr_auc,
    precision_recall_f1,
    roc_auc,
)


def test_precision_recall_f1_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.95])
    p, r, f1 = precision_recall_f1(y, s, threshold=0.5)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_precision_recall_f1_all_predicted_anomalies():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.95, 0.85, 0.99])
    p, r, f1 = precision_recall_f1(y, s, threshold=0.0)
    # Predict all positive: 2 TP, 2 FP, 0 FN.
    assert p == 0.5 and r == 1.0
    assert abs(f1 - 2 / 3) < 1e-9


def test_best_f1_threshold_finds_optimum():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.4, 0.5, 0.7, 0.9])
    t, f1 = best_f1_threshold(y, s)
    # Threshold at 0.5 gets all 3 positives, no false positives.
    assert f1 == 1.0
    assert t == 0.5


def test_roc_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    s_perfect = np.array([0.1, 0.2, 0.9, 0.95])
    assert roc_auc(y, s_perfect) == 1.0

    rng = np.random.default_rng(0)
    y_big = np.array([0] * 500 + [1] * 500)
    s_random = rng.random(1000)
    # Random scores should give AUC ~0.5 — assert loosely.
    assert 0.4 < roc_auc(y_big, s_random) < 0.6


def test_pr_auc_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.95])
    assert pr_auc(y, s) == 1.0


def test_pr_auc_handles_no_positives():
    y = np.array([0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.95])
    assert pr_auc(y, s) == 0.0


def test_evaluate_report_round_trip(tmp_path):
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.95])
    report = evaluate(y, s)
    assert report.n == 4
    assert report.n_anomalies == 2
    assert report.f1 == 1.0
    out = tmp_path / "metrics.json"
    report.to_json(out)
    loaded = json.loads(out.read_text())
    assert loaded["f1"] == 1.0
    assert set(loaded.keys()) >= {
        "n", "n_anomalies", "anomaly_rate", "threshold",
        "precision", "recall", "f1", "roc_auc", "pr_auc",
    }
