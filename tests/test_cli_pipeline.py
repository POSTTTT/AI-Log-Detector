"""End-to-end tests for the `detect` CLI on synthetic logs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("drain3")

from log_detector.cli import main  # noqa: E402
from log_detector.features import FeatureMatrix  # noqa: E402

from .fixtures import SYNTHETIC_HDFS_LOG, SYNTHETIC_LABELS_CSV  # noqa: E402


def test_parse_then_features_end_to_end(tmp_path, monkeypatch):
    log_path = tmp_path / "mini.log"
    log_path.write_text(SYNTHETIC_HDFS_LOG, encoding="utf-8")
    labels_path = tmp_path / "anomaly_label.csv"
    labels_path.write_text(SYNTHETIC_LABELS_CSV, encoding="utf-8")

    parsed_csv = tmp_path / "parsed.csv"
    features_dir = tmp_path / "processed"
    features_dir.mkdir()

    monkeypatch.chdir(tmp_path)
    # Redirect PROCESSED_DIR by patching the module attribute used by the CLI.
    monkeypatch.setattr(
        "log_detector.cli.PROCESSED_DIR",
        features_dir,
    )

    rc = main(["parse", "--input", str(log_path), "--output", str(parsed_csv)])
    assert rc == 0
    assert parsed_csv.exists()

    rc = main(
        [
            "features",
            "--parsed", str(parsed_csv),
            "--labels", str(labels_path),
            "--test-size", "0.25",
            "--seed", "1",
        ]
    )
    assert rc == 0

    train = FeatureMatrix.load(features_dir / "features_train.npz")
    test = FeatureMatrix.load(features_dir / "features_test.npz")
    # All 4 synthetic blocks are labeled.
    assert train.X.shape[0] + test.X.shape[0] == 4
    # Both splits non-empty.
    assert train.X.shape[0] >= 1 and test.X.shape[0] >= 1
    # Labels are 0/1 integers.
    assert set(np.unique(train.y)).issubset({0, 1})
    assert set(np.unique(test.y)).issubset({0, 1})


def test_train_then_score_end_to_end(tmp_path, monkeypatch, capsys):
    pytest.importorskip("sklearn")

    # Build a small labeled feature matrix directly (skipping parse/features for speed).
    rng = np.random.default_rng(0)
    n_normal, n_anom, n_feat = 60, 10, 5
    X = np.vstack(
        [
            rng.integers(0, 3, size=(n_normal, n_feat)),
            rng.integers(10, 20, size=(n_anom, n_feat)),
        ]
    ).astype(np.int32)
    y = np.concatenate(
        [np.zeros(n_normal, dtype=np.int8), np.ones(n_anom, dtype=np.int8)]
    )
    fm = FeatureMatrix(
        X=X,
        y=y,
        session_ids=np.array([f"blk_{i}" for i in range(len(y))]),
        event_ids=np.arange(n_feat),
    )
    features_dir = tmp_path / "processed"
    features_dir.mkdir()
    train_path = features_dir / "features_train.npz"
    test_path = features_dir / "features_test.npz"
    fm.save(train_path)
    fm.save(test_path)

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr("log_detector.cli.PROCESSED_DIR", features_dir)
    monkeypatch.setattr("log_detector.cli.MODELS_DIR", models_dir)
    monkeypatch.setattr("log_detector.cli.REPORTS_DIR", reports_dir)

    model_path = models_dir / "iforest.bin"
    rc = main(["train", "--model", "iforest", "--features", str(train_path),
               "--output", str(model_path)])
    assert rc == 0
    assert model_path.exists()

    scores_path = reports_dir / "scores.csv"
    metrics_path = reports_dir / "metrics.json"
    rc = main(["score", "--model", str(model_path), "--features", str(test_path),
               "--scores-out", str(scores_path), "--metrics-out", str(metrics_path)])
    assert rc == 0
    assert scores_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    # On well-separated synthetic data, F1 should be excellent.
    assert metrics["f1"] >= 0.9
    assert metrics["n"] == len(y)
    assert metrics["n_anomalies"] == n_anom


def test_features_command_errors_when_parsed_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("log_detector.cli.PROCESSED_DIR", tmp_path)
    rc = main(["features", "--parsed", str(tmp_path / "missing.csv")])
    assert rc == 2
    out = capsys.readouterr().out
    assert "Run `detect parse` first" in out or "not found" in out


