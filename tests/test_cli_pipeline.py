"""End-to-end test of `detect parse` → `detect features` on synthetic logs."""

from __future__ import annotations

import os

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


def test_features_command_errors_when_parsed_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("log_detector.cli.PROCESSED_DIR", tmp_path)
    rc = main(["features", "--parsed", str(tmp_path / "missing.csv")])
    assert rc == 2
    out = capsys.readouterr().out
    assert "Run `detect parse` first" in out or "not found" in out


def test_unsupported_phase_command_returns_2(capsys):
    rc = main(["train"])
    assert rc == 2
    assert "not implemented yet" in capsys.readouterr().out
