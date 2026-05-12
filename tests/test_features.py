"""Tests for the feature matrix builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from log_detector.features import (
    FeatureMatrix,
    attach_labels,
    build_count_matrix,
    load_hdfs_labels,
    stratified_split,
)

from .fixtures import SYNTHETIC_LABELS, SYNTHETIC_LABELS_CSV


def _toy_sessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block_id": ["blk_a", "blk_b", "blk_c"],
            "event_sequence": [[1, 1, 2], [2, 3], [1, 3, 3, 3]],
        }
    )


def test_build_count_matrix_shape_and_counts():
    fm = build_count_matrix(_toy_sessions())
    assert fm.X.shape == (3, 3)
    assert list(fm.event_ids) == [1, 2, 3]
    np.testing.assert_array_equal(
        fm.X,
        np.array(
            [
                [2, 1, 0],  # blk_a: 1×2, 2×1
                [0, 1, 1],  # blk_b: 2×1, 3×1
                [1, 0, 3],  # blk_c: 1×1, 3×3
            ],
            dtype=np.int32,
        ),
    )
    assert list(fm.session_ids) == ["blk_a", "blk_b", "blk_c"]
    assert fm.y is None


def test_build_count_matrix_respects_fixed_vocab():
    fm = build_count_matrix(_toy_sessions(), event_vocab=np.array([1, 2, 99]))
    # Event 3 is unknown and should be dropped; event 99 is unseen → all zeros.
    assert list(fm.event_ids) == [1, 2, 99]
    np.testing.assert_array_equal(
        fm.X,
        np.array(
            [
                [2, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_load_hdfs_labels(tmp_path):
    csv = tmp_path / "anomaly_label.csv"
    csv.write_text(SYNTHETIC_LABELS_CSV, encoding="utf-8")
    labels = load_hdfs_labels(csv)
    assert labels == SYNTHETIC_LABELS


def test_attach_labels_filters_and_orders():
    fm = build_count_matrix(_toy_sessions())
    labels = {"blk_a": 0, "blk_c": 1}  # blk_b missing — should be dropped
    out = attach_labels(fm, labels)
    assert list(out.session_ids) == ["blk_a", "blk_c"]
    assert list(out.y) == [0, 1]
    assert out.X.shape == (2, 3)


def test_stratified_split_preserves_classes():
    rng = np.random.default_rng(0)
    n = 100
    fm = FeatureMatrix(
        X=rng.integers(0, 5, size=(n, 4)).astype(np.int32),
        y=np.array([1] * 10 + [0] * 90, dtype=np.int8),
        session_ids=np.array([f"blk_{i}" for i in range(n)]),
        event_ids=np.array([1, 2, 3, 4]),
    )
    train, test = stratified_split(fm, test_size=0.2, seed=0)
    assert train.X.shape[0] + test.X.shape[0] == n
    # Both classes appear in both splits.
    assert set(train.y) == {0, 1}
    assert set(test.y) == {0, 1}
    # Test split is roughly 20%.
    assert 0.15 <= test.X.shape[0] / n <= 0.25


def test_stratified_split_handles_tiny_inputs():
    """Regression: with very small inputs (1 sample per class), index arrays
    can end up empty — they must still be int dtype so X[indices] works."""
    fm = FeatureMatrix(
        X=np.array([[1, 0], [0, 1]], dtype=np.int32),
        y=np.array([0, 1], dtype=np.int8),
        session_ids=np.array(["blk_a", "blk_b"]),
        event_ids=np.array([1, 2]),
    )
    train, test = stratified_split(fm, test_size=0.2, seed=0)
    # Both samples end up on one side or the other — important is that the call
    # doesn't crash with an IndexError on float-dtype indices.
    assert train.X.shape[0] + test.X.shape[0] == 2


def test_feature_matrix_roundtrip(tmp_path):
    fm = build_count_matrix(_toy_sessions())
    labels = {"blk_a": 0, "blk_b": 0, "blk_c": 1}
    fm = attach_labels(fm, labels)
    out = tmp_path / "fm.npz"
    fm.save(out)
    loaded = FeatureMatrix.load(out)
    np.testing.assert_array_equal(loaded.X, fm.X)
    np.testing.assert_array_equal(loaded.y, fm.y)
    assert list(loaded.session_ids) == list(fm.session_ids)
    np.testing.assert_array_equal(loaded.event_ids, fm.event_ids)
