"""Build feature matrices from sessionized event sequences.

Produces a dense numpy count matrix (sessions × event templates) suitable
for Isolation Forest and as input to an autoencoder.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureMatrix:
    X: np.ndarray  # (n_sessions, n_templates) int counts
    y: np.ndarray | None  # (n_sessions,) 0/1 labels, or None if unlabeled
    session_ids: np.ndarray  # (n_sessions,) — block_ids or window_starts
    event_ids: np.ndarray  # (n_templates,) column ordering

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=np.array([]) if self.y is None else self.y,
            session_ids=self.session_ids.astype(str),
            event_ids=self.event_ids,
            has_labels=np.array([self.y is not None]),
        )

    @classmethod
    def load(cls, path: Path) -> FeatureMatrix:
        data = np.load(path, allow_pickle=False)
        has_labels = bool(data["has_labels"][0])
        return cls(
            X=data["X"],
            y=data["y"] if has_labels else None,
            session_ids=data["session_ids"],
            event_ids=data["event_ids"],
        )


def build_count_matrix(
    sessions: pd.DataFrame,
    *,
    session_id_col: str = "block_id",
    event_vocab: np.ndarray | None = None,
) -> FeatureMatrix:
    """Convert a sessions DataFrame into a (n_sessions × n_templates) count matrix.

    If ``event_vocab`` is provided, columns follow that ordering and unknown
    events are dropped — use this to score new data with a vocabulary fixed
    at training time.
    """
    required = {session_id_col, "event_sequence"}
    if not required.issubset(sessions.columns):
        raise ValueError(f"sessions must contain {required}")

    if event_vocab is None:
        all_events = {e for seq in sessions["event_sequence"] for e in seq}
        event_vocab = np.array(sorted(all_events), dtype=np.int64)

    event_index = {int(e): i for i, e in enumerate(event_vocab)}
    n_sessions = len(sessions)
    n_events = len(event_vocab)

    X = np.zeros((n_sessions, n_events), dtype=np.int32)
    for row_idx, seq in enumerate(sessions["event_sequence"]):
        counts = Counter(seq)
        for event, count in counts.items():
            col = event_index.get(int(event))
            if col is not None:
                X[row_idx, col] = count

    return FeatureMatrix(
        X=X,
        y=None,
        session_ids=sessions[session_id_col].to_numpy(),
        event_ids=event_vocab,
    )


def load_hdfs_labels(label_csv: Path) -> dict[str, int]:
    """Load the HDFS_v1 ``anomaly_label.csv`` as a dict[block_id, 0|1].

    The file format is ``BlockId,Label`` where Label is ``Normal`` or ``Anomaly``.
    """
    df = pd.read_csv(label_csv)
    df.columns = [c.strip() for c in df.columns]
    block_col = "BlockId" if "BlockId" in df.columns else df.columns[0]
    label_col = "Label" if "Label" in df.columns else df.columns[1]
    mapping = dict(zip(df[block_col].astype(str), df[label_col].astype(str)))
    return {b: int(lbl.strip().lower() == "anomaly") for b, lbl in mapping.items()}


def attach_labels(fm: FeatureMatrix, labels: dict[str, int]) -> FeatureMatrix:
    """Attach binary labels to a FeatureMatrix using ``session_ids`` as the key.

    Sessions without a matching label are dropped.
    """
    keep_mask = np.array([str(s) in labels for s in fm.session_ids])
    if not keep_mask.any():
        raise ValueError("No session_ids matched the provided label mapping.")
    kept_ids = fm.session_ids[keep_mask]
    y = np.array([labels[str(s)] for s in kept_ids], dtype=np.int8)
    return FeatureMatrix(
        X=fm.X[keep_mask],
        y=y,
        session_ids=kept_ids,
        event_ids=fm.event_ids,
    )


def stratified_split(
    fm: FeatureMatrix,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[FeatureMatrix, FeatureMatrix]:
    """Stratified train/test split preserving the anomaly ratio.

    Requires ``fm.y`` to be set.
    """
    if fm.y is None:
        raise ValueError("stratified_split requires labels (fm.y is None)")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for klass in np.unique(fm.y):
        idx = np.where(fm.y == klass)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:cut].tolist())
        train_idx.extend(idx[cut:].tolist())

    train_idx_arr = np.array(sorted(train_idx), dtype=np.int64)
    test_idx_arr = np.array(sorted(test_idx), dtype=np.int64)

    def _subset(indices: np.ndarray) -> FeatureMatrix:
        return FeatureMatrix(
            X=fm.X[indices],
            y=fm.y[indices],
            session_ids=fm.session_ids[indices],
            event_ids=fm.event_ids,
        )

    return _subset(train_idx_arr), _subset(test_idx_arr)
