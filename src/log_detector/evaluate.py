"""Evaluation utilities for anomaly-detection scores.

All functions take ``y_true`` (0/1 labels, 1 = anomaly) and ``scores``
(higher = more anomalous) and produce metrics or threshold recommendations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalReport:
    n: int
    n_anomalies: int
    anomaly_rate: float
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def precision_recall_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    pred = scores >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Sweep thresholds at unique score values and return (threshold, F1) at max F1."""
    candidates = np.unique(scores)
    if len(candidates) == 0:
        return 0.0, 0.0
    best_t = float(candidates[0])
    best_f1 = 0.0
    for t in candidates:
        _, _, f1 = precision_recall_f1(y_true, scores, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the Mann–Whitney U statistic. Returns 0.5 if degenerate."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Average rank approach: count pairs where pos > neg, ties = 0.5.
    n_pairs = len(pos) * len(neg)
    # Vectorized: rank all scores, sum ranks of positives.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Handle ties: assign average rank to tied values.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    rank_sum_pos = ranks[y_true == 1].sum()
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / n_pairs)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return 0.0
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    # Step-function integration: sum precision over recall increments.
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum(precision * (recall - recall_prev)))


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None = None,
) -> EvalReport:
    """Compute a full evaluation report. If ``threshold`` is None, pick best F1."""
    if threshold is None:
        threshold, _ = best_f1_threshold(y_true, scores)
    precision, recall, f1 = precision_recall_f1(y_true, scores, threshold)
    return EvalReport(
        n=int(len(y_true)),
        n_anomalies=int((y_true == 1).sum()),
        anomaly_rate=float((y_true == 1).mean()),
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc(y_true, scores),
        pr_auc=pr_auc(y_true, scores),
    )
