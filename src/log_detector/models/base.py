"""Common interface for unsupervised anomaly detectors.

Scoring convention: ``score(X)`` returns a 1-D numpy array where **higher
values mean more anomalous**. Downstream code (threshold selection,
PR curves) relies on this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Detector(ABC):
    #: Stable string identifier persisted in saved artifacts and used by
    #: the registry to load the right concrete class.
    kind: str = "abstract"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Detector":
        """Fit on a 2-D feature matrix (treated as 'mostly normal')."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores; higher = more anomalous."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Detector":
        """Reload a previously-saved model from disk."""
