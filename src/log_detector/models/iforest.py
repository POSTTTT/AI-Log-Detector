"""Isolation Forest baseline detector (scikit-learn)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .base import Detector


class IForestDetector(Detector):
    kind = "iforest"

    def __init__(
        self,
        *,
        contamination: float | str = "auto",
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import IsolationForest

        self.params = {
            "contamination": contamination,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.model = IsolationForest(**self.params)

    def fit(self, X: np.ndarray) -> "IForestDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal. Negate so higher = more anomalous.
        return -self.model.decision_function(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"kind": self.kind, "params": self.params, "model": self.model}, f)

    @classmethod
    def load(cls, path: Path) -> "IForestDetector":
        with open(path, "rb") as f:
            blob = pickle.load(f)  # noqa: S301 - trusted local artifact
        if blob.get("kind") != cls.kind:
            raise ValueError(f"Expected kind={cls.kind!r}, got {blob.get('kind')!r}")
        det = cls.__new__(cls)
        det.params = blob["params"]
        det.model = blob["model"]
        return det
