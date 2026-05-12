"""Dense autoencoder anomaly detector (PyTorch).

Trains a small encoder-decoder on (assumed-mostly-normal) log feature
vectors and uses per-sample reconstruction error as the anomaly score.
Inputs are log1p-normalized so high event counts don't dominate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import Detector


@dataclass
class AEConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [16, 8])
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42


class AutoencoderDetector(Detector):
    kind = "autoencoder"

    def __init__(self, config: AEConfig | None = None) -> None:
        self.config = config or AEConfig()
        self._input_dim: int | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._model = None  # set on fit/load

    @staticmethod
    def _build_model(input_dim: int, hidden_dims: list[int]):
        import torch.nn as nn

        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]
        for in_d, out_d in zip(reversed(dims[1:]), reversed(dims[:-1])):
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]
        # Strip the trailing ReLU on the output layer so it can reconstruct
        # the (positive but unbounded) log1p-normalized inputs.
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        Xn = np.log1p(X.astype(np.float32))
        if self._mean is None or self._std is None:
            self._mean = Xn.mean(axis=0)
            self._std = Xn.std(axis=0) + 1e-6
        return (Xn - self._mean) / self._std

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._input_dim = X.shape[1]
        Xn = self._normalize(X)
        self._model = self._build_model(self._input_dim, self.config.hidden_dims)

        tensor_x = torch.from_numpy(Xn).float()
        loader = DataLoader(
            TensorDataset(tensor_x),
            batch_size=min(self.config.batch_size, len(tensor_x)),
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        self._model.train()
        for _ in range(self.config.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self._model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()

        self._model.eval()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("AutoencoderDetector.score called before fit/load.")
        Xn = self._normalize(X)
        with torch.no_grad():
            t = torch.from_numpy(Xn).float()
            recon = self._model(t)
            errs = ((recon - t) ** 2).mean(dim=1).cpu().numpy()
        return errs

    def save(self, path: Path) -> None:
        import torch

        if self._model is None:
            raise RuntimeError("Cannot save an untrained AutoencoderDetector.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": self.kind,
                "config": self.config.__dict__,
                "input_dim": self._input_dim,
                "mean": self._mean,
                "std": self._std,
                "state_dict": self._model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "AutoencoderDetector":
        import torch

        blob = torch.load(path, weights_only=False)
        if blob.get("kind") != cls.kind:
            raise ValueError(f"Expected kind={cls.kind!r}, got {blob.get('kind')!r}")
        det = cls(config=AEConfig(**blob["config"]))
        det._input_dim = blob["input_dim"]
        det._mean = blob["mean"]
        det._std = blob["std"]
        det._model = cls._build_model(det._input_dim, det.config.hidden_dims)
        det._model.load_state_dict(blob["state_dict"])
        det._model.eval()
        return det
