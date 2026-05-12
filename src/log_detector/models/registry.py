"""Detector registry: build by name, load by stored ``kind``."""

from __future__ import annotations

import pickle
from pathlib import Path

from .base import Detector


KNOWN: tuple[str, ...] = ("iforest", "autoencoder")


def build_detector(name: str, **kwargs) -> Detector:
    if name == "iforest":
        from .iforest import IForestDetector

        return IForestDetector(**kwargs)
    if name == "autoencoder":
        from .autoencoder import AEConfig, AutoencoderDetector

        return AutoencoderDetector(config=AEConfig(**kwargs) if kwargs else None)
    raise ValueError(f"Unknown detector {name!r}. Known: {KNOWN}")


def _peek_kind(path: Path) -> str:
    """Return the model 'kind' string from a saved artifact without
    fully deserializing the model object."""
    # Try torch checkpoint first (autoencoder); fall back to pickle (iforest).
    try:
        import torch

        blob = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(blob, dict) and "kind" in blob:
            return blob["kind"]
    except Exception:
        pass
    with open(path, "rb") as f:
        blob = pickle.load(f)  # noqa: S301 - trusted local artifact
    return blob["kind"]


def load_detector(path: Path) -> Detector:
    kind = _peek_kind(path)
    if kind == "iforest":
        from .iforest import IForestDetector

        return IForestDetector.load(path)
    if kind == "autoencoder":
        from .autoencoder import AutoencoderDetector

        return AutoencoderDetector.load(path)
    raise ValueError(f"Unknown model kind {kind!r} in {path}")
