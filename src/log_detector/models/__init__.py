"""Anomaly detection models.

Public API: ``build_detector(name)`` returns a fresh detector by name;
``load_detector(path)`` reloads a saved one based on its stored ``kind``.
"""

from __future__ import annotations

from .base import Detector
from .registry import build_detector, load_detector

__all__ = ["Detector", "build_detector", "load_detector"]
