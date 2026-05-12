"""Canonical filesystem paths used across the project."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ROOT / "reports"


def ensure_dirs() -> None:
    """Create the standard project data directories if they do not exist."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
