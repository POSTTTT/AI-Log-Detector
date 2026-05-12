"""Dataset acquisition helpers.

Phase 1 supports the HDFS_v1 LogHub dataset, the canonical benchmark for
log anomaly detection. Other LogHub datasets can be added by extending
``DATASETS`` below.

LogHub source: https://github.com/logpai/loghub
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from .paths import RAW_DIR


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    archive_name: str
    extracted_dir: str


# Zenodo mirror of LogHub. Stable URLs, no auth needed.
DATASETS: dict[str, Dataset] = {
    "HDFS_v1": Dataset(
        name="HDFS_v1",
        url="https://zenodo.org/records/8196385/files/HDFS_v1.zip",
        archive_name="HDFS_v1.zip",
        extracted_dir="HDFS_v1",
    ),
    "BGL": Dataset(
        name="BGL",
        url="https://zenodo.org/records/8196385/files/BGL.zip",
        archive_name="BGL.zip",
        extracted_dir="BGL",
    ),
}


def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, *, force: bool = False) -> Path:
    """Download and extract a dataset by name. Returns the extracted directory."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {list(DATASETS)}")

    ds = DATASETS[name]
    archive_path = RAW_DIR / ds.archive_name
    extracted_path = RAW_DIR / ds.extracted_dir

    if extracted_path.exists() and not force:
        return extracted_path

    if not archive_path.exists() or force:
        _download(ds.url, archive_path)

    extracted_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extracted_path)

    return extracted_path
