"""Smoke tests — verify the package imports and the CLI is wired up."""

from __future__ import annotations

import log_detector
from log_detector.cli import build_parser
from log_detector.data import DATASETS
from log_detector.paths import RAW_DIR, ROOT


def test_version_is_set():
    assert isinstance(log_detector.__version__, str)
    assert log_detector.__version__.count(".") >= 1


def test_paths_resolve_under_repo_root():
    assert ROOT.exists()
    assert RAW_DIR.is_relative_to(ROOT)


def test_known_datasets_have_required_fields():
    assert "HDFS_v1" in DATASETS
    for ds in DATASETS.values():
        assert ds.url.startswith("https://")
        assert ds.archive_name.endswith(".zip")
        assert ds.extracted_dir


def test_cli_parses_info_command():
    parser = build_parser()
    args = parser.parse_args(["info"])
    assert args.command == "info"


def test_cli_fetch_requires_known_dataset():
    parser = build_parser()
    args = parser.parse_args(["fetch", "HDFS_v1"])
    assert args.command == "fetch"
    assert args.dataset == "HDFS_v1"
    assert args.force is False
