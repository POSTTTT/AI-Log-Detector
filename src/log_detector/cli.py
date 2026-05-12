"""Command-line entry point for the ``detect`` tool.

Phases 1–2 implemented: ``info``, ``fetch``, ``parse``, ``features``.
``train`` and ``score`` are placeholders for Phase 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .data import DATASETS, fetch
from .paths import PROCESSED_DIR, RAW_DIR, ensure_dirs


def _cmd_fetch(args: argparse.Namespace) -> int:
    ensure_dirs()
    path = fetch(args.dataset, force=args.force)
    print(f"Dataset {args.dataset} ready at: {path}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"log-detector v{__version__}")
    print("Available datasets:")
    for name, ds in DATASETS.items():
        print(f"  - {name}: {ds.url}")
    return 0


def _resolve_hdfs_log(dataset_dir: Path) -> Path:
    """Locate HDFS.log inside the extracted dataset dir."""
    candidates = list(dataset_dir.rglob("HDFS.log"))
    if not candidates:
        raise FileNotFoundError(f"HDFS.log not found under {dataset_dir}")
    return candidates[0]


def _resolve_hdfs_labels(dataset_dir: Path) -> Path | None:
    """Locate anomaly_label.csv inside the extracted dataset dir."""
    candidates = list(dataset_dir.rglob("anomaly_label.csv"))
    return candidates[0] if candidates else None


def _cmd_parse(args: argparse.Namespace) -> int:
    from .parse import parse_hdfs_log

    ensure_dirs()
    if args.input:
        log_path = Path(args.input)
    else:
        dataset_dir = RAW_DIR / DATASETS[args.dataset].extracted_dir
        log_path = _resolve_hdfs_log(dataset_dir)

    print(f"Parsing {log_path} (limit={args.limit})...")
    df = parse_hdfs_log(log_path, limit=args.limit, show_progress=True)
    out = Path(args.output) if args.output else PROCESSED_DIR / "parsed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    df_to_save["block_ids"] = df_to_save["block_ids"].apply(lambda xs: " ".join(xs))
    df_to_save.to_csv(out, index=False)
    print(f"Parsed {len(df):,} lines, {df['event_id'].nunique()} templates -> {out}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    from .features import (
        attach_labels,
        build_count_matrix,
        load_hdfs_labels,
        stratified_split,
    )
    from .sessions import sessionize_by_block

    import pandas as pd

    ensure_dirs()
    parsed_path = Path(args.parsed) if args.parsed else PROCESSED_DIR / "parsed.csv"
    if not parsed_path.exists():
        print(f"Parsed file not found: {parsed_path}. Run `detect parse` first.")
        return 2

    parsed = pd.read_csv(parsed_path)
    parsed["block_ids"] = (
        parsed["block_ids"].fillna("").astype(str).apply(lambda s: s.split() if s else [])
    )
    sessions = sessionize_by_block(parsed)
    print(f"Built {len(sessions):,} sessions from {len(parsed):,} parsed lines.")

    fm = build_count_matrix(sessions)
    print(f"Feature matrix: {fm.X.shape} over {len(fm.event_ids)} templates.")

    labels_path = Path(args.labels) if args.labels else None
    if labels_path is None and args.dataset:
        dataset_dir = RAW_DIR / DATASETS[args.dataset].extracted_dir
        labels_path = _resolve_hdfs_labels(dataset_dir)

    if labels_path and labels_path.exists():
        labels = load_hdfs_labels(labels_path)
        fm = attach_labels(fm, labels)
        anomaly_rate = float(fm.y.mean()) if fm.y is not None else 0.0
        print(
            f"Labels attached: {len(fm.session_ids):,} labeled sessions, "
            f"anomaly rate {anomaly_rate:.2%}"
        )
        train, test = stratified_split(fm, test_size=args.test_size, seed=args.seed)
        train.save(PROCESSED_DIR / "features_train.npz")
        test.save(PROCESSED_DIR / "features_test.npz")
        print(
            f"Saved features_train.npz ({train.X.shape}) and "
            f"features_test.npz ({test.X.shape}) to {PROCESSED_DIR}"
        )
    else:
        fm.save(PROCESSED_DIR / "features.npz")
        print(f"No labels supplied; saved unlabeled features.npz ({fm.X.shape})")

    return 0


def _not_implemented(name: str):
    def _run(_: argparse.Namespace) -> int:
        print(f"`detect {name}` is not implemented yet (planned for a later phase).")
        return 2

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect",
        description="AI-driven log anomaly detector.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show version and available datasets.")
    p_info.set_defaults(func=_cmd_info)

    p_fetch = sub.add_parser("fetch", help="Download a benchmark log dataset.")
    p_fetch.add_argument("dataset", choices=sorted(DATASETS), help="Dataset name.")
    p_fetch.add_argument("--force", action="store_true", help="Redownload if present.")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_parse = sub.add_parser(
        "parse",
        help="Parse a raw HDFS log with Drain3 into a CSV of templated events.",
    )
    p_parse.add_argument(
        "--dataset", choices=sorted(DATASETS), default="HDFS_v1",
        help="Dataset name (used to locate HDFS.log under data/raw/).",
    )
    p_parse.add_argument("--input", help="Override: path to a specific HDFS.log file.")
    p_parse.add_argument("--output", help="Override: output CSV path.")
    p_parse.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of input lines (handy for smoke runs).",
    )
    p_parse.set_defaults(func=_cmd_parse)

    p_feat = sub.add_parser(
        "features",
        help="Sessionize parsed events and build a feature matrix.",
    )
    p_feat.add_argument(
        "--dataset", choices=sorted(DATASETS), default="HDFS_v1",
        help="Dataset name (used to locate anomaly_label.csv).",
    )
    p_feat.add_argument("--parsed", help="Override: path to parsed.csv.")
    p_feat.add_argument("--labels", help="Override: path to anomaly_label.csv.")
    p_feat.add_argument("--test-size", type=float, default=0.2)
    p_feat.add_argument("--seed", type=int, default=42)
    p_feat.set_defaults(func=_cmd_features)

    for placeholder in ("train", "score"):
        p = sub.add_parser(placeholder, help=f"[planned] {placeholder} pipeline step.")
        p.set_defaults(func=_not_implemented(placeholder))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
