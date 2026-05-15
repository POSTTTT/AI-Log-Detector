"""Command-line entry point for the ``detect`` tool.

Phases 1–3 implemented: ``info``, ``fetch``, ``parse``, ``features``,
``train``, ``score``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .data import DATASETS, fetch
from .paths import MODELS_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs


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

    state_path = Path(args.drain3_state) if args.drain3_state else MODELS_DIR / "drain3.bin"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Drain3's FilePersistence is incremental; clear stale state on retrain.
    if state_path.exists():
        state_path.unlink()

    print(f"Parsing {log_path} (limit={args.limit})...")
    df = parse_hdfs_log(
        log_path, limit=args.limit, show_progress=True, state_path=state_path
    )
    out = Path(args.output) if args.output else PROCESSED_DIR / "parsed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    df_to_save["block_ids"] = df_to_save["block_ids"].apply(lambda xs: " ".join(xs))
    df_to_save.to_csv(out, index=False)
    print(
        f"Parsed {len(df):,} lines, {df['event_id'].nunique()} templates -> {out} "
        f"(Drain3 state -> {state_path})"
    )
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

    # Persist the event-id vocab so the streaming scorer uses the same columns.
    import numpy as np

    vocab_path = MODELS_DIR / "vocab.npy"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vocab_path, fm.event_ids)

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


def _cmd_train(args: argparse.Namespace) -> int:
    from .features import FeatureMatrix
    from .models import build_detector

    ensure_dirs()
    train_path = Path(args.features) if args.features else PROCESSED_DIR / "features_train.npz"
    if not train_path.exists():
        print(f"Training features not found: {train_path}. Run `detect features` first.")
        return 2

    fm = FeatureMatrix.load(train_path)
    X_fit = fm.X

    # Autoencoder defaults to training on normal-only data (canonical AD setup).
    # Use --normal-only / --all-data to override.
    if args.normal_only is None:
        normal_only = args.model == "autoencoder"
    else:
        normal_only = args.normal_only

    if normal_only:
        if fm.y is None:
            print("--normal-only requires labels in the features file; aborting.")
            return 2
        X_fit = fm.X[fm.y == 0]
        print(
            f"Training {args.model} on {X_fit.shape[0]} normal sessions "
            f"(of {fm.X.shape[0]} total) x {fm.X.shape[1]} templates..."
        )
    else:
        print(
            f"Training {args.model} on {fm.X.shape[0]} sessions "
            f"x {fm.X.shape[1]} templates..."
        )

    detector = build_detector(args.model)
    detector.fit(X_fit)

    out_path = Path(args.output) if args.output else MODELS_DIR / f"{args.model}.bin"
    detector.save(out_path)
    print(f"Saved {args.model} model -> {out_path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from elasticsearch import Elasticsearch

    from .streaming import Scorer, StreamConfig

    ensure_dirs()
    model_path = Path(args.model)
    drain3_path = Path(args.drain3_state) if args.drain3_state else MODELS_DIR / "drain3.bin"
    vocab_path = Path(args.vocab) if args.vocab else MODELS_DIR / "vocab.npy"
    for p, label in [
        (model_path, "model"),
        (drain3_path, "Drain3 state"),
        (vocab_path, "vocab"),
    ]:
        if not p.exists():
            print(f"Missing {label} file: {p}")
            return 2

    es = Elasticsearch(args.es_url, request_timeout=30)
    config = StreamConfig(
        source_index=args.source_index,
        dest_index=args.dest_index,
        interval_seconds=args.interval,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    scorer = Scorer(
        es=es,
        model_path=model_path,
        drain3_state_path=drain3_path,
        vocab_path=vocab_path,
        config=config,
    )
    print(
        f"[scorer] reading from {config.source_index}, writing to {config.dest_index} "
        f"every {config.interval_seconds:.0f}s. Ctrl-C to stop."
    )
    try:
        scorer.run_forever(max_iters=args.max_iters)
    except KeyboardInterrupt:
        print("[scorer] interrupted; exiting cleanly.")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    import numpy as np

    from .evaluate import evaluate
    from .features import FeatureMatrix
    from .models import load_detector

    ensure_dirs()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run `detect train` first.")
        return 2

    features_path = Path(args.features) if args.features else PROCESSED_DIR / "features_test.npz"
    if not features_path.exists():
        print(f"Features not found: {features_path}.")
        return 2

    detector = load_detector(model_path)
    fm = FeatureMatrix.load(features_path)
    scores = detector.score(fm.X)

    scores_out = Path(args.scores_out) if args.scores_out else REPORTS_DIR / "scores.csv"
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_out, "w", encoding="utf-8") as f:
        f.write("session_id,score" + (",label\n" if fm.y is not None else "\n"))
        for i, sid in enumerate(fm.session_ids):
            if fm.y is not None:
                f.write(f"{sid},{scores[i]:.6f},{int(fm.y[i])}\n")
            else:
                f.write(f"{sid},{scores[i]:.6f}\n")
    print(f"Scored {len(scores)} sessions -> {scores_out}")

    if fm.y is not None:
        report = evaluate(np.asarray(fm.y), scores, threshold=args.threshold)
        metrics_out = (
            Path(args.metrics_out) if args.metrics_out else REPORTS_DIR / "metrics.json"
        )
        report.to_json(metrics_out)
        print(
            f"  threshold={report.threshold:.4f}  "
            f"precision={report.precision:.3f}  recall={report.recall:.3f}  "
            f"f1={report.f1:.3f}  roc_auc={report.roc_auc:.3f}  pr_auc={report.pr_auc:.3f}"
        )
        print(f"Saved metrics -> {metrics_out}")
    else:
        print("No labels in features file; skipping metrics.")
    return 0


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
    p_parse.add_argument(
        "--drain3-state",
        help="Where to persist Drain3 template state (default: models/drain3.bin).",
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

    p_train = sub.add_parser("train", help="Train an anomaly detector on a feature matrix.")
    p_train.add_argument(
        "--model", choices=("iforest", "autoencoder"), default="iforest",
        help="Detector type.",
    )
    p_train.add_argument("--features", help="Path to features_train.npz.")
    p_train.add_argument("--output", help="Where to save the trained model.")
    p_train.add_argument(
        "--normal-only", dest="normal_only", action="store_true", default=None,
        help="Train on labeled-normal sessions only (default for autoencoder).",
    )
    p_train.add_argument(
        "--all-data", dest="normal_only", action="store_false",
        help="Train on all sessions, ignoring labels (default for iforest).",
    )
    p_train.set_defaults(func=_cmd_train)

    p_score = sub.add_parser(
        "score",
        help="Score a feature matrix with a saved model; emit per-session scores + metrics.",
    )
    p_score.add_argument("--model", required=True, help="Path to a saved model file.")
    p_score.add_argument("--features", help="Path to features_test.npz.")
    p_score.add_argument("--scores-out", help="Where to write per-session scores CSV.")
    p_score.add_argument("--metrics-out", help="Where to write metrics JSON.")
    p_score.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed anomaly threshold. If omitted, the best-F1 threshold is chosen.",
    )
    p_score.set_defaults(func=_cmd_score)

    p_serve = sub.add_parser(
        "serve",
        help="Stream-score logs from an Elasticsearch index into a destination index.",
    )
    p_serve.add_argument("--model", required=True, help="Path to a trained model.")
    p_serve.add_argument(
        "--drain3-state", help="Drain3 template state (default: models/drain3.bin)."
    )
    p_serve.add_argument(
        "--vocab", help="Event vocab from training (default: models/vocab.npy)."
    )
    p_serve.add_argument(
        "--es-url", default="http://localhost:9200", help="Elasticsearch URL."
    )
    p_serve.add_argument(
        "--source-index", default="logs-raw-*",
        help="Index pattern to read raw logs from.",
    )
    p_serve.add_argument(
        "--dest-index", default="logs-scored",
        help="Index to write per-session anomaly scores into.",
    )
    p_serve.add_argument(
        "--interval", type=float, default=30.0,
        help="Seconds between polls.",
    )
    p_serve.add_argument("--batch-size", type=int, default=5000)
    p_serve.add_argument(
        "--threshold", type=float, default=None,
        help="Optional anomaly threshold (adds is_anomaly flag to scored docs).",
    )
    p_serve.add_argument(
        "--max-iters", type=int, default=None,
        help="Stop after N polls (default: run forever).",
    )
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
