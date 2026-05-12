"""Command-line entry point for the ``detect`` tool.

Phase 1 only implements the ``fetch`` subcommand. ``parse``, ``features``,
``train``, and ``score`` are placeholders to be filled in later phases.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .data import DATASETS, fetch
from .paths import ensure_dirs


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

    for placeholder in ("parse", "features", "train", "score"):
        p = sub.add_parser(placeholder, help=f"[planned] {placeholder} pipeline step.")
        p.set_defaults(func=_not_implemented(placeholder))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
