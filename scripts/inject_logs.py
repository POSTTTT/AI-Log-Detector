"""Drip-feed synthetic HDFS log lines into ``data/live/app.log``.

Two modes share the same generator:

1. **Live demo (default)** — write to ``data/live/app.log`` at a slow rate
   so Filebeat ships lines to Elasticsearch incrementally and the scorer
   picks them up as fresh sessions.
2. **Training corpus** — pass ``--labels-out`` and ``--rate 0`` to dump a
   labeled training corpus alongside an ``anomaly_label.csv`` that the
   ``detect features`` step can consume.

Usage:
    # Live drip: 10 normal blocks at the default rate.
    python scripts/inject_logs.py

    # Live drip with anomalies and a faster rate.
    python scripts/inject_logs.py --blocks 12 --anomalies 3 --rate 5

    # Training corpus: 200 normal + 20 anomalous blocks, no rate limit,
    # plus a labels CSV for stratified training.
    python scripts/inject_logs.py \\
        --output data/training/corpus.log \\
        --labels-out data/training/anomaly_label.csv \\
        --blocks 200 --anomalies 20 --rate 0 --truncate --seed 0
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

NORMAL_TEMPLATES = [
    "INFO dfs.DataNode$DataXceiver: Receiving block {blk} src: /10.250.19.{ip}:54106 dest: /10.250.19.{ip}:50010",
    "INFO dfs.DataNode$PacketResponder: Received block {blk} of size 67108864 from /10.250.19.{ip}",
    "INFO dfs.DataNode$PacketResponder: PacketResponder {pid} for block {blk} terminating",
    "INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/file_{n} {blk}",
]

ANOMALY_TEMPLATES = [
    "WARN dfs.DataNode$DataXceiver: writeBlock {blk} received exception java.io.IOException",
    "ERROR dfs.DataNode$DataXceiver: PacketResponder {pid} for block {blk} Exception java.io.IOException",
    "WARN dfs.DataNode: DataNode is shutting down; abandoning block {blk}",
]


def make_block_id(rng: random.Random) -> str:
    """Return a numeric HDFS-style block id (``blk_<signed 64-bit int>``)."""
    return f"blk_{rng.randint(-(10**18), 10**18)}"


def render_line(template: str, *, blk: str, rng: random.Random) -> str:
    now = datetime.now()
    head = f"{now:%y%m%d %H%M%S} {rng.randint(100, 999)}"
    body = template.format(
        blk=blk,
        ip=rng.randint(2, 254),
        pid=rng.randint(0, 5),
        n=rng.randint(1, 1000),
    )
    return f"{head} {body}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", default="data/live/app.log", help="Log file Filebeat tails.")
    ap.add_argument("--blocks", type=int, default=10, help="How many normal blocks to emit.")
    ap.add_argument(
        "--anomalies", type=int, default=0, help="How many anomalous blocks to inject."
    )
    ap.add_argument(
        "--rate", type=float, default=5.0,
        help="Lines per second; pass 0 to write as fast as possible (corpus mode).",
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    ap.add_argument(
        "--truncate", action="store_true",
        help="Overwrite --output (default appends). Recommended for corpus generation.",
    )
    ap.add_argument(
        "--labels-out", default=None,
        help="If set, write a 'BlockId,Label' CSV alongside the log "
             "(format consumed by `detect features --labels`).",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    delay = 0.0 if args.rate <= 0 else 1.0 / args.rate
    mode = "w" if args.truncate else "a"

    labels: list[tuple[str, str]] = []
    total_lines = 0
    with open(out, mode, encoding="utf-8") as f:
        for _ in range(args.blocks):
            blk = make_block_id(rng)
            labels.append((blk, "Normal"))
            # Normal block: 3-5 lifecycle events from NORMAL_TEMPLATES.
            for _ in range(rng.randint(3, 5)):
                tpl = rng.choice(NORMAL_TEMPLATES)
                f.write(render_line(tpl, blk=blk, rng=rng) + "\n")
                f.flush()
                total_lines += 1
                if delay:
                    time.sleep(delay)

        for _ in range(args.anomalies):
            blk = make_block_id(rng)
            labels.append((blk, "Anomaly"))
            # Anomaly block: 1-2 normal events then 2-4 errors.
            for _ in range(rng.randint(1, 2)):
                f.write(render_line(rng.choice(NORMAL_TEMPLATES), blk=blk, rng=rng) + "\n")
                total_lines += 1
                if delay:
                    time.sleep(delay)
            for _ in range(rng.randint(2, 4)):
                f.write(render_line(rng.choice(ANOMALY_TEMPLATES), blk=blk, rng=rng) + "\n")
                f.flush()
                total_lines += 1
                if delay:
                    time.sleep(delay)

    print(f"Wrote {total_lines} lines ({args.blocks} normal + {args.anomalies} anomaly blocks) to {out}")

    if args.labels_out:
        lpath = Path(args.labels_out)
        lpath.parent.mkdir(parents=True, exist_ok=True)
        with open(lpath, "w", encoding="utf-8", newline="") as f:
            f.write("BlockId,Label\n")
            for blk, lbl in labels:
                f.write(f"{blk},{lbl}\n")
        print(f"Wrote {len(labels)} labels to {lpath}")


if __name__ == "__main__":
    main()
