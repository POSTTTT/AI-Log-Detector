"""Drip-feed synthetic HDFS log lines into ``data/live/app.log``.

Filebeat tails ``data/live/*.log`` and ships each line to Elasticsearch.
This script lets you watch the scorer pick up new sessions in Kibana
without needing the real 1.5 GB HDFS dataset.

Usage:
    python scripts/inject_logs.py                  # 10 normal blocks
    python scripts/inject_logs.py --anomalies 3    # inject some anomalies
    python scripts/inject_logs.py --rate 2         # 2 lines/sec
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/live/app.log", help="Log file Filebeat tails.")
    ap.add_argument("--blocks", type=int, default=10, help="How many normal blocks to emit.")
    ap.add_argument("--anomalies", type=int, default=0, help="How many anomalous blocks to inject.")
    ap.add_argument("--rate", type=float, default=5.0, help="Lines per second.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    delay = 1.0 / max(args.rate, 0.01)

    total_lines = 0
    with open(out, "a", encoding="utf-8") as f:
        for _ in range(args.blocks):
            blk = make_block_id(rng)
            # A "normal" block: 3-5 lifecycle events from NORMAL_TEMPLATES.
            for _ in range(rng.randint(3, 5)):
                tpl = rng.choice(NORMAL_TEMPLATES)
                f.write(render_line(tpl, blk=blk, rng=rng) + "\n")
                f.flush()
                total_lines += 1
                time.sleep(delay)

        for _ in range(args.anomalies):
            blk = make_block_id(rng)
            # An anomalous block: 1-2 normal events then an error.
            for _ in range(rng.randint(1, 2)):
                f.write(render_line(rng.choice(NORMAL_TEMPLATES), blk=blk, rng=rng) + "\n")
                total_lines += 1
                time.sleep(delay)
            for _ in range(rng.randint(2, 4)):
                f.write(render_line(rng.choice(ANOMALY_TEMPLATES), blk=blk, rng=rng) + "\n")
                f.flush()
                total_lines += 1
                time.sleep(delay)

    print(f"Wrote {total_lines} lines to {out}")


if __name__ == "__main__":
    main()
