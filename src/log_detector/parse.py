"""HDFS log parsing with Drain3 template extraction.

Reads a raw HDFS log file and produces a structured DataFrame with:
- Original fields (date, time, pid, level, component, content)
- Mined event template id and template string (via Drain3)
- Extracted block_id(s) for downstream sessionization

The HDFS_v1 line format from LogHub looks like:
    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999... ...
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Header regex for HDFS_v1 lines.
_HDFS_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>\w+)\s+"
    r"(?P<component>[\w.$]+):\s+"
    r"(?P<content>.*)$"
)

_BLOCK_RE = re.compile(r"blk_-?\d+")


@dataclass
class ParsedLine:
    line_id: int
    date: str
    time: str
    pid: str
    level: str
    component: str
    content: str
    block_ids: list[str]


def parse_hdfs_line(line: str, line_id: int) -> ParsedLine | None:
    """Parse one raw HDFS log line. Returns None for unparseable lines."""
    m = _HDFS_LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    content = m.group("content")
    return ParsedLine(
        line_id=line_id,
        date=m.group("date"),
        time=m.group("time"),
        pid=m.group("pid"),
        level=m.group("level"),
        component=m.group("component"),
        content=content,
        block_ids=_BLOCK_RE.findall(content),
    )


def iter_hdfs_lines(path: Path, limit: int | None = None) -> Iterator[ParsedLine]:
    """Stream-parse an HDFS log file. ``limit`` caps lines for quick smoke runs."""
    # utf-8-sig transparently strips a BOM if present (e.g. files written by
    # Windows PowerShell 5.1's Set-Content -Encoding utf8).
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for i, raw in enumerate(f):
            if limit is not None and i >= limit:
                break
            pl = parse_hdfs_line(raw, i)
            if pl is not None:
                yield pl


def _make_miner(state_path: Path | None = None):
    """Build a Drain3 TemplateMiner with sensible defaults. Imported lazily.

    If ``state_path`` is given, Drain3 will persist (and load) its template
    state to that file — allowing the scorer to reuse the exact same
    template IDs that were assigned at training time.
    """
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = 0.4
    config.drain_depth = 4

    if state_path is not None:
        from drain3.file_persistence import FilePersistence

        return TemplateMiner(persistence_handler=FilePersistence(str(state_path)), config=config)
    return TemplateMiner(config=config)


def mine_templates(
    lines: Iterable[ParsedLine],
    *,
    show_progress: bool = False,
    state_path: Path | None = None,
) -> pd.DataFrame:
    """Run Drain3 over parsed lines and return a DataFrame keyed by line_id.

    Output columns:
        line_id, date, time, pid, level, component, content,
        block_ids, event_id, template
    """
    miner = _make_miner(state_path=state_path)
    rows: list[dict] = []
    iterator = tqdm(lines, desc="drain3", unit="lines") if show_progress else lines
    for pl in iterator:
        result = miner.add_log_message(pl.content)
        rows.append(
            {
                "line_id": pl.line_id,
                "date": pl.date,
                "time": pl.time,
                "pid": pl.pid,
                "level": pl.level,
                "component": pl.component,
                "content": pl.content,
                "block_ids": pl.block_ids,
                "event_id": int(result["cluster_id"]),
                "template": result["template_mined"],
            }
        )
    # Drain3's FilePersistence flushes on a snapshot interval (default 60s);
    # force a final save so the scorer can load the same template clusters.
    if state_path is not None:
        miner.save_state("training_complete")
    return pd.DataFrame(rows)


def parse_hdfs_log(
    path: Path,
    *,
    limit: int | None = None,
    show_progress: bool = False,
    state_path: Path | None = None,
) -> pd.DataFrame:
    """End-to-end: read HDFS log → parse lines → mine templates → DataFrame."""
    return mine_templates(
        iter_hdfs_lines(path, limit=limit),
        show_progress=show_progress,
        state_path=state_path,
    )
