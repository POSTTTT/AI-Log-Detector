"""Tests for HDFS log parsing and Drain3 template mining."""

from __future__ import annotations

import pytest

pytest.importorskip("drain3")

from log_detector.parse import (  # noqa: E402
    iter_hdfs_lines,
    mine_templates,
    parse_hdfs_line,
    parse_hdfs_log,
)

from .fixtures import SYNTHETIC_HDFS_LOG  # noqa: E402


def test_parse_line_extracts_fields():
    line = (
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
        "Received block blk_111 of size 67108864 from /10.250.19.102"
    )
    pl = parse_hdfs_line(line, line_id=0)
    assert pl is not None
    assert pl.date == "081109"
    assert pl.time == "203615"
    assert pl.pid == "148"
    assert pl.level == "INFO"
    assert pl.component == "dfs.DataNode$PacketResponder"
    assert "Received block" in pl.content
    assert pl.block_ids == ["blk_111"]


def test_parse_line_returns_none_for_garbage():
    assert parse_hdfs_line("not a real log line", 0) is None


def test_iter_hdfs_lines(tmp_path):
    log = tmp_path / "mini.log"
    log.write_text(SYNTHETIC_HDFS_LOG, encoding="utf-8")
    parsed = list(iter_hdfs_lines(log))
    assert len(parsed) == 11
    assert all(p.block_ids for p in parsed)


def test_mine_templates_produces_event_ids(tmp_path):
    log = tmp_path / "mini.log"
    log.write_text(SYNTHETIC_HDFS_LOG, encoding="utf-8")
    df = mine_templates(iter_hdfs_lines(log))
    assert len(df) == 11
    assert {"event_id", "template", "block_ids"} <= set(df.columns)
    # Drain3 should cluster the repeated "Received block ... of size ... from ..."
    # lines into the same template.
    received_block_events = df[df["content"].str.startswith("Received block")]
    assert received_block_events["event_id"].nunique() == 1


def test_parse_hdfs_log_end_to_end(tmp_path):
    log = tmp_path / "mini.log"
    log.write_text(SYNTHETIC_HDFS_LOG, encoding="utf-8")
    df = parse_hdfs_log(log)
    assert len(df) == 11
    # We have at least a handful of distinct templates from the synthetic log.
    assert df["event_id"].nunique() >= 3
    # Every line has a block_id.
    assert df["block_ids"].apply(len).min() >= 1


def test_parse_hdfs_log_limit(tmp_path):
    log = tmp_path / "mini.log"
    log.write_text(SYNTHETIC_HDFS_LOG, encoding="utf-8")
    df = parse_hdfs_log(log, limit=3)
    assert len(df) == 3


def test_parse_hdfs_log_handles_utf8_bom(tmp_path):
    """Regression: PowerShell's Set-Content -Encoding utf8 prepends a BOM.
    The parser must not drop the first line because of it."""
    log = tmp_path / "mini_bom.log"
    log.write_bytes(b"\xef\xbb\xbf" + SYNTHETIC_HDFS_LOG.encode("utf-8"))
    df = parse_hdfs_log(log)
    assert len(df) == 11
