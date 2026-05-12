"""Tests for sessionization (block-based and time-window)."""

from __future__ import annotations

import pandas as pd

from log_detector.sessions import sessionize_by_block, sessionize_by_time_window


def test_sessionize_by_block_groups_events_per_block():
    parsed = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 3, 1],
            "block_ids": [["blk_a"], ["blk_a"], ["blk_b"], ["blk_b"], ["blk_a"]],
        }
    )
    sessions = sessionize_by_block(parsed)
    sessions = sessions.set_index("block_id")
    assert sorted(sessions.index) == ["blk_a", "blk_b"]
    assert sessions.loc["blk_a", "event_sequence"] == [1, 1, 1]
    assert sessions.loc["blk_b", "event_sequence"] == [2, 3]
    assert sessions.loc["blk_a", "length"] == 3
    assert sessions.loc["blk_b", "length"] == 2


def test_sessionize_by_block_explodes_multi_block_lines():
    parsed = pd.DataFrame(
        {
            "event_id": [1, 2],
            "block_ids": [["blk_a", "blk_b"], ["blk_a"]],
        }
    )
    sessions = sessionize_by_block(parsed).set_index("block_id")
    assert sessions.loc["blk_a", "event_sequence"] == [1, 2]
    assert sessions.loc["blk_b", "event_sequence"] == [1]


def test_sessionize_by_block_drops_empty_block_lists():
    parsed = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "block_ids": [["blk_a"], [], ["blk_a"]],
        }
    )
    sessions = sessionize_by_block(parsed).set_index("block_id")
    assert sessions.loc["blk_a", "event_sequence"] == [1, 3]


def test_sessionize_by_time_window():
    parsed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:00:40",
                    "2024-01-01 00:01:05",
                    "2024-01-01 00:01:50",
                ]
            ),
            "event_id": [1, 2, 3, 4],
        }
    )
    sessions = sessionize_by_time_window(parsed, window_seconds=60)
    assert len(sessions) == 2
    assert sessions.iloc[0]["event_sequence"] == [1, 2]
    assert sessions.iloc[1]["event_sequence"] == [3, 4]
