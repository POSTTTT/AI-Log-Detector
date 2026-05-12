"""Group parsed log events into sessions.

HDFS convention: one session per ``block_id``. Each session is the ordered
list of event template IDs (``event_id``) emitted for that block.

A generic time-window grouping is also provided for log sources that lack
a natural session key.
"""

from __future__ import annotations

import pandas as pd


def sessionize_by_block(parsed: pd.DataFrame) -> pd.DataFrame:
    """Explode ``block_ids`` and aggregate event sequences per block.

    Returns a DataFrame with columns: ``block_id``, ``event_sequence`` (list[int]),
    ``length`` (int).
    """
    if "block_ids" not in parsed.columns or "event_id" not in parsed.columns:
        raise ValueError("parsed must contain 'block_ids' and 'event_id' columns")

    exploded = parsed[["event_id", "block_ids"]].explode("block_ids")
    exploded = exploded.dropna(subset=["block_ids"])
    exploded = exploded.rename(columns={"block_ids": "block_id"})

    grouped = (
        exploded.groupby("block_id", sort=False)["event_id"]
        .apply(list)
        .reset_index(name="event_sequence")
    )
    grouped["length"] = grouped["event_sequence"].str.len()
    return grouped


def sessionize_by_time_window(
    parsed: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    window_seconds: int = 60,
) -> pd.DataFrame:
    """Group events into fixed time windows.

    Expects a datetime column. Returns: ``window_start``, ``event_sequence``, ``length``.
    """
    if timestamp_col not in parsed.columns:
        raise ValueError(f"parsed must contain '{timestamp_col}' column")

    ts = pd.to_datetime(parsed[timestamp_col])
    windows = ts.dt.floor(f"{window_seconds}s")
    df = parsed.assign(window_start=windows)

    grouped = (
        df.groupby("window_start", sort=True)["event_id"]
        .apply(list)
        .reset_index(name="event_sequence")
    )
    grouped["length"] = grouped["event_sequence"].str.len()
    return grouped
