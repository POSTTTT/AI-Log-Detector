"""Batch-poll streaming scorer.

Reads new log documents from a source Elasticsearch index, parses each
``message`` field with the persisted Drain3 state, sessionizes by block_id,
builds count vectors against the frozen training vocab, scores them with a
saved detector, and writes one document per session into a destination
index.

Keeps a "checkpoint" of the highest ``@timestamp`` seen so each poll only
processes new data.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .features import build_count_matrix
from .models import load_detector
from .parse import _BLOCK_RE, parse_hdfs_line


class ESLike(Protocol):
    """Subset of the elasticsearch.Elasticsearch interface we depend on.

    Defining a Protocol lets us inject a fake client in tests without
    pulling in the real elasticsearch package at test time.
    """

    def search(self, *, index: str, body: dict, size: int) -> dict: ...
    def index(self, *, index: str, document: dict) -> dict: ...


@dataclass
class StreamConfig:
    source_index: str = "logs-raw-*"
    dest_index: str = "logs-scored"
    interval_seconds: float = 30.0
    batch_size: int = 5000
    threshold: float | None = None


class Scorer:
    def __init__(
        self,
        *,
        es: ESLike,
        model_path: Path,
        drain3_state_path: Path,
        vocab_path: Path,
        config: StreamConfig | None = None,
    ) -> None:
        from drain3 import TemplateMiner
        from drain3.file_persistence import FilePersistence
        from drain3.template_miner_config import TemplateMinerConfig

        self.es = es
        self.detector = load_detector(model_path)
        self.vocab = np.load(vocab_path)
        self.config = config or StreamConfig()

        miner_config = TemplateMinerConfig()
        miner_config.profiling_enabled = False
        miner_config.drain_sim_th = 0.4
        miner_config.drain_depth = 4
        self.miner = TemplateMiner(
            persistence_handler=FilePersistence(str(drain3_state_path)),
            config=miner_config,
        )

        self.checkpoint: str | None = None  # ISO timestamp string

    # ------------------------------------------------------------------
    # Pure helpers (no I/O) — easy to unit test.
    # ------------------------------------------------------------------

    def _docs_to_sessions(
        self, docs: list[dict]
    ) -> tuple[dict[str, list[int]], dict[str, str], str | None]:
        """Parse docs and group event ids by block_id.

        Returns (sessions, last_seen_ts_per_block, max_ts_overall).
        """
        sessions: dict[str, list[int]] = defaultdict(list)
        last_ts: dict[str, str] = {}
        max_ts: str | None = self.checkpoint

        for doc in docs:
            source = doc.get("_source", doc)
            message = source.get("message") or ""
            ts = source.get("@timestamp") or ""

            # Always advance the checkpoint, even if we can't parse this doc.
            if ts and (max_ts is None or ts > max_ts):
                max_ts = ts

            # Try a real HDFS line first; fall back to grabbing block IDs and
            # treating the whole message as the content.
            pl = parse_hdfs_line(message, line_id=0)
            if pl is None:
                block_ids = _BLOCK_RE.findall(message)
                if not block_ids:
                    continue
                content = message
            else:
                block_ids = pl.block_ids
                content = pl.content

            result = self.miner.add_log_message(content)
            event_id = int(result["cluster_id"])
            for blk in block_ids:
                sessions[blk].append(event_id)
                if ts > last_ts.get(blk, ""):
                    last_ts[blk] = ts

        return dict(sessions), last_ts, max_ts

    def _score_sessions(
        self, sessions: dict[str, list[int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a count matrix against the frozen vocab and score it.

        Returns (session_ids_array, anomaly_scores).
        """
        import pandas as pd

        if not sessions:
            return np.empty((0,), dtype=object), np.empty((0,), dtype=np.float32)

        df = pd.DataFrame(
            {
                "block_id": list(sessions.keys()),
                "event_sequence": list(sessions.values()),
            }
        )
        fm = build_count_matrix(df, event_vocab=self.vocab)
        scores = self.detector.score(fm.X)
        return fm.session_ids, scores

    # ------------------------------------------------------------------
    # I/O.
    # ------------------------------------------------------------------

    def _fetch_new_docs(self) -> list[dict]:
        gt = self.checkpoint or "1970-01-01T00:00:00Z"
        body = {
            "query": {"range": {"@timestamp": {"gt": gt}}},
            "sort": [{"@timestamp": {"order": "asc"}}],
        }
        resp = self.es.search(index=self.config.source_index, body=body, size=self.config.batch_size)
        return resp.get("hits", {}).get("hits", [])

    def _write_scores(
        self,
        session_ids: np.ndarray,
        scores: np.ndarray,
        last_ts: dict[str, str],
    ) -> int:
        n = 0
        for sid, score in zip(session_ids, scores):
            doc = {
                "@timestamp": last_ts.get(str(sid)) or self.checkpoint,
                "session_id": str(sid),
                "score": float(score),
                "model_kind": self.detector.kind,
            }
            if self.config.threshold is not None:
                doc["threshold"] = float(self.config.threshold)
                doc["is_anomaly"] = bool(score >= self.config.threshold)
            self.es.index(index=self.config.dest_index, document=doc)
            n += 1
        return n

    def poll_once(self) -> int:
        """One pass: fetch new docs, score, write back. Returns # sessions scored."""
        docs = self._fetch_new_docs()
        if not docs:
            return 0
        sessions, last_ts, max_ts = self._docs_to_sessions(docs)
        if not sessions:
            self.checkpoint = max_ts or self.checkpoint
            return 0
        session_ids, scores = self._score_sessions(sessions)
        n = self._write_scores(session_ids, scores, last_ts)
        self.checkpoint = max_ts or self.checkpoint
        return n

    def run_forever(self, *, max_iters: int | None = None) -> None:
        """Poll in a loop. ``max_iters`` is for tests; None = run forever."""
        i = 0
        while max_iters is None or i < max_iters:
            try:
                n = self.poll_once()
                ts = self.checkpoint or "(no data yet)"
                print(f"[scorer] poll: scored {n} sessions, checkpoint={ts}")
            except Exception as exc:  # noqa: BLE001 - we want the loop resilient
                print(f"[scorer] error during poll: {exc}")
            i += 1
            if max_iters is None or i < max_iters:
                time.sleep(self.config.interval_seconds)
