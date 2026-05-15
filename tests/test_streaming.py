"""Tests for the streaming scorer (Elasticsearch is faked)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("drain3")
pytest.importorskip("sklearn")

from log_detector.features import FeatureMatrix  # noqa: E402
from log_detector.models import build_detector  # noqa: E402
from log_detector.streaming import Scorer, StreamConfig  # noqa: E402


class FakeES:
    """Minimal in-memory stand-in for elasticsearch.Elasticsearch."""

    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = list(docs or [])
        self.written: list[tuple[str, dict]] = []

    def search(self, *, index: str, body: dict, size: int) -> dict:
        gt = body["query"]["range"]["@timestamp"]["gt"]
        hits = [
            {"_source": d}
            for d in self.docs
            if d.get("@timestamp", "") > gt
        ]
        hits.sort(key=lambda h: h["_source"].get("@timestamp", ""))
        return {"hits": {"hits": hits[:size]}}

    def index(self, *, index: str, document: dict) -> dict:
        self.written.append((index, document))
        return {"result": "created"}


@pytest.fixture
def trained_artifacts(tmp_path: Path) -> dict:
    """Train a tiny model and persist Drain3 state + vocab to disk."""
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
    from drain3.template_miner_config import TemplateMinerConfig

    drain3_state = tmp_path / "drain3.bin"
    vocab_path = tmp_path / "vocab.npy"
    model_path = tmp_path / "iforest.bin"

    # Train Drain3 on a handful of HDFS-shaped messages to seed templates.
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(persistence_handler=FilePersistence(str(drain3_state)), config=cfg)
    seed_messages = [
        "Received block blk_111 of size 67108864 from /10.0.0.1",
        "Received block blk_222 of size 67108864 from /10.0.0.2",
        "Receiving block blk_333 src: /10.0.0.3 dest: /10.0.0.4",
        "writeBlock blk_444 received exception java.io.IOException",
    ]
    template_ids: list[int] = []
    for m in seed_messages:
        r = miner.add_log_message(m)
        template_ids.append(int(r["cluster_id"]))
    miner.save_state("seed")  # FilePersistence is interval-based by default.

    # Freeze a vocab and train an IForest on a tiny synthetic count matrix.
    vocab = np.array(sorted(set(template_ids)), dtype=np.int64)
    np.save(vocab_path, vocab)

    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.integers(0, 3, size=(50, len(vocab))),  # normal sessions
            rng.integers(10, 20, size=(5, len(vocab))),  # anomalies
        ]
    ).astype(np.int32)
    det = build_detector("iforest", n_estimators=50, random_state=0)
    det.fit(X)
    det.save(model_path)

    return {
        "drain3": drain3_state,
        "vocab": vocab_path,
        "model": model_path,
    }


def test_poll_once_with_no_data_returns_zero(trained_artifacts):
    es = FakeES(docs=[])
    scorer = Scorer(
        es=es,
        model_path=trained_artifacts["model"],
        drain3_state_path=trained_artifacts["drain3"],
        vocab_path=trained_artifacts["vocab"],
        config=StreamConfig(interval_seconds=0.0),
    )
    assert scorer.poll_once() == 0
    assert es.written == []


def test_poll_once_scores_sessions_and_writes_back(trained_artifacts):
    docs = [
        {"@timestamp": "2026-01-01T00:00:01Z",
         "message": "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
                    "Received block blk_555 of size 67108864 from /10.0.0.1"},
        {"@timestamp": "2026-01-01T00:00:02Z",
         "message": "081109 203616 148 INFO dfs.DataNode$PacketResponder: "
                    "Received block blk_555 of size 67108864 from /10.0.0.2"},
        {"@timestamp": "2026-01-01T00:00:03Z",
         "message": "081109 203617 148 INFO dfs.DataNode$DataXceiver: "
                    "Receiving block blk_666 src: /10.0.0.3 dest: /10.0.0.4"},
    ]
    es = FakeES(docs=docs)
    scorer = Scorer(
        es=es,
        model_path=trained_artifacts["model"],
        drain3_state_path=trained_artifacts["drain3"],
        vocab_path=trained_artifacts["vocab"],
        config=StreamConfig(
            source_index="logs-raw-*", dest_index="logs-scored",
            interval_seconds=0.0, threshold=0.0,
        ),
    )
    n = scorer.poll_once()
    assert n == 2  # two distinct block_ids -> two sessions

    indexes = {idx for idx, _ in es.written}
    assert indexes == {"logs-scored"}

    # Each written doc has the expected shape.
    for _, doc in es.written:
        assert set(doc) >= {"@timestamp", "session_id", "score", "model_kind",
                            "threshold", "is_anomaly"}
        assert doc["model_kind"] == "iforest"
        assert doc["session_id"] in {"blk_555", "blk_666"}

    # Checkpoint advanced to the latest timestamp.
    assert scorer.checkpoint == "2026-01-01T00:00:03Z"


def test_poll_once_advances_checkpoint_even_when_no_blocks(trained_artifacts):
    docs = [
        {"@timestamp": "2026-01-01T00:00:01Z", "message": "no block id here at all"},
    ]
    es = FakeES(docs=docs)
    scorer = Scorer(
        es=es,
        model_path=trained_artifacts["model"],
        drain3_state_path=trained_artifacts["drain3"],
        vocab_path=trained_artifacts["vocab"],
        config=StreamConfig(interval_seconds=0.0),
    )
    n = scorer.poll_once()
    assert n == 0
    assert scorer.checkpoint == "2026-01-01T00:00:01Z"


def test_second_poll_skips_already_seen_docs(trained_artifacts):
    docs = [
        {"@timestamp": "2026-01-01T00:00:01Z",
         "message": "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
                    "Received block blk_555 of size 67108864 from /10.0.0.1"},
    ]
    es = FakeES(docs=docs)
    scorer = Scorer(
        es=es,
        model_path=trained_artifacts["model"],
        drain3_state_path=trained_artifacts["drain3"],
        vocab_path=trained_artifacts["vocab"],
        config=StreamConfig(interval_seconds=0.0),
    )
    assert scorer.poll_once() == 1
    written_first = len(es.written)
    # Second poll: nothing new past the checkpoint.
    assert scorer.poll_once() == 0
    assert len(es.written) == written_first
