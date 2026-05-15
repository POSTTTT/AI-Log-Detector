# AI-Log-Detector

AI-Driven Log Anomaly Detector — feeds system or network logs into unsupervised ML models
(Isolation Forest, autoencoders) that learn "normal" patterns and flag anomalies. Ships
with a Dockerized ELK ingestion stack and a streaming scorer that writes per-session
anomaly scores back to Elasticsearch for Kibana visualization.

See [PLAN.md](PLAN.md) for the phased build plan.

## Status

All five phases complete:

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Foundation — scaffold, CLI skeleton, dataset downloader | ✅ |
| 2 | Parsing & features — Drain3 templates, sessionize, count matrix | ✅ |
| 3 | Models — Isolation Forest + PyTorch autoencoder, PR/ROC/F1 eval | ✅ |
| 4 | Pipeline — ELK stack + Drain3-state-aware streaming scorer | ✅ |
| 5 | Demo & polish — one-shot demo orchestrator + Kibana bootstrap | ✅ |

## Architecture

```
                    OFFLINE TRAINING                              LIVE SCORING
                    ─────────────────                             ────────────

  raw HDFS log ─┐                                       data/live/*.log
                ▼                                              │
            Drain3 parser  ────┐                               ▼
            (parse.py)         │                          Filebeat
                ▼              │                               │
         parsed.csv            │                               ▼
                ▼              │                  ┌────────────────────────┐
          sessionize           │                  │   Elasticsearch        │
        (sessions.py)          │                  │   logs-raw-* index     │
                ▼              │                  └───────────┬────────────┘
        count matrix           │                              │
        (features.py)          │                              ▼
                ▼              ▼                       streaming.Scorer
         train detector ──► models/iforest.bin ──────────────┤
        (Isolation Forest                models/drain3.bin   │
         or Autoencoder)                 models/vocab.npy    │
                                                             ▼
                                                  ┌────────────────────────┐
                                                  │   Elasticsearch        │
                                                  │   logs-scored index    │
                                                  └───────────┬────────────┘
                                                              │
                                                              ▼
                                                          Kibana
                                                  (Discover / Lens / dashboards)
```

The same Drain3 template state is used at training time and scoring time so an event
clustered as "template #7 = Receiving block X" during training maps to the same template
ID in the streaming pipeline. The `event_vocab` from training freezes the feature-matrix
column ordering so new sessions get scored against the exact same input space the model
was trained on.

## Try the demo in 5 minutes

Prerequisites: Python 3.10+, Docker Desktop running, ~3 GB free RAM.

```powershell
# Windows PowerShell — Linux/macOS substitute the venv activation line.
git clone <repo> && cd AI-Log-Detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"

python scripts/run_demo.py
```

`run_demo.py` is fully automated. It:

1. Confirms the Docker daemon is reachable.
2. Generates a 200-block synthetic training corpus (+ labels).
3. Runs `detect parse / features / train` to produce `models/iforest.bin`,
   `models/drain3.bin`, `models/vocab.npy`.
4. Brings up Elasticsearch + Kibana + Filebeat with `docker compose`.
5. Waits for Kibana and creates the `logs-raw` and `logs-scored` data views.
6. Drips synthetic live logs into `data/live/app.log` (Filebeat ships them).
7. Runs the streaming scorer for a few polls (writes to `logs-scored`).
8. Prints Kibana URLs and suggested KQL queries.

**Open Kibana** at <http://localhost:5601/app/discover>, select the `logs-scored` data
view, and run `is_anomaly : true`. You should see the anomalous blocks the injector
generated.

**Tear down:**

```powershell
docker compose -f docker/docker-compose.yml down
```

The demo is idempotent — re-running it skips training when `models/` already has the
artifacts. Pass `--force-retrain` to rebuild.

## Manual workflow (without the demo runner)

Useful when you want to inspect intermediate artifacts or run against the real HDFS
dataset.

```powershell
# 1. (Optional) Real HDFS_v1 dataset — ~1.5 GB download.
detect fetch HDFS_v1
detect parse --dataset HDFS_v1
detect features --dataset HDFS_v1
detect train --model iforest
detect score --model models/iforest.bin

# 2. Start the ELK stack and bootstrap Kibana.
docker compose -f docker/docker-compose.yml up -d
python scripts/bootstrap_kibana.py

# 3. Feed some logs and score them.
python scripts/inject_logs.py --blocks 20 --anomalies 3 --rate 5
detect serve --model models/iforest.bin --threshold 0.0 --interval 10 --max-iters 3
```

## CLI reference

```bash
detect info                                  # version + available datasets
detect fetch HDFS_v1                         # download a benchmark dataset
detect parse  --dataset HDFS_v1              # Drain3 -> data/processed/parsed.csv
                                             #         + models/drain3.bin
detect features --dataset HDFS_v1            # sessionize + count vectors
                                             # -> data/processed/features_{train,test}.npz
                                             # -> models/vocab.npy
detect train --model iforest|autoencoder     # train (autoencoder: --normal-only by default)
                                             # -> models/<model>.bin
detect score --model models/iforest.bin      # offline scoring -> reports/scores.csv
                                             #                 + reports/metrics.json
detect serve --model models/iforest.bin      # live scoring against Elasticsearch
                                             # source: logs-raw-*  dest: logs-scored
```

## Extras (install only what you need)

| Extra      | Pulls in                | Required for                                     |
|------------|-------------------------|--------------------------------------------------|
| `[parse]`  | drain3                  | `detect parse`                                   |
| `[ml]`     | scikit-learn, matplotlib, seaborn | `detect train --model iforest`         |
| `[torch]`  | PyTorch                 | `detect train --model autoencoder`               |
| `[stream]` | elasticsearch (8.x)     | `detect serve`, `scripts/bootstrap_kibana.py`    |
| `[dev]`    | pytest, ruff, coverage  | tests + linting                                  |
| `[all]`    | everything above        | running the full demo                            |

## Project layout

```
.
├── PLAN.md                 # Phased project plan
├── pyproject.toml          # Package + dependency definition (extras-driven)
├── src/log_detector/
│   ├── __init__.py
│   ├── cli.py              # `detect` CLI entry point (info/fetch/parse/features/train/score/serve)
│   ├── data.py             # Benchmark dataset downloader (HDFS_v1, BGL)
│   ├── paths.py            # Canonical project paths
│   ├── parse.py            # Drain3-based HDFS log parser (with state persistence)
│   ├── sessions.py         # Group events by block_id / time window
│   ├── features.py         # FeatureMatrix, count vectors, label join, stratified split
│   ├── evaluate.py         # Precision/recall/F1, ROC/PR-AUC, best-F1 threshold sweep
│   ├── streaming.py        # ES batch-poll scoring loop (`detect serve`)
│   └── models/             # Detector implementations
│       ├── base.py         # Detector abstract base (higher score = more anomalous)
│       ├── iforest.py      # IsolationForestDetector (sklearn)
│       ├── autoencoder.py  # AutoencoderDetector (PyTorch dense AE)
│       └── registry.py     # build_detector(name) / load_detector(path)
├── docker/
│   ├── docker-compose.yml  # Elasticsearch + Kibana + Filebeat (8.x)
│   ├── filebeat/
│   │   └── filebeat.yml    # Tails data/live/*.log, ships to logs-raw-*
│   └── .env.example        # Copy to .env to pin a specific Elastic version
├── scripts/
│   ├── inject_logs.py      # Synthetic HDFS-style log generator (live drip + labeled corpus)
│   ├── bootstrap_kibana.py # Creates Kibana data views idempotently
│   └── run_demo.py         # One-shot end-to-end demo orchestrator
├── tests/                  # 50 tests covering parse, sessions, features, models,
│                           # evaluate, streaming, Kibana bootstrap, end-to-end CLI
├── data/
│   ├── raw/                # Downloaded benchmark datasets (gitignored)
│   ├── processed/          # Parsed CSV + feature matrices (gitignored)
│   ├── training/           # Synthetic training corpus from run_demo.py (gitignored)
│   └── live/               # Filebeat tail target (gitignored)
├── models/                 # Trained model + Drain3 state + vocab (gitignored)
├── reports/                # Evaluation metrics + scores CSV (gitignored)
└── notebooks/              # Exploratory analysis (empty by default)
```

## Running tests

```bash
pytest -q
```

50 tests covering: log parsing (Drain3, BOM tolerance, HDFS regex), sessionization
(block_id grouping, time windows), feature engineering (count matrix, vocab freeze,
label join, stratified split), models (IForest + autoencoder fit/score/save/load,
detector registry), evaluation (precision/recall/F1, ROC/PR-AUC, best-threshold sweep),
streaming scorer (FakeES, checkpoint advance, dedupe by `@timestamp`), Kibana bootstrap
(FakeSession, idempotent create), and CLI pipelines (parse → features → train → score).

Streaming and model tests use `pytest.importorskip` so the suite degrades gracefully
when optional extras are missing.

## License

MIT — see [LICENSE](LICENSE).
