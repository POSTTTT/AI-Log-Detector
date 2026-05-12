# AI-Log-Detector

AI-Driven Log Anomaly Detector — feeds system or network logs into unsupervised ML models (Isolation Forest, autoencoders) that learn "normal" patterns and flag anomalies. See [PLAN.md](PLAN.md) for the full roadmap.

## Status

**Phase 1 — Foundation** (in progress). Project scaffold, CLI skeleton, and dataset downloader are in place. Parsing, modeling, and ELK ingestion come in later phases.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

For ML and parsing work in later phases, install extras:

```bash
pip install -e ".[parse,ml,torch,dev]"
# or:
pip install -e ".[all]"
```

## Quickstart

```bash
detect info                # show version and available datasets
detect fetch HDFS_v1       # download + extract the HDFS benchmark dataset
```

Downloaded data lands in `data/raw/HDFS_v1/`. The HDFS_v1 archive is ~1.5 GB compressed; only fetch it when you have the bandwidth.

## Project Layout

```
.
├── PLAN.md                 # Phased project plan
├── pyproject.toml          # Package + dependency definition
├── src/log_detector/
│   ├── __init__.py
│   ├── cli.py              # `detect` CLI entry point
│   ├── data.py             # Benchmark dataset downloader
│   └── paths.py            # Canonical project paths
├── tests/                  # Smoke tests
├── data/
│   ├── raw/                # Downloaded datasets (gitignored)
│   └── processed/          # Parsed/feature-engineered output (gitignored)
├── models/                 # Trained model artifacts (gitignored)
├── reports/                # Evaluation metrics + plots (gitignored)
└── notebooks/              # Exploratory analysis
```

## Running Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
