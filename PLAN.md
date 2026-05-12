# AI-Driven Log Anomaly Detector — Plan

Feed system or network logs into an ML model that learns "normal" patterns and flags anomalies. Unsupervised learning (Isolation Forest, autoencoders, optional LSTM) over logs collected by the ELK stack (or Splunk Free).

## Goals

- Detect anomalous behavior in log streams without labeled training data.
- Reproducible pipeline: raw logs → parsed templates → features → model → scored events → dashboard.
- Demonstrable end-to-end on a public benchmark dataset, then extensible to live logs.

## Non-Goals (for v1)

- Real-time sub-second alerting (batch / near-real-time is fine).
- Multi-tenant deployment, auth, RBAC.
- Supervised classification of attack types — only "normal vs anomalous".

## Scope Decision

Two viable starting points:

- **System logs** (Linux syslog/auth.log, Windows Event Log) — easier, well-studied benchmark datasets exist.
- **Network logs** (Zeek, Suricata, firewall) — richer features but harder parsing and labeling.

**Decision:** Start with **HDFS_v1** or **BGL** from [LogHub](https://github.com/logpai/loghub) for development (labeled, reproducible). Layer in live ELK ingestion in Phase 4.

---

## Phase 1 — Foundation (1–2 days) ✅ complete

1. Python project scaffold:
   - `pyproject.toml`
   - `src/log_detector/`
   - `tests/`
   - `data/raw/`, `data/processed/`
   - `notebooks/` for exploration
2. Pin dependencies:
   - `scikit-learn`, `pandas`, `numpy`
   - `tensorflow` or `torch` (autoencoder)
   - `drain3` (log template parsing)
   - `matplotlib`, `seaborn`
3. Download one LogHub dataset (HDFS_v1: ~575k log lines, labeled anomaly blocks).
4. Set up `.gitignore` for `data/`, `models/`, `__pycache__/`, virtualenv.

**Deliverable:** running `pip install -e .` works; sample dataset loaded in a notebook.

---

## Phase 2 — Parsing & Feature Engineering (2–3 days) ✅ complete

5. **Log template extraction** with Drain3 — converts raw lines into event IDs.
   - Example: `Received block blk_123 from 10.0.0.1` → template `E5`.
6. **Session / window grouping**:
   - HDFS: group by `block_id` (canonical approach for this dataset).
   - Syslog-style: fixed time windows (e.g. 60s) or sliding windows.
7. **Feature vectors per session**:
   - Event count vectors (raw counts or TF-IDF).
   - Numeric features: event rate, unique-event count, time-of-day, session duration.
8. Train/test split that preserves anomaly ratio; persist processed features to `data/processed/`.

**Deliverable:** `make features` produces a feature matrix + labels file.

---

## Phase 3 — Models (3–5 days) ✅ complete (LSTM stretch goal deferred)

9. **Baseline: Isolation Forest** on count-vector features.
   - Fast, no GPU, good reference ROC/PR numbers.
10. **Autoencoder** (Keras or PyTorch):
    - Dense AE on same features.
    - Anomaly score = reconstruction error; threshold by percentile on validation set.
11. *(Stretch)* **DeepLog-style LSTM** on event sequences:
    - Predicts next event; flags low-probability transitions.
    - Paper-grade approach, sequence-aware.
12. **Evaluation:**
    - Precision / Recall / F1 on labeled split.
    - PR curve (more informative than ROC for rare-class problems).
    - **Do not trust accuracy** — anomalies are <3% of data.
13. Save best model + threshold to `models/`.

**Deliverable:** `detect train` CLI command; metrics report in `reports/`.

---

## Phase 4 — Pipeline & Ingestion (2–4 days)

14. CLI:
    - `detect train --input logs.csv --model isoforest`
    - `detect score --input new.log --model models/isoforest.pkl`
15. **ELK vs Splunk Free decision:**
    - **ELK** (Elasticsearch + Logstash/Filebeat + Kibana) — free, flexible, runs in Docker, no daily ingest cap. **Chosen.**
    - Splunk Free — capped at 500 MB/day, no auth, harder to script against.
16. **Docker Compose stack:**
    - Elasticsearch + Kibana + Filebeat tailing a local log dir.
    - Scoring loop reads from Elasticsearch (or a Kafka topic), writes anomaly scores back to a separate index.
17. Scheduled scoring job (cron or simple Python loop) for near-real-time detection.

**Deliverable:** `docker compose up` → logs flow in → anomalies appear in a separate ES index within minutes.

---

## Phase 5 — Demo & Polish (1–2 days)

18. **Kibana dashboard:**
    - Anomaly score timeline.
    - Top anomalous templates / hosts.
    - Drill-down panel to raw log lines for a flagged window.
19. **Synthetic anomaly injection** script:
    - Inject port-scan bursts, failed-auth floods, disk errors.
    - Demonstrates live detection without waiting for real incidents.
20. README with:
    - Architecture diagram.
    - Reproduction steps (dataset download → train → score → dashboard).
    - Metrics table comparing Isolation Forest vs Autoencoder vs (optional) LSTM.

**Deliverable:** A reviewer can clone, run `docker compose up`, and see anomalies in Kibana within 15 minutes.

---

## Architecture

```
                            ┌──────────────────┐
   raw logs ──► Filebeat ──►│  Elasticsearch   │◄── anomaly scores
                            └────────┬─────────┘         ▲
                                     │                   │
                                     ▼                   │
                              ┌───────────┐       ┌─────────────┐
                              │  Kibana   │       │  Scorer     │
                              │ dashboard │       │ (Python)    │
                              └───────────┘       └──────┬──────┘
                                                         │
                                                  ┌──────▼──────┐
                                                  │  Model      │
                                                  │ (IF / AE)   │
                                                  └─────────────┘
            offline training:
              raw logs ─► Drain3 parser ─► feature builder ─► trainer ─► model artifact
```

---

## Key Risks

- **Concept drift** — model trained on weekday traffic flags weekend as anomalous. Mitigation: rolling-window retraining or per-segment models.
- **Parser failures** on novel log formats — Drain3 needs tuning per source.
- **Label scarcity** outside benchmark datasets — on your own logs, evaluation becomes qualitative (score distribution sanity checks, manual spot-checks) rather than F1.
- **False-positive fatigue** — without threshold tuning a noisy model is worse than none. Calibrate thresholds against a held-out clean window.

---

## Tech Stack Summary

| Layer        | Choice                                     |
|--------------|--------------------------------------------|
| Language     | Python 3.11+                               |
| Parsing      | Drain3                                     |
| ML           | scikit-learn (Isolation Forest), PyTorch or TensorFlow (Autoencoder) |
| Ingestion    | Filebeat → Elasticsearch                   |
| Visualization| Kibana                                     |
| Orchestration| Docker Compose                             |
| Dataset      | LogHub HDFS_v1 (dev), live logs (later)    |

---

## Milestones

- **M1:** Phase 1–2 complete — features built from HDFS dataset.
- **M2:** Phase 3 complete — Isolation Forest + Autoencoder trained, metrics reported.
- **M3:** Phase 4 complete — ELK stack running, scoring loop live.
- **M4:** Phase 5 complete — Kibana dashboard + injection demo + README.
