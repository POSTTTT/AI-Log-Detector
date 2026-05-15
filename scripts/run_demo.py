"""One-shot end-to-end demo for AI-Log-Detector.

Runs the whole pipeline against a synthetic corpus so you can see the
ELK + scorer + Kibana stack working without downloading the 1.5 GB HDFS
benchmark.

Steps performed (each is skipped if its output already exists):

1. Pre-flight: Docker daemon reachable?
2. Generate a labeled synthetic training corpus -> ``data/training/``
3. Parse + sessionize + featurize + train an Isolation Forest
   -> ``models/{drain3.bin, vocab.npy, iforest.bin}``
4. ``docker compose up -d`` the ELK stack
5. Wait for Kibana and create data views (``logs-raw``, ``logs-scored``)
6. Drip-feed live logs into ``data/live/app.log``
7. Run the streaming scorer for a few polls
8. Print URLs so you can open Kibana and explore

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --force-retrain        # rebuild model from scratch
    python scripts/run_demo.py --skip-stack           # assume stack is already up
    python scripts/run_demo.py --live-blocks 30 --live-anomalies 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Make the in-tree bootstrap_kibana importable without installing scripts as a package.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import bootstrap_kibana  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.yml"
TRAINING_DIR = REPO_ROOT / "data" / "training"
LIVE_LOG = REPO_ROOT / "data" / "live" / "app.log"
MODELS_DIR = REPO_ROOT / "models"


def banner(msg: str) -> None:
    bar = "=" * len(msg)
    print(f"\n{bar}\n{msg}\n{bar}")


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    pretty = " ".join(str(c) for c in cmd)
    print(f"$ {pretty}")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


# --------------------------------------------------------------------------
# Step 1: pre-flight
# --------------------------------------------------------------------------

def check_docker() -> None:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Docker CLI not found on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("Docker CLI hung — is the daemon healthy?") from exc

    if result.returncode != 0:
        raise SystemExit(
            f"Docker daemon not reachable. Start Docker Desktop and retry.\n"
            f"stderr:\n{result.stderr}"
        )
    print(f"[preflight] Docker server: {result.stdout.strip()}")


# --------------------------------------------------------------------------
# Step 2 + 3: build the trained pipeline if absent
# --------------------------------------------------------------------------

def trained_artifacts_present(force: bool) -> bool:
    if force:
        return False
    return all(
        (MODELS_DIR / name).exists()
        for name in ("drain3.bin", "vocab.npy", "iforest.bin")
    )


def build_training_pipeline(*, blocks: int, anomalies: int, seed: int) -> None:
    banner("Generating synthetic training corpus")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    corpus = TRAINING_DIR / "corpus.log"
    labels = TRAINING_DIR / "anomaly_label.csv"
    run([
        sys.executable, "scripts/inject_logs.py",
        "--output", str(corpus),
        "--labels-out", str(labels),
        "--blocks", str(blocks),
        "--anomalies", str(anomalies),
        "--rate", "0",
        "--truncate",
        "--seed", str(seed),
    ])

    banner("Parsing corpus with Drain3")
    run([
        sys.executable, "-m", "log_detector.cli",
        "parse", "--input", str(corpus),
        "--output", str(REPO_ROOT / "data" / "processed" / "parsed.csv"),
    ])

    banner("Building features and labels")
    run([
        sys.executable, "-m", "log_detector.cli",
        "features",
        "--parsed", str(REPO_ROOT / "data" / "processed" / "parsed.csv"),
        "--labels", str(labels),
    ])

    banner("Training Isolation Forest")
    run([sys.executable, "-m", "log_detector.cli", "train", "--model", "iforest"])


# --------------------------------------------------------------------------
# Step 4: bring up the docker stack
# --------------------------------------------------------------------------

def start_stack() -> None:
    banner("Starting ELK stack")
    run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"])


def stop_stack() -> None:
    banner("Stopping ELK stack")
    run(["docker", "compose", "-f", str(COMPOSE_FILE), "down"])


# --------------------------------------------------------------------------
# Step 6: inject live logs
# --------------------------------------------------------------------------

def inject_live(*, blocks: int, anomalies: int, rate: float, seed: int) -> None:
    banner("Injecting live logs into data/live/app.log")
    if LIVE_LOG.exists():
        LIVE_LOG.unlink()
    run([
        sys.executable, "scripts/inject_logs.py",
        "--output", str(LIVE_LOG),
        "--blocks", str(blocks),
        "--anomalies", str(anomalies),
        "--rate", str(rate),
        "--seed", str(seed),
    ])


# --------------------------------------------------------------------------
# Step 7: run the scorer for a few polls
# --------------------------------------------------------------------------

def run_scorer(*, polls: int, interval: float, threshold: float) -> None:
    banner(f"Running scorer for {polls} polls")
    run([
        sys.executable, "-m", "log_detector.cli",
        "serve",
        "--model", str(MODELS_DIR / "iforest.bin"),
        "--threshold", str(threshold),
        "--interval", str(interval),
        "--max-iters", str(polls),
    ])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Rebuild the trained model even if artifacts exist.")
    ap.add_argument("--skip-stack", action="store_true",
                    help="Don't run docker compose up (assume the stack is already running).")
    ap.add_argument("--train-blocks", type=int, default=200)
    ap.add_argument("--train-anomalies", type=int, default=20)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--live-blocks", type=int, default=15)
    ap.add_argument("--live-anomalies", type=int, default=3)
    ap.add_argument("--live-rate", type=float, default=20.0)
    ap.add_argument("--live-seed", type=int, default=42)
    ap.add_argument("--polls", type=int, default=3)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--kibana-url", default="http://localhost:5601")
    ap.add_argument("--filebeat-wait", type=float, default=8.0,
                    help="Seconds to wait after injection before scoring (lets Filebeat ship).")
    args = ap.parse_args(argv)

    banner("AI-Log-Detector end-to-end demo")
    print(f"Repo root: {REPO_ROOT}")

    check_docker()

    if trained_artifacts_present(args.force_retrain):
        banner("Skipping training (artifacts already present in models/)")
        print("Use --force-retrain to rebuild.")
    else:
        build_training_pipeline(
            blocks=args.train_blocks,
            anomalies=args.train_anomalies,
            seed=args.train_seed,
        )

    if not args.skip_stack:
        start_stack()

    banner("Waiting for Kibana and creating data views")
    try:
        result = bootstrap_kibana.bootstrap(args.kibana_url, timeout=240.0)
    except bootstrap_kibana.KibanaError as exc:
        print(f"[kibana] error: {exc}", file=sys.stderr)
        return 2
    print(f"[kibana] ready after {result.waited_seconds:.1f}s")
    for t in result.created:
        print(f"[kibana] created data view: {t}")
    for t in result.already_existed:
        print(f"[kibana] already existed:     {t}")

    inject_live(
        blocks=args.live_blocks,
        anomalies=args.live_anomalies,
        rate=args.live_rate,
        seed=args.live_seed,
    )

    print(f"[demo] sleeping {args.filebeat_wait}s so Filebeat can ship the new lines...")
    time.sleep(args.filebeat_wait)

    run_scorer(
        polls=args.polls,
        interval=args.poll_interval,
        threshold=args.threshold,
    )

    banner("Demo complete")
    print(f"Open Kibana:           {args.kibana_url}/app/discover")
    print(f"Manage data views:     {args.kibana_url}/app/management/kibana/dataViews")
    print()
    print("Suggested queries in Discover:")
    print('  Data view: logs-scored      KQL: is_anomaly : true')
    print('  Data view: logs-scored      KQL: score > 0.05')
    print('  Data view: logs-raw         KQL: message : *exception* or message : *ERROR*')
    print()
    print(f"Tear down the stack: docker compose -f {COMPOSE_FILE} down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
