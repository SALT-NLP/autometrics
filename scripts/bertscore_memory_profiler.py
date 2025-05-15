#!/usr/bin/env python3
"""
BERTScore Memory Profiler
=========================

• direct   : sanity-check that the model loads
• test     : run synthetic short / medium / long benchmarks
• analyze  : aggregate every *|final* RSS line into a CSV
"""

from __future__ import annotations
import argparse
import contextlib
import csv
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import psutil
import tracemalloc

from autometrics.metrics.reference_based.BERTScore import BERTScore


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def make_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "memory_profile.log"

    fmt = logging.Formatter(
        "%(asctime)s  [%(label)s]  %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    log = logging.getLogger(f"Profiler:{out_dir.name}")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(log_path, "w")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


@contextlib.contextmanager
def mem_span(logger: logging.Logger, tag: str):
    extra = {"label": tag}
    snap = tracemalloc.take_snapshot()
    try:
        yield
    finally:
        logger.info("RSS=%.1f MB", rss_mb(), extra=extra)
        logger.info("--- diff for %s ---", tag, extra=extra)
        for stat in tracemalloc.take_snapshot().compare_to(snap, "lineno")[:10]:
            logger.info(str(stat), extra=extra)


# ──────────────────────────────────────────────────────────────────────────────
# synthetic text factory
# ──────────────────────────────────────────────────────────────────────────────
_WORDS = (
    "time person year way day thing man world life hand part child eye woman "
    "place work week case point government company number group problem fact"
).split()


def random_sentence(n_words: int) -> str:
    return " ".join(random.choice(_WORDS) for _ in range(n_words)) + "."


def make_pair(length: str) -> Tuple[str, str]:
    if length == "short":
        n_cand, n_ref = 12, 15
    elif length == "medium":
        n_cand, n_ref = 90, 100
    else:  # long
        n_cand, n_ref = 260, 280
    return random_sentence(n_cand), random_sentence(n_ref)


# ──────────────────────────────────────────────────────────────────────────────
# bench-runs
# ──────────────────────────────────────────────────────────────────────────────
def run_direct(logger: logging.Logger):
    with mem_span(logger, "direct|before_api"):
        BERTScore(model="roberta-large", persistent=False)
    logger.info("RSS=%.1f MB", rss_mb(), extra={"label": "direct|final"})


def run_single_test(
    length: str,
    num_samples: int,
    out_dir: str,
    *,
    persistent: bool = False,
):
    logger = make_logger(Path(out_dir))

    with mem_span(logger, f"{length}|before_create"):
        metric = BERTScore(model="roberta-large", persistent=persistent)

    for idx in range(1, num_samples + 1):
        cand, ref = make_pair(length)
        # key fix ⤵︎ – supply a *non-empty* reference list
        with mem_span(logger, f"{length}|calc{idx}"):
            metric.calculate("<src>", cand, [ref])

    if not persistent:
        with mem_span(logger, f"{length}|after_unload"):
            del metric

    logger.info("RSS=%.1f MB", rss_mb(), extra={"label": f"{length}|final"})


# ──────────────────────────────────────────────────────────────────────────────
# analyze mode
# ──────────────────────────────────────────────────────────────────────────────
_FINAL_RE = re.compile(r"\[([^|\]]+\|final)]\s+RSS=([0-9.]+) MB")


def parse_rss(path: Path) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for line in path.read_text().splitlines():
        if m := _FINAL_RE.search(line):
            rows.append((m.group(1), float(m.group(2))))
    return rows


def run_analyze(out_dir: str):
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    csv_file = dest / "memory_summary.csv"

    all_rows: List[Tuple[str, str, float]] = []
    for log_path in Path("outputs").glob("*/memory_profile.log"):
        suite = log_path.parent.name
        for tag, rss in parse_rss(log_path):
            all_rows.append((suite, tag, rss))

    all_rows.sort()
    with csv_file.open("w", newline="") as fp:
        csv.writer(fp).writerows([("suite", "tag", "rss_mb"), *all_rows])

    print(f"[analyze] wrote summary to {csv_file}")


# ──────────────────────────────────────────────────────────────────────────────
# cli
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    tracemalloc.start()

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "test", "analyze"], required=True)
    ap.add_argument("--length", choices=["short", "medium", "long"])
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--persistent", action="store_true")
    args = ap.parse_args()

    if args.mode == "direct":
        run_direct(make_logger(Path(args.output_dir)))

    elif args.mode == "test":
        if args.length is None:
            ap.error("--length is required when --mode test")
        run_single_test(
            args.length,
            args.num_samples,
            args.output_dir,
            persistent=args.persistent,
        )

    else:
        run_analyze(args.output_dir)


if __name__ == "__main__":
    main()