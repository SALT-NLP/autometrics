#!/usr/bin/env python3
"""
roberta_memory_experiment.py  ────────────────────────────────────────
A one‑shot script to explore *where* RAM goes when we load **RoBERTa‑large**
under several settings.

Scenarios
─────────
import_only              – just import transformers + torch (baseline)
low_cpu_mem_true         – load with   low_cpu_mem_usage=True  (no touching)
low_cpu_mem_false        – load with   low_cpu_mem_usage=False (no touching)
low_cpu_mem_true_touch   – same as above, but iterate over every param
low_cpu_mem_false_touch  – ditto for low_cpu_mem_usage=False
frozen_6_layers          – low_cpu_mem_usage=True   → keep only 6 encoder blocks

Each scenario is executed in **its own subprocess** so RSS numbers aren’t
polluted by earlier runs.  We write a tiny JSON‑lines file per run:
    { "label": "after_load", "rss_mb": 652.4, "timestamp": "15:46:00" }
The parent then aggregates and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psutil

RESULTS_DIR = Path(__file__).resolve().parent / "roberta_mem_results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "roberta-large"

###############################################################################
# Small helpers
###############################################################################

def rss_mb() -> float:
    """Return Resident‑Set Size for *this* process, in MB (binary)."""
    return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20  # MiB


def log(label: str) -> None:
    print(label.ljust(22), f"RSS={rss_mb():.1f}\u202fMB", flush=True)


def dump(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

###############################################################################
# Child logic – runs ONE scenario and exits
###############################################################################

def run_child(cfg: Dict[str, Any]) -> None:
    label      = cfg["name"]
    low_cpu    = cfg["low_cpu"]
    touch      = cfg.get("touch", False)
    num_layers = cfg.get("num_layers")

    rows: List[Dict[str, Any]] = []

    log(f"[{label}] start")
    rows.append({"label": "start", "rss_mb": rss_mb(), "ts": datetime.now().isoformat(timespec="seconds")})

    # ---------------------------------------------------------------------
    # Load model under test
    # ---------------------------------------------------------------------
    print(f"[{label}] Loading model (low_cpu_mem_usage={low_cpu}) …", flush=True)
    from transformers import AutoModel

    load_kwargs = {"low_cpu_mem_usage": low_cpu}
    # If we plan to *touch* parameters and we used low‑cpu loader, also pass a
    # device map so weights are eagerly materialised on CPU – avoiding meta
    # tensors that can’t be .item()’d.
    if low_cpu and touch:
        load_kwargs["device_map"] = "cpu"

    model = AutoModel.from_pretrained(MODEL_NAME, **load_kwargs)

    # Trim layers if requested ------------------------------------------------
    if num_layers is not None:
        print(f"[{label}] Trimming to first {num_layers} layers …", flush=True)
        enc = model.encoder if hasattr(model, "encoder") else model
        enc.layer = enc.layer[:num_layers]

    # ---------------------------------------------------------------------
    log(f"[{label}] after_load")
    rows.append({"label": "after_load", "rss_mb": rss_mb(), "ts": datetime.now().isoformat(timespec="seconds")})

    # ---------------------------------------------------------------------
    # Materialise + touch every parameter (optional)
    # ---------------------------------------------------------------------
    if touch:
        if any(p.is_meta for p in model.parameters()):
            print(f"[{label}] Materialising meta tensors → CPU …", flush=True)
            # torch>=2.1 provides to_empty; we can move meta -> cpu by allocating
            # a *new* empty tensor of the same shape, then copying the data via
            # _load_from_state_dict hooks already held by HF.
            from transformers.modeling_utils import load_state_dict
            state_dict = load_state_dict(MODEL_NAME)
            model.load_state_dict(state_dict)

        print(f"[{label}] Touching every parameter (sum -> float) …", flush=True)
        total = 0.0
        for p in model.parameters():
            if p.is_meta:
                continue  # those never materialised (rare if device_map="cpu")
            total += p.detach().float().sum().item()
        print(f"[{label}]   dummy checksum: {total:.4f}")
        log(f"[{label}] after_touch")
        rows.append({"label": "after_touch", "rss_mb": rss_mb(), "ts": datetime.now().isoformat(timespec="seconds")})

    # ---------------------------------------------------------------------
    # Persist results and finish
    # ---------------------------------------------------------------------
    out_file = RESULTS_DIR / ("tmp" + next(tempfile._get_candidate_names()) + ".jsonl")
    dump(out_file, rows)

    print(f"[{label}] Finished; wrote {len(rows)} measurements to {out_file}")
    # IMPORTANT: this magic line allows parent to capture the path robustly
    print(f"RESULT_PATH: {out_file}")

###############################################################################
# Parent logic – orchestrates every scenario in its own Python subprocess
###############################################################################

SCENARIOS = [
    {"name": "import_only",            "low_cpu": False, "touch": False, "num_layers": None},
    {"name": "low_cpu_mem_true",       "low_cpu": True,  "touch": False, "num_layers": None},
    {"name": "low_cpu_mem_false",      "low_cpu": False, "touch": False, "num_layers": None},
    {"name": "low_cpu_mem_true_touch", "low_cpu": True,  "touch": True,  "num_layers": None},
    {"name": "low_cpu_mem_false_touch","low_cpu": False, "touch": True,  "num_layers": None},
    {"name": "frozen_6_layers",        "low_cpu": True,  "touch": True,  "num_layers": 6},
]


def run_parent() -> None:
    print()
    result_paths: List[Path] = []

    for cfg in SCENARIOS:
        name = cfg["name"]
        print(f"\n>>> Running scenario: {name}")
        try:
            result = subprocess.run(
                [sys.executable, __file__, "--child", "--config", json.dumps(cfg)],
                capture_output=True,
                text=True,
                check=False,
            )
            print(result.stdout)
            if result.returncode == 0:
                # Parse RESULT_PATH line
                for line in result.stdout.splitlines():
                    if line.startswith("RESULT_PATH:"):
                        result_paths.append(Path(line.split(":", 1)[1].strip()))
                        break
            else:
                print(result.stderr or "(no stderr)")
                print(f"!!! Scenario {name} FAILED", file=sys.stderr)
        except Exception as e:
            print(f"!!! Scenario {name} FAILED with {e}")

    # ------------------------------------------------------------------
    # Aggregate & show summary
    # ------------------------------------------------------------------
    if not result_paths:
        print("No successful runs – nothing to summarise.")
        return

    print("\n=== RSS summary (MB) ===")
    for path in result_paths:
        with path.open() as f:
            for row in map(json.loads, f):
                print(path.name.ljust(15), row["label"].ljust(15), f"{row['rss_mb']:.1f} MB", "at", row["ts"].split("T")[-1])

###############################################################################
# Entrypoint routing
###############################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help="INTERNAL – run the child worker")
    p.add_argument("--config", type=str, help="INTERNAL – JSON config for the child")
    args = p.parse_args()

    if args.child:
        cfg = json.loads(args.config)
        run_child(cfg)
    else:
        run_parent()
