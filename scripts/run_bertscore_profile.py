#!/usr/bin/env python3
"""
Run the BERTScore memory profiler in isolated subprocesses and then aggregate results.

This script will:
  1. Profile the raw bert_score.score API load.
  2. Profile short, medium and long tests in separate processes.
  3. Run the analysis step to stitch together all timelines and produce plots.
"""
import os
import sys
import subprocess
import time

# Path to your profiler CLI
PROFILER = os.path.join("scripts", "bertscore_memory_profiler.py")

def call_mode(mode, **kwargs):
    cmd = [sys.executable, PROFILER, "--mode", mode]
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k.replace('_','-')}", str(v)]
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    os.makedirs("outputs", exist_ok=True)
    start = time.time()

    print("\n1) Direct API load profile")
    call_mode("direct", output_dir="outputs/direct")

    print("\n2) Short-text test")
    call_mode("test", length="short", num_samples=2, output_dir="outputs/short")

    print("\n3) Medium-text test")
    call_mode("test", length="medium", num_samples=2, output_dir="outputs/medium")

    print("\n4) Long-text test (persistent)")
    call_mode("test", length="long", persistent=True, num_samples=1, output_dir="outputs/long")

    print("\n5) Aggregate & analyze")
    call_mode("analyze", output_dir="outputs/aggregate")

    print(f"\nAll profiling done in {time.time()-start:.1f}s")
    print("Results:")
    for d in ["direct","short","medium","long","aggregate"]:
        print(f"  - outputs/{d}")

if __name__ == "__main__":
    sys.exit(main())