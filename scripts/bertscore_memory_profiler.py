#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, csv, importlib, logging, random, re, sys
from pathlib import Path
from typing import List, Tuple
import psutil, tracemalloc

# ── tiny helpers ────────────────────────────────────────────────────────────
def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def make_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "memory_profile.log"

    fmt = logging.Formatter("%(asctime)s  [%(label)s]  %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    lg = logging.getLogger(out_dir.name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    sh = logging.StreamHandler(sys.stdout);  sh.setFormatter(fmt); lg.addHandler(sh)
    fh = logging.FileHandler(log_path, "w"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


@contextlib.contextmanager
def mem_span(lg: logging.Logger, tag: str):
    snap = tracemalloc.take_snapshot(); extra = {"label": tag}
    try:   yield
    finally:
        lg.info("RSS=%.1f MB", rss_mb(), extra=extra)
        lg.info("--- diff for %s ---", tag, extra=extra)
        for s in tracemalloc.take_snapshot().compare_to(snap, "lineno")[:10]:
            lg.info(str(s), extra=extra)


# ── cheap sentence generator ────────────────────────────────────────────────
_WORDS = ("time person year way day thing man world life hand part child eye woman "
          "place work week case point government company number group problem fact").split()
rand_sent = lambda n: " ".join(random.choice(_WORDS) for _ in range(n)) + "."

def make_pair(kind: str) -> Tuple[str, str]:
    a, b = {"short":(12,15), "medium":(90,100)}.get(kind, (260,280))
    return rand_sent(a), rand_sent(b)


# ── the actual runs ─────────────────────────────────────────────────────────
def run_direct(lg: logging.Logger):
    lg.info("RSS=%.1f MB", rss_mb(), extra={"label":"direct|start"})
    with mem_span(lg, "direct|model_load"):
        BERTScore = importlib.import_module(
            "autometrics.metrics.reference_based.BERTScore"
        ).BERTScore
        BERTScore(model="roberta-large", persistent=False)
    lg.info("RSS=%.1f MB", rss_mb(), extra={"label":"direct|final"})


def run_single(kind: str, n: int, out_dir: str, *, persistent: bool):
    lg = make_logger(Path(out_dir))
    lg.info("RSS=%.1f MB", rss_mb(), extra={"label":f"{kind}|start"})

    # import & construct
    with mem_span(lg, f"{kind}|before_create"):
        BERTScore = importlib.import_module(
            "autometrics.metrics.reference_based.BERTScore"
        ).BERTScore
        metric = BERTScore(model="roberta-large", persistent=persistent)

    # first call → weights appear
    c, r = make_pair(kind)
    with mem_span(lg, f"{kind}|model_load"):
        metric.calculate("<src>", c, [r])

    # further inferences
    for i in range(2, n+1):
        c, r = make_pair(kind)
        with mem_span(lg, f"{kind}|infer{i-1}"):
            metric.calculate("<src>", c, [r])

    if not persistent:
        with mem_span(lg, f"{kind}|after_unload"):
            del metric

    lg.info("RSS=%.1f MB", rss_mb(), extra={"label":f"{kind}|final"})


# ── analysis helper ─────────────────────────────────────────────────────────
_FINAL_RE = re.compile(r"\[([^|\]]+\|final)]\s+RSS=([0-9.]+) MB")
def parse_rss(p: Path) -> List[Tuple[str,float]]:
    out=[];  txt=p.read_text().splitlines()
    for ln in txt:
        if m:=_FINAL_RE.search(ln): out.append((m.group(1),float(m.group(2))))
    return out

def analyze(out_dir: str):
    dest = Path(out_dir); dest.mkdir(parents=True, exist_ok=True)
    rows=[]
    for log in Path("outputs").glob("*/memory_profile.log"):
        suite=log.parent.name
        rows += [(suite,*r) for r in parse_rss(log)]
    rows.sort()
    with (dest/"memory_summary.csv").open("w",newline="") as fp:
        csv.writer(fp).writerows([("suite","tag","rss_mb"),*rows])
    print(f"[analyze] wrote summary to {dest/'memory_summary.csv'}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    tracemalloc.start()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct","test","analyze"], required=True)
    ap.add_argument("--length", choices=["short","medium","long"])
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--persistent", action="store_true")
    a = ap.parse_args()

    if a.mode=="direct":
        run_direct(make_logger(Path(a.output_dir)))
    elif a.mode=="test":
        if not a.length: ap.error("--length required with --mode test")
        run_single(a.length, a.num_samples, a.output_dir, persistent=a.persistent)
    else:
        analyze(a.output_dir)

if __name__=="__main__":
    main()