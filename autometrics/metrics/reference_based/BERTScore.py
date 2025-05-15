import gc
import logging
import os
import time
from typing import Tuple

import psutil
import torch
import bert_score

from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import (
    ReferenceBasedMultiMetric,
)

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def log_memory(label: str) -> Tuple[float, float]:
    """Log **RSS, PSS, and shared** memory, plus CUDA memory if available.

    Returns
    -------
    (rss_mb, gpu_mb)
        RSS in MB and GPU‑allocated MB.  We keep the signature unchanged so that
        callers elsewhere in the codebase don’t break, but we also compute and
        log PSS / Shared which are usually where the *file‑mapped* model weights
        show up.
    """
    process = psutil.Process(os.getpid())

    # psutil docs: memory_full_info() exposes fields like pss & shared on Linux.
    mem_full = process.memory_full_info()
    rss_mb = mem_full.rss / (1024 * 1024)

    # pss/shared may not exist on non‑Linux systems, so guard with getattr
    pss_mb = getattr(mem_full, "pss", 0) / (1024 * 1024)
    shared_mb = getattr(mem_full, "shared", 0) / (1024 * 1024)

    gpu_mb = 0.0
    if torch.cuda.is_available():
        try:
            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass

    logger.info(
        f"MEMORY [{label}]: RSS={rss_mb:.1f} MB  PSS={pss_mb:.1f} MB  SHARED={shared_mb:.1f} MB  GPU={gpu_mb:.1f} MB"
    )

    # Keep the original two‑value return for callers downstream
    return rss_mb, gpu_mb


# -----------------------------------------------------------------------------
# Metric logic
# -----------------------------------------------------------------------------

def compute_bertscore(
    original,
    output,
    references,
    *,
    model: str = "roberta-large",
    type: str = "all",
    compare_to_original: bool = False,
):
    """Compute BERTScore and log detailed memory usage along the way."""

    log_memory(f"compute_bertscore START – {len(original)} samples")
    start_time = time.time()

    all_origs, all_refs, all_cands = [], [], []

    total_input_chars = sum(len(str(o)) for o in original)
    total_output_chars = sum(len(str(o)) for o in output)
    total_ref_chars = sum(len(str(r)) for refs in references for r in refs)
    logger.info(
        f"Input={total_input_chars} chars | Output={total_output_chars} chars | Refs={total_ref_chars} chars"
    )

    for orig, hyp, refs in zip(original, output, references):
        for ref in refs:
            all_refs.append(ref.lower())
            all_cands.append(hyp.lower())
            all_origs.append(orig.lower())

    log_memory("before bert_score.score call")

    method = "orig" if compare_to_original else "ref"
    logger.info(f"Calling bert_score.score (compare_to_{method}, model={model})")

    if compare_to_original:
        (P, R, F), _ = bert_score.score(
            all_cands,
            all_origs,
            lang="en",
            return_hash=True,
            verbose=False,
            idf=False,
            model_type=model,
        )
    else:
        (P, R, F), _ = bert_score.score(
            all_cands,
            all_refs,
            lang="en",
            return_hash=True,
            verbose=False,
            idf=False,
            model_type=model,
        )

    log_memory("after bert_score.score call")

    ind = 0
    pscores, rscores, fscores = [], [], []
    for _, _, refs in zip(original, output, references):
        tmp_p, tmp_r, tmp_f = [], [], []
        for _ in refs:
            tmp_f.append(F[ind].item())
            tmp_p.append(P[ind].item())
            tmp_r.append(R[ind].item())
            ind += 1
        pscores.append(max(tmp_p))
        rscores.append(max(tmp_r))
        fscores.append(max(tmp_f))

    assert len(pscores) == len(original)

    elapsed = time.time() - start_time
    logger.info(f"compute_bertscore completed in {elapsed:.2f} s")
    log_memory("compute_bertscore END")

    if type == "precision":
        return pscores
    elif type == "recall":
        return rscores
    elif type == "f1":
        return fscores
    else:
        return pscores, rscores, fscores


class BERTScore(ReferenceBasedMultiMetric):
    """Reference‑based metric wrapper that keeps track of memory responsibly."""

    def __init__(self, model: str = "roberta-large", persistent: bool = True, **kwargs):
        log_memory(f"BERTScore.__init__ START model={model} persistent={persistent}")
        self.model = model
        self.persistent = persistent
        self._model_loaded = False

        submetrics = ["P", "R", "F"]
        name = f"BERTScore_{model}"
        desc = (
            "BERTScore measures token‑level semantic similarity between candidate and reference"
        )

        super().__init__(
            name=name,
            description=desc,
            model=model,
            submetric_names=[f"BERTScore{sub}_{model}" for sub in submetrics],
            **kwargs,
        )

        if self.persistent:
            logger.info("persistent=True – eager model load")
            self._load_model()

        self.exclude_from_cache_key("persistent")
        log_memory("BERTScore.__init__ END")

    # ------------------------------------------------------------------
    # Model loading / unloading
    # ------------------------------------------------------------------
    def _load_model(self):
        log_memory("_load_model START")
        # Nothing to do explicitly; bert_score will lazy‑load. Flag set for logic.
        self._model_loaded = True
        if hasattr(bert_score, "_models"):
            logger.info(f"bert_score has {len(bert_score._models)} cached models before load")
        log_memory("_load_model END")

    def _unload_model(self):
        log_memory("_unload_model START")
        if hasattr(bert_score, "_models"):
            logger.info(f"Clearing bert_score._models with keys: {list(bert_score._models.keys())}")
            bert_score._models = {}
        if hasattr(bert_score, "_idf_dict"):
            bert_score._idf_dict = {}
        self._model_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory("_unload_model END")

    # ------------------------------------------------------------------
    # Metric calculation
    # ------------------------------------------------------------------
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        log_memory(f"_calculate_batched_impl START – {len(inputs)} samples")
        if references is None:
            references = [None] * len(inputs)
        if not self._model_loaded:
            self._load_model()
        try:
            results = compute_bertscore(
                inputs,
                outputs,
                references,
                model=self.model,
                type="all",
            )
            return results
        finally:
            if not self.persistent:
                self._unload_model()
            log_memory("_calculate_batched_impl END")

    def _calculate_impl(self, input, output, references=None, **kwargs):
        log_memory("_calculate_impl START")
        if references is None:
            references = []
        if not self._model_loaded:
            self._load_model()
        try:
            p, r, f = compute_bertscore(
                [input],
                [output],
                [references],
                model=self.model,
                type="all",
            )
            return p[0], r[0], f[0]
        finally:
            if not self.persistent:
                self._unload_model()
            log_memory("_calculate_impl END")
