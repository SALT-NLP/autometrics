# NOTE: This file was refactored to delay heavy metric instantiation until
# they are actually needed.  We provide factory helpers that build metrics on
# demand, with optional common parameters such as cache_dir and seed.

from __future__ import annotations

from typing import List, Dict, Type, Any
import inspect
import os

# ---------------------------------------------------------------------------
# Import metric *classes* only (light-weight) – do NOT instantiate here.
# Each import is wrapped in try/except so users with a partial install (e.g.
# generated-only mode with just the base deps) can still import MetricBank
# and get back whichever metrics *can* be loaded. Missing metrics are logged
# once at import time and simply omitted from the registries.
# ---------------------------------------------------------------------------

import warnings as _warnings


def _try_import(module_path: str, *names: str):
    """Import ``names`` from ``module_path``; return dict of available ones."""
    try:
        module = __import__(module_path, fromlist=list(names))
    except Exception as exc:  # noqa: BLE001 – any import failure counts
        _warnings.warn(
            f"[MetricBank] Skipping {module_path} ({', '.join(names)}): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    return {n: getattr(module, n) for n in names if hasattr(module, n)}


_rb = {}
for _mod, _cls in [
    ("autometrics.metrics.reference_based.BLEU", "BLEU"),
    ("autometrics.metrics.reference_based.CHRF", "CHRF"),
    ("autometrics.metrics.reference_based.TER", "TER"),
    ("autometrics.metrics.reference_based.GLEU", "GLEU"),
    ("autometrics.metrics.reference_based.SARI", "SARI"),
    ("autometrics.metrics.reference_based.BERTScore", "BERTScore"),
    ("autometrics.metrics.reference_based.ROUGE", "ROUGE"),
    ("autometrics.metrics.reference_based.MOVERScore", "MOVERScore"),
    ("autometrics.metrics.reference_based.BARTScore", "BARTScore"),
    ("autometrics.metrics.reference_based.UniEvalDialogue", "UniEvalDialogue"),
    ("autometrics.metrics.reference_based.UniEvalSum", "UniEvalSum"),
    ("autometrics.metrics.reference_based.CIDEr", "CIDEr"),
    ("autometrics.metrics.reference_based.METEOR", "METEOR"),
    ("autometrics.metrics.reference_based.ParaScore", "ParaScore"),
    ("autometrics.metrics.reference_based.YiSi", "YiSi"),
    ("autometrics.metrics.reference_based.MAUVE", "MAUVE"),
    ("autometrics.metrics.reference_based.PseudoPARENT", "PseudoPARENT"),
    ("autometrics.metrics.reference_based.NIST", "NIST"),
    ("autometrics.metrics.reference_based.IBLEU", "IBLEU"),
    ("autometrics.metrics.reference_based.UpdateROUGE", "UpdateROUGE"),
    ("autometrics.metrics.reference_based.BLEURT", "BLEURT"),
    ("autometrics.metrics.reference_based.LENS", "LENS"),
    ("autometrics.metrics.reference_based.CharCut", "CharCut"),
    ("autometrics.metrics.reference_based.InfoLM", "InfoLM"),
]:
    _rb.update(_try_import(_mod, _cls))

_rb.update(
    _try_import(
        "autometrics.metrics.reference_based.StringSimilarity",
        "LevenshteinDistance",
        "LevenshteinRatio",
        "HammingDistance",
        "JaroSimilarity",
        "JaroWinklerSimilarity",
        "JaccardDistance",
    )
)

_rf = {}
for _mod, _cls in [
    ("autometrics.metrics.reference_free.FKGL", "FKGL"),
    ("autometrics.metrics.reference_free.UniEvalFact", "UniEvalFact"),
    ("autometrics.metrics.reference_free.Perplexity", "Perplexity"),
    ("autometrics.metrics.reference_free.ParaScoreFree", "ParaScoreFree"),
    ("autometrics.metrics.reference_free.INFORMRewardModel", "INFORMRewardModel"),
    ("autometrics.metrics.reference_free.PRMRewardModel", "MathProcessRewardModel"),
    ("autometrics.metrics.reference_free.SummaQA", "SummaQA"),
    ("autometrics.metrics.reference_free.DistinctNGram", "DistinctNGram"),
    ("autometrics.metrics.reference_free.FastTextToxicity", "FastTextToxicity"),
    ("autometrics.metrics.reference_free.FastTextNSFW", "FastTextNSFW"),
    ("autometrics.metrics.reference_free.FastTextEducationalValue", "FastTextEducationalValue"),
    ("autometrics.metrics.reference_free.SelfBLEU", "SelfBLEU"),
    ("autometrics.metrics.reference_free.FactCC", "FactCC"),
    ("autometrics.metrics.reference_free.Toxicity", "Toxicity"),
    ("autometrics.metrics.reference_free.GRMRewardModel", "GRMRewardModel"),
    ("autometrics.metrics.reference_free.LDLRewardModel", "LDLRewardModel"),
    ("autometrics.metrics.reference_free.Sentiment", "Sentiment"),
    ("autometrics.metrics.reference_free.LENS_SALSA", "LENS_SALSA"),
]:
    _rf.update(_try_import(_mod, _cls))

# Preserve the original metric ordering for tests / reproducibility.
_REFERENCE_BASED_ORDER = [
    "BLEU", "CHRF", "TER", "GLEU", "SARI", "BERTScore", "ROUGE", "MOVERScore",
    "BARTScore", "UniEvalDialogue", "UniEvalSum", "CIDEr", "METEOR", "BLEURT",
    "LevenshteinDistance", "LevenshteinRatio", "HammingDistance", "JaroSimilarity",
    "JaroWinklerSimilarity", "JaccardDistance", "ParaScore", "YiSi", "MAUVE",
    "PseudoPARENT", "NIST", "IBLEU", "UpdateROUGE", "LENS", "CharCut", "InfoLM",
]
_REFERENCE_FREE_ORDER = [
    "FKGL", "UniEvalFact", "Perplexity", "ParaScoreFree", "INFORMRewardModel",
    "MathProcessRewardModel", "SummaQA", "DistinctNGram", "FastTextToxicity",
    "FastTextNSFW", "FastTextEducationalValue", "SelfBLEU", "FactCC", "Toxicity",
    "Sentiment", "GRMRewardModel", "LENS_SALSA", "LDLRewardModel",
]

# Expose successfully-loaded classes at module scope for ``from ... import X``
# style access elsewhere in the codebase.
globals().update(_rb)
globals().update(_rf)

# ---------------------------------------------------------------------------
# Metric class registries
# ---------------------------------------------------------------------------

reference_based_metric_classes: List[Type] = [_rb[n] for n in _REFERENCE_BASED_ORDER if n in _rb]
reference_free_metric_classes: List[Type] = [_rf[n] for n in _REFERENCE_FREE_ORDER if n in _rf]
all_metric_classes: List[Type] = reference_based_metric_classes + reference_free_metric_classes

# ---------------------------------------------------------------------------
# Default per-metric kwargs (to replicate previous behaviour)
# ---------------------------------------------------------------------------

_DEFAULT_EXTRA_KWARGS: Dict[str, Dict[str, Any]] = {
    "Perplexity": {"batch_size": 2},
}

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

# GPU allocation helper - moved to lazy import to avoid 4.8s startup delay

def _instantiate_metric(cls: Type, kwargs: Dict[str, Any]):
    """Instantiate a metric class with the provided kwargs (already filtered)."""
    try:
        return cls(**kwargs)
    except Exception as e:
        print(f"[MetricBank] Failed to instantiate {cls.__name__} with kwargs {kwargs}: {e}. Trying default constructor …")
        try:
            return cls()
        except Exception as e2:
            print(f"[MetricBank] Giving up on {cls.__name__}: {e2}")
            return None


def _get_cache_dir() -> str:
    """
    Get the cache directory from environment variable AUTOMETRICS_CACHE_DIR,
    with fallback to "./autometrics_cache" if not set.
    
    Returns:
        Cache directory path as string
    """
    return os.environ.get("AUTOMETRICS_CACHE_DIR", "./autometrics_cache")


def build_metrics(
    classes: List[Type],
    cache_dir: str | None = None,
    seed: int | None = None,
    use_cache: bool = True,
    overrides: Dict[str, Dict[str, Any]] | None = None,
    gpu_buffer_ratio: float = 0.10,
) -> List[Any]:
    """Instantiate a list of metric classes with common kwargs and cache override."""
    # Ensure global meta-tensor safe patch is applied exactly once for all models
    try:
        from autometrics.metrics.utils.device_utils import (
            apply_meta_tensor_safe_module_to_patch,
            apply_roberta_token_type_guard,
        )
        apply_meta_tensor_safe_module_to_patch()
        apply_roberta_token_type_guard()
    except Exception as _patch_err:
        print(f"[MetricBank] Warning: failed to apply meta-tensor safe Module.to patch: {_patch_err}")
    common_kwargs = {
        "cache_dir": cache_dir or _get_cache_dir(),
        "seed": seed,
        "use_cache": use_cache,
    }
    overrides = overrides or {}

    # --------------------------------------------------------
    # GPU allocation planning (performed once per batch)
    # Check if any metrics actually need GPUs before attempting allocation
    # --------------------------------------------------------
    allocation_map = {}
    try:
        # Check if any metrics actually need GPUs
        needs_gpu = any(getattr(cls, "gpu_mem", 0) > 0 for cls in classes)
        
        if needs_gpu:
            # Lazy import GPU allocation utilities only when needed (avoids 4.8s startup delay)
            from autometrics.metrics.utils import allocate_gpus
            allocation_map = allocate_gpus(classes, buffer_ratio=gpu_buffer_ratio)
        else:
            # No metrics need GPUs, skip allocation entirely
            pass
    except Exception as e:
        # If GPU allocation fails (e.g., NVML not available), warn and continue with CPU
        print(f"[MetricBank] GPU allocation failed: {e}. Falling back to CPU-only execution.")
        allocation_map = {}

    metrics = []
    for cls in classes:
        # Build merged kwargs
        sig = inspect.signature(cls.__init__)
        merged: Dict[str, Any] = {}
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        # common first
        for k, v in common_kwargs.items():
            if k in sig.parameters or has_var_kw:
                merged[k] = v
        # per-metric overrides highest priority (call-supplied)
        for k, v in overrides.get(cls.__name__, {}).items():
            if k in sig.parameters or has_var_kw:
                merged[k] = v
        # GPU allocation overrides (device/device_map) take precedence over
        # defaults but *not* over explicit user-supplied overrides above.
        for k, v in allocation_map.get(cls.__name__, {}).items():
            if k in merged:
                continue  # user already set explicitly via overrides
            if k in sig.parameters or has_var_kw:
                merged[k] = v
        
        # Debug: Show what kwargs are being passed to each metric
        if allocation_map.get(cls.__name__):
            print(f"[MetricBank] {cls.__name__} kwargs: {merged}")
            print(f"[MetricBank] {cls.__name__} GPU allocation: {allocation_map.get(cls.__name__)}")
        # fill with metric defaults if still missing
        for k, v in _DEFAULT_EXTRA_KWARGS.get(cls.__name__, {}).items():
            if k in sig.parameters or has_var_kw:
                merged.setdefault(k, v)

        metric = _instantiate_metric(cls, merged)
        if metric is None:
            continue
        
        # Debug: Show what device the metric is actually using
        if hasattr(metric, 'model') and metric.model is not None:
            try:
                if hasattr(metric.model, 'device'):
                    print(f"[MetricBank] {cls.__name__} model device: {metric.model.device}")
                elif hasattr(metric.model, 'hf_device_map'):
                    print(f"[MetricBank] {cls.__name__} model hf_device_map: {metric.model.hf_device_map}")
                else:
                    print(f"[MetricBank] {cls.__name__} model has no device info")
            except Exception as e:
                print(f"[MetricBank] {cls.__name__} could not determine model device: {e}")
        
        metrics.append(metric)
    return metrics


def build_reference_based_metrics(**kwargs) -> List[Any]:
    return build_metrics(reference_based_metric_classes, **kwargs)


def build_reference_free_metrics(**kwargs) -> List[Any]:
    return build_metrics(reference_free_metric_classes, **kwargs)


def build_all_metrics(**kwargs) -> List[Any]:
    return build_metrics(all_metric_classes, **kwargs)