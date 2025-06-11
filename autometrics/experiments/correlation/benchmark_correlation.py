#!/usr/bin/env python3
"""
Benchmark Correlation Script
===========================

Run correlation experiments over a chosen dataset using the CorrelationExperiment
framework.  Inspired by `benchmark_utilization.py` but streamlined for
correlation analysis.

Example
-------
python benchmark_correlation.py --dataset summeval --top-k 10 --correlation spearman
"""

import os
import sys
import argparse
import importlib
import logging
from typing import List
import ast

# Note: we intentionally avoid importing heavy autometrics modules at the top-level
# so that running this script with `--help` does not trigger model loads or GPU
# initialisation.  Heavy imports are performed inside `main()` after we know the
# user isn't just requesting usage information.

import pandas as pd

# Ensure project root is on path when executing standalone
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from autometrics.dataset.Dataset import Dataset  # type: ignore

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _scan_dataset_classes() -> List[str]:
    """Lightweight scan of dataset directories to collect class names without importing them."""
    datasets_dir = os.path.join(repo_root, "autometrics", "dataset", "datasets")
    classes = []
    for dir_name in os.listdir(datasets_dir):
        path_dir = os.path.join(datasets_dir, dir_name)
        if not os.path.isdir(path_dir) or dir_name.startswith("__"):
            continue
        module_file = os.path.join(path_dir, f"{dir_name}.py")
        if not os.path.isfile(module_file):
            continue
        try:
            with open(module_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=module_file)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except Exception:
            continue
    return classes


def get_dataset_suggestions() -> List[str]:
    """Return a combined list of directory names and discovered class names for user guidance."""
    datasets_dir = os.path.join(repo_root, "autometrics", "dataset", "datasets")
    dirs = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d)) and not d.startswith("__")]
    return sorted(set(dirs + _scan_dataset_classes()))


def load_dataset(identifier: str):
    """Resolve a dataset identifier to a Dataset instance.

    The identifier can be:
      • <module_dir>                 → e.g. 'summeval' – loads first Dataset subclass in that module
      • <ClassName>                  → e.g. 'HelpSteer2' – searches all modules for a matching class name
      • <module_dir>.<ClassName>     → e.g. 'helpsteer.HelpSteer2'
    """
    from autometrics.dataset.Dataset import Dataset

    # Helper to import module safely
    def _import_module(dir_name: str):
        module_path = f"autometrics.dataset.datasets.{dir_name}.{dir_name}"
        return importlib.import_module(module_path)

    # Case 1: explicit module and class via dot
    if "." in identifier:
        module_part, class_part = identifier.split(".", 1)
        module = _import_module(module_part)
        if not hasattr(module, class_part):
            raise RuntimeError(f"Module '{module_part}' does not have class '{class_part}'.")
        cls = getattr(module, class_part)
        if not (isinstance(cls, type) and issubclass(cls, Dataset)):
            raise RuntimeError(f"{class_part} is not a Dataset subclass.")
        return cls()

    # Case 2: try identifier as module directory
    try:
        module = _import_module(identifier)
        # Find first Dataset subclass (or one matching identifier)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Dataset) and attr is not Dataset:
                # If attr_name matches identifier ignoring case use that, else first found
                if attr_name.lower() == identifier.lower():
                    return attr()
                candidate = attr
        else:
            # Fallback to candidate if set
            if 'candidate' in locals():
                return candidate()
    except ImportError:
        pass

    # Case 3: treat identifier as class name search across dataset modules
    datasets_dir = os.path.join(repo_root, "autometrics", "dataset", "datasets")
    for dir_name in os.listdir(datasets_dir):
        if dir_name.startswith("__") or not os.path.isdir(os.path.join(datasets_dir, dir_name)):
            continue
        try:
            module = _import_module(dir_name)
        except Exception:
            continue
        if hasattr(module, identifier):
            cls = getattr(module, identifier)
            if isinstance(cls, type) and issubclass(cls, Dataset):
                return cls()

    raise RuntimeError(f"Could not resolve dataset identifier '{identifier}'.")


def correlation_func_from_str(name: str):
    name = name.lower()
    if name.startswith("pearson"):
        from scipy.stats import pearsonr
        return pearsonr
    if name.startswith("spearman"):
        from scipy.stats import spearmanr
        return spearmanr
    if name.startswith("kendall") or name.startswith("tau"):
        from scipy.stats import kendalltau
        return kendalltau
    raise ValueError(f"Unknown correlation function '{name}'. Supported: pearson, spearman, kendall.")

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    suggestions = get_dataset_suggestions()
    parser = argparse.ArgumentParser(description="Benchmark correlation of metrics on a dataset."
                                   "\nAvailable datasets: " + ", ".join(suggestions))

    parser.add_argument("--dataset", default="summeval", help="Dataset identifier to evaluate (see list above or use --list-datasets).")
    parser.add_argument("--output-dir", default=None, help="Directory to write outputs. Defaults to outputs/correlation/<dataset>.")
    parser.add_argument("--correlation", default="pearson", help="Correlation(s) to use: comma-separated list of pearson, spearman, kendall OR 'all' for all three.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top metrics to visualize (default 5, 0 = all).")
    parser.add_argument("--skip-reference-based", action="store_true", help="Exclude reference-based metrics.")
    parser.add_argument("--skip-reference-free", action="store_true", help="Exclude reference-free metrics.")
    parser.add_argument("--metric", nargs="*", default=None, help="Specific metric names to include (overrides skip flags if provided).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    parser.add_argument("--list-datasets", action="store_true", help="List available dataset identifiers and exit.")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.list_datasets:
        print("Available datasets:\n" + "\n".join(get_dataset_suggestions()))
        return 0

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("benchmark_correlation")

    # We delay heavy imports until after we check if argparse has exited (which
    # it does automatically on `--help`).  At this point we know we're supposed
    # to execute the benchmarking logic.

    from autometrics.experiments.correlation.correlation import CorrelationExperiment
    from autometrics.metrics.MetricBank import reference_based_metrics, reference_free_metrics, all_metrics

    # Prepare base output directory (dataset-specific)
    base_output_dir = args.output_dir or os.path.join("outputs", "correlation", args.dataset)
    os.makedirs(base_output_dir, exist_ok=True)
    logger.info(f"Writing outputs to: {base_output_dir}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    # Select metrics ------------------------------------------------------------------
    if args.metric:
        # Filter all_metrics by supplied names (case-insensitive)
        selected = []
        requested = {m.lower() for m in args.metric}
        for metric in all_metrics:
            if metric.get_name().lower() in requested:
                selected.append(metric)
        missing = requested - {m.get_name().lower() for m in selected}
        if missing:
            logger.warning(f"Some requested metrics not found: {', '.join(missing)}")
        metrics = selected
    else:
        metrics = []

        # Auto-skip reference-based metrics if dataset lacks reference columns
        auto_skip_ref_based = False
        try:
            if hasattr(dataset, "get_reference_columns") and len(dataset.get_reference_columns()) == 0:
                auto_skip_ref_based = True
        except Exception:
            pass

        skip_reference_based = args.skip_reference_based or auto_skip_ref_based

        if auto_skip_ref_based and not args.skip_reference_based:
            logger.info("Dataset has no reference columns ‑ automatically skipping reference-based metrics.")

        if not skip_reference_based:
            metrics += reference_based_metrics
        if not args.skip_reference_free:
            metrics += reference_free_metrics

        if not metrics:
            logger.error("No metrics selected after applying skip flags / auto-skip policies.")
            return 1

    logger.info(f"Using {len(metrics)} metrics.")

    if args.correlation.lower() == "all":
        corr_names = ["pearson", "spearman", "kendall"]
    else:
        corr_names = [c.strip().lower() for c in args.correlation.split(",") if c.strip()]

    valid_set = {"pearson", "spearman", "kendall"}
    for c in corr_names:
        if c not in valid_set:
            logger.error(f"Unsupported correlation name '{c}'.")
            return 1

    logger.info(f"Running correlations: {', '.join(corr_names)}")

    corr_funcs = {name: correlation_func_from_str(name) for name in corr_names}

    experiment = CorrelationExperiment(
        name=f"Correlation Benchmark – {args.dataset}",
        description=f"Benchmarking correlations ({', '.join(corr_names)}) on {args.dataset}",
        metrics=metrics,
        output_dir=base_output_dir,
        dataset=dataset,
        correlation_funcs=corr_funcs,
        top_k=(None if args.top_k == 0 else args.top_k),
        seed=args.seed,
    )

    experiment.run(print_results=True)
    experiment.save_results()

    logger.info("Correlation benchmarking complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
