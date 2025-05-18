#!/usr/bin/env python3
"""
Benchmark Utilizer Script - Run utilization benchmarks for all metrics in the MetricBank

This script automatically runs utilization benchmarks for all metrics in the
MetricBank, saving results as it goes and allowing for interrupted runs to
be resumed without repeating work.
"""

import os
import sys
import time
import argparse
import logging
import importlib
import pandas as pd
import glob
from typing import List, Dict, Any, Optional

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from autometrics.metrics.MetricBank import reference_based_metrics, reference_free_metrics
from autometrics.experiments.utilization.utilization import UtilizationExperiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_utilizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default benchmark settings
DEFAULT_OUTPUT_DIR = "outputs/utilization"
DEFAULT_NUM_EXAMPLES = 50
DEFAULT_BURN_IN = 5
DEFAULT_LENGTHS = ["short", "medium", "long"]
DEFAULT_USE_SYNTHETIC = True

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run utilization benchmarks for all metrics in the MetricBank"
    )
    
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store outputs (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=DEFAULT_NUM_EXAMPLES,
        help=f"Number of examples to test per length category (default: {DEFAULT_NUM_EXAMPLES})"
    )
    
    parser.add_argument(
        "--burn-in", 
        type=int, 
        default=DEFAULT_BURN_IN,
        help=f"Number of burn-in samples to run (default: {DEFAULT_BURN_IN})"
    )
    
    parser.add_argument(
        "--lengths", 
        default=",".join(DEFAULT_LENGTHS),
        help=f"Comma-separated list of length categories (default: {','.join(DEFAULT_LENGTHS)})"
    )
    
    parser.add_argument(
        "--skip-reference-based",
        action="store_true",
        help="Skip reference-based metrics"
    )
    
    parser.add_argument(
        "--skip-reference-free",
        action="store_true",
        help="Skip reference-free metrics"
    )
    
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-run of already completed metrics"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def metric_has_complete_results(metric_name: str, output_dir: str, lengths: List[str]) -> bool:
    """
    Check if a metric has complete utilization results.
    
    Args:
        metric_name: Name of the metric
        output_dir: Base output directory
        lengths: List of length categories to check
        
    Returns:
        True if the metric has complete results, False otherwise
    """
    # For metrics used with synthetic data, we need to check if results exist for each length
    metric_dir = os.path.join(output_dir, "synthetic", metric_name)
    
    # Check for the summary.csv file in each length category directory
    for length in lengths:
        summary_path = os.path.join(metric_dir, length, "summary.csv")
        if not os.path.exists(summary_path):
            return False
    
    # Also check if raw_data.csv exists for each length (ensures complete run)
    for length in lengths:
        raw_data_path = os.path.join(metric_dir, length, "raw_data.csv")
        if not os.path.exists(raw_data_path):
            return False
            
    return True

def safely_import_metric(metric_class_path: str) -> Optional[Any]:
    """
    Safely import a metric class without crashing if dependencies are missing.
    
    Args:
        metric_class_path: Fully qualified path to the metric class (e.g., 'autometrics.metrics.reference_based.BLEU.BLEU')
        
    Returns:
        The metric instance or None if import failed
    """
    try:
        module_path, class_name = metric_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        MetricClass = getattr(module, class_name)
        return MetricClass()
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(f"Could not import {metric_class_path}: {str(e)}")
        return None

def get_metric_class_path(metric) -> str:
    """Get the fully qualified class path for a metric instance."""
    return f"{metric.__class__.__module__}.{metric.__class__.__name__}"

def aggregate_results(output_dir: str) -> pd.DataFrame:
    """
    Aggregate results from all metrics into a single DataFrame.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        DataFrame containing aggregated results
    """
    all_summaries = []
    
    # Find all summary.csv files in the output directory
    synthetic_dir = os.path.join(output_dir, "synthetic")
    pattern = os.path.join(synthetic_dir, "*", "*", "summary.csv")
    summary_files = glob.glob(pattern)
    
    for file_path in summary_files:
        # Extract metric name and length category from path
        # Path format: {output_dir}/synthetic/{metric_name}/{length}/summary.csv
        parts = file_path.split(os.sep)
        metric_name = parts[-3]
        length = parts[-2]
        
        try:
            df = pd.read_csv(file_path)
            # Add columns for metric name and length
            df['metric'] = metric_name
            df['length'] = length
            all_summaries.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {str(e)}")
    
    if not all_summaries:
        logger.warning("No summary files found to aggregate")
        return pd.DataFrame()
        
    # Combine all summaries
    combined_df = pd.concat(all_summaries, ignore_index=True)
    
    # Reorder columns to put metric and length first
    cols = ['metric', 'length'] + [col for col in combined_df.columns if col not in ['metric', 'length']]
    combined_df = combined_df[cols]
    
    return combined_df

def run_benchmark_for_metric(metric, args):
    """
    Run the utilization benchmark for a single metric.
    
    Args:
        metric: The metric instance to benchmark
        args: Command-line arguments
    """
    metric_name = metric.get_name()
    logger.info(f"Starting benchmark for {metric_name}")
    
    try:
        # Create the experiment
        experiment = UtilizationExperiment(
            name=f"{metric_name} Utilization Benchmark",
            description=f"Resource utilization benchmark for {metric_name}",
            metrics=[metric],
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            num_burn_in=args.burn_in,
            lengths=args.lengths.split(','),
            use_synthetic=DEFAULT_USE_SYNTHETIC,
            seed=args.seed,
            measure_import_costs=True,
            use_isolated_trials=True,
            use_deterministic_examples=True
        )
        
        # Run the experiment
        experiment.run(print_results=args.verbose)
        
        # Save the results
        experiment.save_results()
        logger.info(f"Benchmark completed for {metric_name}")
        return True
    except Exception as e:
        logger.error(f"Error benchmarking {metric_name}: {str(e)}", exc_info=True)
        return False

def main():
    """Main function to run the benchmarks."""
    args = parse_args()
    
    # Prepare the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log the start of the benchmarking process
    logger.info("Starting metric utilization benchmarking")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of examples: {args.num_examples}")
    logger.info(f"Burn-in samples: {args.burn_in}")
    logger.info(f"Length categories: {args.lengths}")
    
    # Collect metrics to benchmark
    metrics_to_benchmark = []
    
    if not args.skip_reference_based:
        metrics_to_benchmark.extend(reference_based_metrics)
        logger.info(f"Including {len(reference_based_metrics)} reference-based metrics")
    
    if not args.skip_reference_free:
        metrics_to_benchmark.extend(reference_free_metrics)
        logger.info(f"Including {len(reference_free_metrics)} reference-free metrics")
    
    logger.info(f"Total metrics to process: {len(metrics_to_benchmark)}")
    
    # Initialize counters
    total_metrics = len(metrics_to_benchmark)
    completed_metrics = 0
    skipped_metrics = 0
    failed_metrics = 0
    
    # Track metrics for the final report
    results_tracker = {
        "completed": [],
        "skipped": [],
        "failed": []
    }
    
    # Process each metric
    for i, metric in enumerate(metrics_to_benchmark, 1):
        metric_name = metric.get_name()
        logger.info(f"Processing metric {i}/{total_metrics}: {metric_name}")
        
        # Check if this metric already has complete results
        if not args.force_rerun and metric_has_complete_results(
            metric_name, args.output_dir, args.lengths.split(',')
        ):
            logger.info(f"Skipping {metric_name} - results already exist")
            skipped_metrics += 1
            results_tracker["skipped"].append(metric_name)
            continue
        
        # Run the benchmark
        success = run_benchmark_for_metric(metric, args)
        
        if success:
            completed_metrics += 1
            results_tracker["completed"].append(metric_name)
        else:
            failed_metrics += 1
            results_tracker["failed"].append(metric_name)
        
        # Log progress
        logger.info(f"Progress: {i}/{total_metrics} metrics processed")
        logger.info(f"Status: {completed_metrics} completed, {skipped_metrics} skipped, {failed_metrics} failed")
    
    # Aggregate results
    logger.info("Aggregating results from all metrics")
    combined_results = aggregate_results(args.output_dir)
    
    # Save the aggregated results
    if not combined_results.empty:
        aggregate_path = os.path.join(args.output_dir, "aggregated_results.csv")
        combined_results.to_csv(aggregate_path, index=False)
        logger.info(f"Aggregated results saved to {aggregate_path}")
    
    # Save the benchmark summary
    summary = {
        "total_metrics": total_metrics,
        "completed_metrics": completed_metrics,
        "skipped_metrics": skipped_metrics,
        "failed_metrics": failed_metrics,
        "completed_list": results_tracker["completed"],
        "skipped_list": results_tracker["skipped"],
        "failed_list": results_tracker["failed"]
    }
    
    summary_df = pd.DataFrame({
        "Category": ["Total Metrics", "Completed", "Skipped", "Failed"],
        "Count": [total_metrics, completed_metrics, skipped_metrics, failed_metrics]
    })
    
    summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed lists
    detailed_summary = pd.DataFrame({
        "Metric": (
            [f"COMPLETED: {m}" for m in results_tracker["completed"]] +
            [f"SKIPPED: {m}" for m in results_tracker["skipped"]] +
            [f"FAILED: {m}" for m in results_tracker["failed"]]
        )
    })
    detailed_summary_path = os.path.join(args.output_dir, "benchmark_detailed_summary.csv")
    detailed_summary.to_csv(detailed_summary_path, index=False)
    
    # Print final summary
    logger.info("=" * 50)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total metrics: {total_metrics}")
    logger.info(f"Completed metrics: {completed_metrics}")
    logger.info(f"Skipped metrics (already had results): {skipped_metrics}")
    logger.info(f"Failed metrics: {failed_metrics}")
    
    if failed_metrics > 0:
        logger.info("Failed metrics:")
        for metric in results_tracker["failed"]:
            logger.info(f"  - {metric}")
    
    logger.info(f"Aggregated results saved to {os.path.join(args.output_dir, 'aggregated_results.csv')}")
    logger.info(f"Benchmark summary saved to {os.path.join(args.output_dir, 'benchmark_summary.csv')}")
    
    return 0

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    logger.info(f"Total runtime: {duration_minutes:.2f} minutes")
    sys.exit(exit_code)