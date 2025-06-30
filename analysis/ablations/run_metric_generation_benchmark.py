#!/usr/bin/env python3
"""
Metric Generation Benchmark Experiment

This script benchmarks different metric generation approaches by generating metrics
using different generators and evaluating their correlation with human annotations.

The benchmark follows this methodology:
1. Uses persistent train sets for metric generation (training data for generators)
2. Uses persistent validation sets for evaluation (to measure correlation)
3. Generates different numbers of metrics per generator type:
   - LLMJudgeProposer: 10 metrics per trial (5 trials = 50 total)
   - RubricGenerator (Prometheus): 10 metrics per trial (5 trials = 50 total)
   - RubricGenerator (DSPy): 10 metrics per trial (5 trials = 50 total)
   - GEvalJudgeProposer: 10 metrics per trial (5 trials = 50 total)
   - CodeGenerator: 10 metrics per trial (5 trials = 50 total)
   - OptimizedJudgeProposer: 1 metric per trial (5 trials = 5 total)
   - FinetuneGenerator: 1 metric per trial (5 trials = 5 total)
   - LLMJudgeExampleProposer: 1 metric per trial (5 trials = 5 total)

For generators with 10 metrics per trial, correlation results are averaged together.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from pathlib import Path
import dspy
from collections import defaultdict

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.dataset.Dataset import Dataset
from autometrics.experiments.correlation.correlation import CorrelationExperiment, correlation_func_from_name

# Generators
from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer
from autometrics.generator.LLMJudgeExampleProposer import LLMJudgeExampleProposer
from autometrics.generator.OptimizedJudgeProposer import OptimizedJudgeProposer
from autometrics.generator.GEvalJudgeProposer import GEvalJudgeProposer
from autometrics.generator.CodeGenerator import CodeGenerator
from autometrics.generator.RubricGenerator import RubricGenerator
from autometrics.generator.FinetuneGenerator import FinetuneGenerator


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    
    # Suppress verbose logging from dependencies when not in verbose mode
    if not verbose:
        # DSPy can be very verbose
        logging.getLogger('dspy').setLevel(logging.WARNING)
        
        # Cache-related logging
        logging.getLogger('diskcache').setLevel(logging.WARNING)
        logging.getLogger('diskcache.core').setLevel(logging.WARNING)
        
        # HTTP and API related
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        # Model and ML related
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('tokenizers').setLevel(logging.WARNING)
        
        # Autometrics internals
        logging.getLogger('autometrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.metrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.metrics.Metric').setLevel(logging.WARNING)
        logging.getLogger('autometrics.experiments').setLevel(logging.WARNING)
        
        # General noise suppression
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('concurrent.futures').setLevel(logging.WARNING)
        
        # Cache key creation logging (various libraries might do this)
        logging.getLogger('cache').setLevel(logging.WARNING)
        logging.getLogger('caching').setLevel(logging.WARNING)
        
        # General third-party library noise
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('plotly').setLevel(logging.WARNING)
        
        # Set root logger to WARNING to catch anything else
        logging.getLogger().setLevel(logging.WARNING)
        
        # But keep our script's logger at INFO so we still see our progress messages
        logger.setLevel(logging.INFO)
        
        # Also make sure we see warnings and errors from our specific logger
        if logger.level > logging.WARNING:
            logger.setLevel(logging.WARNING)
    
    return logger


def load_dataset(dataset_name: str) -> Dataset:
    """Load a dataset by name."""
    # Import the specific dataset class
    if dataset_name == "Primock57":
        from autometrics.dataset.datasets.primock57.primock57 import Primock57
        return Primock57()
    elif dataset_name == "HelpSteer":
        from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer
        return HelpSteer()
    elif dataset_name == "HelpSteer2":
        from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer2
        return HelpSteer2()
    elif dataset_name == "SummEval":
        from autometrics.dataset.datasets.summeval.summeval import SummEval
        return SummEval()
    elif dataset_name == "SimpDA":
        from autometrics.dataset.datasets.simplification.simplification import SimpDA
        return SimpDA()
    elif dataset_name == "SimpEval":
        from autometrics.dataset.datasets.simplification.simplification import SimpEval
        return SimpEval()
    elif dataset_name.startswith("CoGym"):
        from autometrics.dataset.datasets.cogym.cogym import (
            CoGymTabularOutcome, CoGymTabularProcess, 
            CoGymTravelOutcome, CoGymTravelProcess, 
            CoGymLessonOutcome, CoGymLessonProcess
        )
        if dataset_name == "CoGymTabularOutcome":
            return CoGymTabularOutcome()
        elif dataset_name == "CoGymTabularProcess":
            return CoGymTabularProcess()
        elif dataset_name == "CoGymTravelOutcome":
            return CoGymTravelOutcome()
        elif dataset_name == "CoGymTravelProcess":
            return CoGymTravelProcess()
        elif dataset_name == "CoGymLessonOutcome":
            return CoGymLessonOutcome()
        elif dataset_name == "CoGymLessonProcess":
            return CoGymLessonProcess()
    elif dataset_name.startswith("EvalGen"):
        from autometrics.dataset.datasets.evalgen.evalgen import EvalGen
        if dataset_name == "EvalGenMedical":
            return EvalGen('./autometrics/dataset/datasets/evalgen/medical.csv')
        elif dataset_name == "EvalGenProduct":
            return EvalGen('./autometrics/dataset/datasets/evalgen/product.csv')
    elif dataset_name == "RealHumanEval":
        from autometrics.dataset.datasets.realhumaneval.realhumaneval import RealHumanEval
        return RealHumanEval()
    
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_available_datasets_measures() -> List[Tuple[str, str]]:
    """Get all available dataset-measure combinations for metric generation benchmark."""
    datasets_measures = []
    
    # Datasets with their target measures
    dataset_configs = {
        "HelpSteer": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
        "HelpSteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
        "SimpDA": ["fluency", "meaning", "simplicity"], 
        "SimpEval": ["score"],
        "SummEval": ["coherence", "consistency", "fluency", "relevance"],
        "Primock57": ["inc_plus_omi", "incorrect", "omissions", "time_sec"],
        "CoGymTabularOutcome": ["outcomeRating"],
        "CoGymTabularProcess": ["agentRating", "communicationRating"],
        "CoGymTravelOutcome": ["outcomeRating"],
        "CoGymTravelProcess": ["agentRating", "communicationRating"],
        "CoGymLessonOutcome": ["outcomeRating"],
        "CoGymLessonProcess": ["agentRating", "communicationRating"],
        "EvalGenMedical": ["grade"],
        "EvalGenProduct": ["grade"],
        "RealHumanEval": ["accepted"]
    }
    
    for dataset_name, measures in dataset_configs.items():
        for measure in measures:
            datasets_measures.append((dataset_name, measure))
    
    return datasets_measures


def create_llm_model(model_name: str, api_base: Optional[str] = None, seed: int = 42) -> dspy.LM:
    """Create LLM model instance based on model name with unique cache busting per seed."""
    
    temperature = 0.0001 * seed

    if model_name == "gpt4o_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY before running with gpt4o_mini.")
        model = dspy.LM("openai/gpt-4o-mini", api_key=api_key, temperature=temperature)
    
    elif model_name == "qwen3_32b":
        # Use provided api_base or default to localhost (for local server)
        base_url = api_base or "http://localhost:7410/v1"
        model = dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base=base_url, temperature=temperature, max_tokens=4096)
    
    elif model_name == "prometheus":
        # Use provided api_base or default to permanent prometheus server
        base_url = api_base or "http://future-hgx-1:7420/v1"
        model = dspy.LM("litellm_proxy/Unbabel/M-Prometheus-14B", api_base=base_url, temperature=temperature)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def create_generator(
    generator_type: str, 
    generator_llm: dspy.LM, 
    judge_llm: dspy.LM, 
    seed: int, 
    model_save_dir: Optional[str] = None
):
    """Create generator instance based on type with seed-specific configuration."""
    
    if generator_type == "llm_judge":
        return BasicLLMJudgeProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            seed=seed,
        )
    
    elif generator_type == "llm_judge_examples":
        return LLMJudgeExampleProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            seed=seed,
            max_optimization_samples=100  # Limit for faster processing
        )
    
    elif generator_type == "llm_judge_optimized":
        return OptimizedJudgeProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            auto_mode="medium",
            num_threads=16,
            eval_function_name='inverse_distance',
            seed=seed
        )
    
    elif generator_type == "geval":
        return GEvalJudgeProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            seed=seed,
        )
    
    elif generator_type == "codegen":
        return CodeGenerator(
            generator_llm=generator_llm,
            executor_kwargs={"seed": seed},
            seed=seed,
        )
    
    elif generator_type == "rubric_prometheus":
        return RubricGenerator(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            use_prometheus=True,
            seed=seed,
        )
    
    elif generator_type == "rubric_dspy":
        return RubricGenerator(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            use_prometheus=False,
            seed=seed,
        )
    
    elif generator_type == "finetune":
        return FinetuneGenerator(
            generator_llm=generator_llm,
            model_save_dir=model_save_dir,
            seed=seed
        )
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def save_generated_metrics(metrics: List, generator_type: str, dataset_name: str, measure: str, seed: int, output_dir: str):
    """Save generated metrics to organized directory structure."""
    
    # Create output directory structure
    generator_dir = Path(output_dir) / "generated_metrics" / generator_type / f"seed_{seed}" / dataset_name
    generator_dir.mkdir(parents=True, exist_ok=True)
    
    metric_paths = []
    
    for i, metric in enumerate(metrics):
        # Create clean filename
        safe_dataset_name = dataset_name.replace(" ", "_").replace("/", "_")
        safe_measure_name = measure.replace(" ", "_").replace("/", "_")
        metric_filename = f"{safe_dataset_name}_{safe_measure_name}_{generator_type}_seed{seed}_metric{i+1:02d}.py"
        
        metric_path = generator_dir / metric_filename
        
        # Save metric as standalone Python file
        metric.save_python_code(str(metric_path))
        metric_paths.append(str(metric_path))
    
    return metric_paths


def run_correlation_evaluation(
    metrics: List, 
    val_dataset: Dataset, 
    measure: str, 
    correlation_funcs: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, List[float]]:
    """Run correlation evaluation for a list of metrics on validation dataset using all correlation functions."""
    
    # Results dictionary: correlation_func_name -> list of correlations for each metric
    all_correlations = {}
    
    logger.debug(f"Starting correlation evaluation with {len(metrics)} metrics")
    logger.debug(f"Metrics: {[m.name if m else 'None' for m in metrics]}")
    
    # Filter out None metrics
    valid_metrics = [m for m in metrics if m is not None]
    if len(valid_metrics) != len(metrics):
        logger.warning(f"Filtered out {len(metrics) - len(valid_metrics)} None metrics")
    
    if not valid_metrics:
        logger.error("No valid metrics to evaluate")
        # Return empty results for all correlation functions
        return {corr_name: [] for corr_name in correlation_funcs}
    
    for i, metric in enumerate(valid_metrics):
        try:
            logger.debug(f"Processing metric {i+1}/{len(valid_metrics)}: {metric.name}")
            
            # Run correlation experiment for single metric with ALL correlation functions
            experiment = CorrelationExperiment(
                name=f"MetricGen Eval - {metric.name}",
                description=f"Evaluating generated metric: {metric.name}",
                metrics=[metric],
                output_dir=f"/tmp/metricgen_{hash(metric.name)}",
                dataset=val_dataset,
                correlation_funcs=correlation_funcs,
                should_split=False
            )
            
            logger.debug(f"Running correlation experiment for {metric.name}")
            results = experiment.run(print_results=False)
            logger.debug(f"Experiment results keys: {list(results.keys())}")
            
            # Extract correlations for each correlation function
            for corr_name, correlations_for_func in results.items():
                if corr_name not in all_correlations:
                    all_correlations[corr_name] = []
                
                logger.debug(f"Processing {corr_name} results for {metric.name}")
                logger.debug(f"Available measures in results: {list(correlations_for_func.keys())}")
                
                if measure not in correlations_for_func:
                    logger.error(f"Measure {measure} not found in {corr_name} results")
                    all_correlations[corr_name].append(np.nan)
                    continue
                
                df_corr = correlations_for_func[measure]
                logger.debug(f"Correlation DataFrame shape: {df_corr.shape}")
                logger.debug(f"Available metrics in DataFrame: {list(df_corr['Metric'].values)}")
                
                # Find the correlation for our specific metric
                metric_row = df_corr[df_corr['Metric'] == metric.name]
                
                if metric_row.empty:
                    logger.error(f"Metric {metric.name} not found in correlation results for {corr_name}")
                    all_correlations[corr_name].append(np.nan)
                else:
                    correlation = metric_row.iloc[0]['Correlation']
                    logger.debug(f"Found correlation for {metric.name} ({corr_name}): {correlation}")
                    all_correlations[corr_name].append(correlation)
                
        except Exception as e:
            logger.error(f"Error evaluating metric {metric.name}: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            # Add NaN for all correlation functions for this metric
            for corr_name in correlation_funcs:
                if corr_name not in all_correlations:
                    all_correlations[corr_name] = []
                all_correlations[corr_name].append(np.nan)
    
    logger.debug(f"Final correlation results: {all_correlations}")
    return all_correlations


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical measures for correlation values."""
    valid_values = [v for v in values if not pd.isna(v)]
    n = len(valid_values)
    
    if n == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_range': np.nan,
            'num_successful_runs': 0
        }
    
    mean_val = np.mean(valid_values)
    
    if n == 1:
        return {
            'mean': mean_val,
            'std': 0.0,
            'ci_lower': mean_val,
            'ci_upper': mean_val,
            'ci_range': 0.0,
            'num_successful_runs': n
        }
    
    std_val = np.std(valid_values, ddof=1)
    
    # 95% confidence interval using t-distribution
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_value * std_val / np.sqrt(n)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'ci_lower': mean_val - margin_error,
        'ci_upper': mean_val + margin_error,
        'ci_range': margin_error,
        'num_successful_runs': n
    }


def format_mean_ci(mean: float, ci_range: float) -> str:
    """Format mean ± CI for easy copying to papers."""
    if np.isnan(mean) or np.isnan(ci_range):
        return "N/A"
    return f"{mean:.4f} ± {ci_range:.4f}"


def save_results(results: List[Dict], output_file: str, logger: logging.Logger):
    """Save results to CSV file with properly sorted columns."""
    try:
        if not results:
            logger.warning(f"No results to save to {output_file}")
            return
            
        df = pd.DataFrame(results)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        raise


def load_existing_results(output_file: str) -> List[Dict]:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            return df.to_dict('records')
        except Exception as e:
            logging.warning(f"Could not read existing results file {output_file}: {e}")
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark metric generation approaches across multiple seeds and datasets"
    )
    
    parser.add_argument(
        "--generator-model",
        default="gpt4o_mini",
        choices=["gpt4o_mini", "qwen3_32b"],
        help="LLM model to use for metric generation"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt4o_mini",
        choices=["gpt4o_mini", "qwen3_32b", "prometheus"],
        help="LLM model to use for judging/evaluation"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for the model (e.g., http://localhost:7410/v1)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablations/metric_generation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model-save-dir",
        default="/sphinx/u/salt-checkpoints/autometrics/models",
        help="Directory to save fine-tuned models"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to test"
    )
    parser.add_argument(
        "--correlation",
        default="all",
        help="Correlation function(s): 'pearson', 'spearman', 'kendall', or 'all' for all three"
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        help="Filter to specific datasets (e.g., HelpSteer SimpEval)"
    )
    parser.add_argument(
        "--measure", 
        nargs="*",
        help="Filter to specific measures (e.g., helpfulness fluency)"
    )
    parser.add_argument(
        "--generator",
        nargs="*",
        choices=["llm_judge", "llm_judge_examples", "llm_judge_optimized", "geval", 
                "codegen", "rubric_prometheus", "rubric_dspy", "finetune"],
        help="Filter to specific generator types"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("Starting Metric Generation Benchmark")
    logger.info(f"Generator Model: {args.generator_model}")
    logger.info(f"Judge Model: {args.judge_model}")
    logger.info(f"API Base: {args.api_base}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Correlation: {args.correlation}")

    output_dir = f"{args.output_dir}/{args.generator_model}"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/sub_results", exist_ok=True)
    
    # Get correlation functions
    if args.correlation.lower() == "all":
        correlation_specs = ["kendall", "pearson", "spearman"]
    else:
        correlation_specs = [c.strip() for c in args.correlation.split(",")]
    
    correlation_funcs = {}
    for spec in correlation_specs:
        correlation_funcs[spec] = correlation_func_from_name(spec)
    
    print(f"Using correlation functions: {list(correlation_funcs.keys())}")
    
    # Get all available dataset-measure combinations
    all_dataset_measures = get_available_datasets_measures()
    
    # Filter by dataset if specified
    if args.dataset:
        allowed_datasets = set(args.dataset)
        all_dataset_measures = [(d, m) for d, m in all_dataset_measures if d in allowed_datasets]
        logger.info(f"Filtered to datasets: {args.dataset}")
    
    # Filter by measure if specified
    if args.measure:
        allowed_measures = set(args.measure)
        all_dataset_measures = [(d, m) for d, m in all_dataset_measures if m in allowed_measures]
        logger.info(f"Filtered to measures: {args.measure}")
    
    if not all_dataset_measures:
        logger.error("No dataset-measure combinations to process after filtering")
        return 1
    
    logger.info(f"Processing {len(all_dataset_measures)} dataset-measure combinations")
    
    # Define generator configurations
    generator_configs = {
        "llm_judge": {"metrics_per_trial": 10, "description": "Basic LLM Judge"},
        "rubric_prometheus": {"metrics_per_trial": 10, "description": "Rubric Generator (Prometheus)"},
        "rubric_dspy": {"metrics_per_trial": 10, "description": "Rubric Generator (DSPy)"},
        "geval": {"metrics_per_trial": 10, "description": "G-Eval"},
        "codegen": {"metrics_per_trial": 10, "description": "Code Generation"},
        "llm_judge_optimized": {"metrics_per_trial": 1, "description": "LLM Judge (MIPROv2-Optimized)"},
        "finetune": {"metrics_per_trial": 1, "description": "Fine-tuned ModernBERT"},
        "llm_judge_examples": {"metrics_per_trial": 1, "description": "LLM Judge (Example-Based)"},
    }
    
    # Filter generators if specified
    if args.generator:
        generator_configs = {k: v for k, v in generator_configs.items() if k in args.generator}
        logger.info(f"Filtered to generators: {args.generator}")
    
    # Process each correlation type
    for correlation_type in correlation_funcs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {correlation_type.upper()} correlation analysis")
        logger.info(f"{'='*60}")
        
        # Get correlation function
        correlation_func = correlation_funcs[correlation_type]
        single_correlation_funcs = {correlation_type: correlation_func}
        
        # All results across datasets for final merging
        all_results_for_correlation = []
        dataset_output_files = []
        
        # Load existing merged results for this correlation function
        output_base = f"{output_dir}/metric_generation_benchmark_{args.generator_model}_{args.judge_model}"
        merged_output_file = f"{output_base}_{correlation_type}.csv"
        existing_merged_results = load_existing_results(merged_output_file)
        logger.info(f"Loaded {len(existing_merged_results)} existing {correlation_type} results from {merged_output_file}")

        # Group dataset-measure combinations for dataset-specific files
        dataset_measures_by_dataset = {}
        for dataset_name, measure in all_dataset_measures:
            if dataset_name not in dataset_measures_by_dataset:
                dataset_measures_by_dataset[dataset_name] = []
            dataset_measures_by_dataset[dataset_name].append((dataset_name, measure))

        # Process each dataset separately
        for dataset_name, dataset_measures in dataset_measures_by_dataset.items():
            logger.info(f"\n--- Processing dataset: {dataset_name} ---")
            
            # Dataset-specific output file in sub-results directory
            dataset_output_file = f"{output_dir}/sub_results/metric_generation_benchmark_{args.generator_model}_{args.judge_model}_{correlation_type}_{dataset_name}.csv"
            dataset_output_files.append(dataset_output_file)
            
            logger.info(f"Dataset output file: {dataset_output_file}")
            
            # Load existing dataset-specific results
            existing_dataset_results = load_existing_results(dataset_output_file)
            dataset_results = existing_dataset_results.copy()
            logger.info(f"Loaded {len(existing_dataset_results)} existing dataset results from {dataset_output_file}")

            # Process each dataset-measure combination for this dataset
            for dataset_name, measure in dataset_measures:
                logger.info(f"\nProcessing: {dataset_name} - {measure}")
                
                try:
                    # Load dataset
                    dataset = load_dataset(dataset_name)
                    task_description = dataset.get_task_description()
                    
                    logger.info(f"Dataset loaded: {dataset_name}")
                    logger.info(f"Task: {task_description}")
                    logger.info(f"Target measure: {measure}")
                    
                    # Process each generator type
                    for generator_type, config in generator_configs.items():
                        logger.info(f"\n--- Generator: {config['description']} ---")
                        
                        metrics_per_trial = config["metrics_per_trial"]
                        
                        # Check existing results for this dataset-measure-generator combination
                        existing_result = None
                        for result in dataset_results:
                            if (result.get('dataset') == dataset_name and 
                                result.get('measure') == measure and 
                                result.get('generator_type') == generator_type):
                                existing_result = result
                                break
                        
                        # Collect seed results
                        seed_correlations = []
                        seed_errors = []
                        
                        for seed in args.seeds:
                            # Check if we already have this seed's result
                            seed_key = f'seed_{seed}_avg_correlation'
                            
                            if existing_result and seed_key in existing_result and not pd.isna(existing_result[seed_key]):
                                seed_correlations.append(existing_result[seed_key])
                                logger.info(f"  Seed {seed}: Using cached result")
                                continue
                            
                            logger.info(f"  Running seed {seed}...")
                            
                            try:
                                # Create models with seed-specific temperature
                                generator_llm = create_llm_model(args.generator_model, args.api_base, seed)
                                judge_llm = create_llm_model(args.judge_model, args.api_base, seed)
                                
                                # Create generator
                                generator = create_generator(
                                    generator_type, 
                                    generator_llm, 
                                    judge_llm, 
                                    seed,
                                    args.model_save_dir
                                )
                                
                                # Get persistent splits - use TRAIN for generation, VAL for evaluation
                                train_dataset, val_dataset, _ = dataset.load_permanent_splits()
                                logger.info(f"    Using persistent splits - Train: {len(train_dataset.get_dataframe())} examples, Val: {len(val_dataset.get_dataframe())} examples")
                                
                                # Generate metrics using TRAINING data
                                logger.info(f"    Generating {config['metrics_per_trial']} metrics using training split...")
                                try:
                                    logger.debug(f"    Creating generator: {generator_type}")
                                    logger.debug(f"    Generator config: {config}")
                                    logger.debug(f"    Seed being passed: {seed}")
                                    
                                    metrics = generator.generate(
                                        dataset=train_dataset, 
                                        target_measure=measure, 
                                        n_metrics=config['metrics_per_trial']
                                    )
                                    logger.debug(f"    Generated {len(metrics) if metrics else 0} metrics: {[m.name if m else 'None' for m in (metrics or [])]}")
                                    
                                    # Check for None metrics or empty list
                                    if not metrics:
                                        logger.error(f"    No metrics generated for seed {seed}")
                                        raise ValueError("Generator returned empty metrics list")
                                        
                                    none_metrics = [i for i, m in enumerate(metrics) if m is None]
                                    if none_metrics:
                                        logger.error(f"    Found None metrics at indices: {none_metrics}")
                                        raise ValueError(f"Generator produced None metrics at indices: {none_metrics}")
                                        
                                except Exception as e:
                                    logger.error(f"    Metric generation failed: {str(e)}")
                                    logger.debug(f"    Full traceback: {traceback.format_exc()}")
                                    # Record this as a seed error and continue
                                    error_msg = f"Seed {seed}: {str(e)}"
                                    seed_errors.append(error_msg)
                                    seed_correlations.append(np.nan)
                                    continue

                                # Evaluate correlations using VALIDATION data
                                logger.info(f"    Evaluating correlations on validation split...")
                                try:
                                    logger.debug(f"    Metrics for correlation: {[m.name if m else 'None' for m in metrics]}")
                                    logger.debug(f"    Validation dataset size: {len(val_dataset.get_dataframe())}")
                                    logger.debug(f"    Target measure: {measure}")
                                    logger.debug(f"    Correlation function: {correlation_type}")
                                    
                                    all_correlations = run_correlation_evaluation(
                                        metrics, val_dataset, measure, single_correlation_funcs, logger
                                    )
                                    logger.debug(f"    Correlation results: {all_correlations}")
                                    
                                    # Save generated metrics
                                    metric_paths = save_generated_metrics(
                                        metrics, generator_type, dataset_name, measure, seed, output_dir
                                    )
                                    logger.info(f"    Saved metrics to: {len(metric_paths)} files")
                                    for path in metric_paths:
                                        logger.info(f"    {path}")
                                    
                                except Exception as e:
                                    logger.error(f"    Correlation evaluation failed: {str(e)}")
                                    logger.debug(f"    Full traceback: {traceback.format_exc()}")
                                    # Record this as a seed error and continue
                                    error_msg = f"Seed {seed}: Correlation evaluation failed: {str(e)}"
                                    seed_errors.append(error_msg)
                                    seed_correlations.append(np.nan)
                                    continue
                                
                                # Process results for this correlation function
                                correlations = all_correlations[correlation_type]
                                
                                # For generators with multiple metrics, average the correlations
                                if metrics_per_trial > 1:
                                    valid_correlations = [c for c in correlations if not pd.isna(c)]
                                    if valid_correlations:
                                        avg_correlation = np.mean([abs(c) for c in valid_correlations])
                                        logger.info(f"    Average {correlation_type} correlation: {avg_correlation:.4f} (from {len(valid_correlations)}/{len(correlations)} valid)")
                                    else:
                                        avg_correlation = np.nan
                                        logger.warning(f"    No valid {correlation_type} correlations from {len(correlations)} metrics")
                                else:
                                    avg_correlation = abs(correlations[0]) if correlations and not pd.isna(correlations[0]) else np.nan
                                    logger.info(f"    {correlation_type} correlation: {avg_correlation:.4f}")
                                
                                seed_correlations.append(avg_correlation)
                                
                            except Exception as e:
                                error_msg = f"Seed {seed}: {str(e)}"
                                seed_errors.append(error_msg)
                                logger.error(f"    Error: {error_msg}")
                                seed_correlations.append(np.nan)
                        
                        # Compute statistics
                        stats_result = compute_statistics(seed_correlations)
                        
                        # Create result record
                        result = {
                            'dataset': dataset_name,
                            'measure': measure,
                            'generator_type': generator_type,
                            'generator_description': config['description'],
                            'metrics_per_trial': metrics_per_trial,
                            'num_successful_runs': stats_result['num_successful_runs'],
                            'errors': '; '.join(seed_errors) if seed_errors else ''
                        }
                        
                        # Add individual seed results
                        for i, seed in enumerate(args.seeds):
                            result[f'seed_{seed}_avg_correlation'] = seed_correlations[i] if i < len(seed_correlations) else np.nan
                        
                        # Add statistics
                        result.update({
                            'mean_correlation': stats_result['mean'],
                            'std_correlation': stats_result['std'],
                            'ci_lower_correlation': stats_result['ci_lower'],
                            'ci_upper_correlation': stats_result['ci_upper'],
                            'mean_±_ci': format_mean_ci(stats_result['mean'], stats_result['ci_range'])
                        })
                        
                        # Update existing results or add new result
                        if existing_result:
                            # Update existing result in dataset_results
                            for i, res in enumerate(dataset_results):
                                if (res.get('dataset') == dataset_name and 
                                    res.get('measure') == measure and 
                                    res.get('generator_type') == generator_type):
                                    dataset_results[i] = result
                                    break
                            logger.info(f"Updated existing result for {generator_type}")
                        else:
                            dataset_results.append(result)
                            logger.info(f"Added new result for {generator_type}")
                        
                        # Save results immediately after each generator completes
                        try:
                            save_results(dataset_results, dataset_output_file, logger)
                            logger.info(f"Saved intermediate results to {dataset_output_file}")
                        except Exception as e:
                            logger.warning(f"Failed to save intermediate results: {e}")
                        
                        logger.info(f"Generator {generator_type} completed: mean={stats_result['mean']:.4f}, "
                                  f"CI=[{stats_result['ci_lower']:.4f}, {stats_result['ci_upper']:.4f}]")
                
                except Exception as e:
                    logger.error(f"Error processing {dataset_name}-{measure}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    continue
            
            # Save dataset-specific results
            if dataset_results:
                save_results(dataset_results, dataset_output_file, logger)
                all_results_for_correlation.extend(dataset_results)  # Add to combined results
                logger.info(f"Dataset {dataset_name} results saved to {dataset_output_file}")
            else:
                logger.warning(f"No results generated for dataset {dataset_name}")
        
        # Save merged results for this correlation function
        if all_results_for_correlation:
            # Merge with existing results
            merged_results = existing_merged_results.copy()
            for result in all_results_for_correlation:
                # Check if this result already exists in merged_results
                found = False
                for i, existing in enumerate(merged_results):
                    if (existing.get('dataset') == result['dataset'] and 
                        existing.get('measure') == result['measure'] and 
                        existing.get('generator_type') == result['generator_type']):
                        merged_results[i] = result  # Update existing
                        found = True
                        break
                if not found:
                    merged_results.append(result)  # Add new
            
            save_results(merged_results, merged_output_file, logger)
            
            # Print summary for this correlation type
            logger.info(f"\nSummary of Metric Generation {correlation_type.upper()} Results:")
            logger.info(f"Generator Model: {args.generator_model}")
            logger.info(f"Judge Model: {args.judge_model}")
            logger.info(f"Seeds: {args.seeds}")
            logger.info(f"Merged results saved to: {merged_output_file}")
            logger.info(f"Dataset-specific results saved to:")
            for dataset_file in dataset_output_files:
                logger.info(f"  {dataset_file}")
            
            # Show top performers
            logger.info(f"\nTop 5 performing generator-dataset combinations ({correlation_type}):")
            df_results = pd.DataFrame(merged_results)
            if not df_results.empty and 'mean_correlation' in df_results.columns:
                df_top = df_results.nlargest(5, 'mean_correlation')
                
                for _, row in df_top.iterrows():
                    logger.info(f"  {row['generator_description']} on {row['dataset']}.{row['measure']}: "
                               f"{row['mean_correlation']:.4f} ± {row['ci_upper_correlation'] - row['ci_lower_correlation']:.4f}")
        
        else:
            logger.error(f"No results generated for {correlation_type}")

    logger.info("Metric Generation Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 