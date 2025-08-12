#!/usr/bin/env python3
"""
Simple and elegant experiments file for running autometrics experiments.

This script:
1. Checks if experiment has already completed (score_[seed].txt exists with valid float)
2. If not, loads dataset with persistent splits and runs autometrics pipeline
3. Evaluates regression metric on test set to get final score
4. Saves results to output directory

Usage:
    python run_main_autometrics.py <dataset_name> <target_name> <seed> <output_dir>
"""

import os
import sys
import json
import dspy
import numpy as np
import argparse
from typing import Optional, Dict

# Add autometrics to path
sys.path.append('/nlp/scr2/nlp/personal-rm/autometrics')

from autometrics.autometrics import Autometrics
from autometrics.dataset.Dataset import Dataset


def load_dataset(dataset_name: str) -> Dataset:
    """Load a dataset by name with persistent splits."""
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
    elif dataset_name == "Design2Code":
        from autometrics.dataset.datasets.design2code.design2code import Design2Code
        return Design2Code()
    elif dataset_name == "AI_Researcher":
        from autometrics.dataset.datasets.airesearcher.ai_researcher import AI_Researcher
        return AI_Researcher()
    
    raise ValueError(f"Unknown dataset: {dataset_name}")


def check_experiment_completed(output_dir: str, seed: int) -> Optional[Dict[str, float]]:
    """Check if experiment has already completed and return scores if so."""
    # Check for all correlation types
    correlation_types = ['pearson', 'spearman', 'kendall']
    score_files = [os.path.join(output_dir, f"score_{corr_type}_{seed}.txt") for corr_type in correlation_types]
    log_file = os.path.join(output_dir, f"log_{seed}.json")
    
    # Check if all files exist
    if not (all(os.path.exists(f) for f in score_files) and os.path.exists(log_file)):
        return None
    
    # Try to read all scores
    try:
        scores = {}
        for corr_type, score_file in zip(correlation_types, score_files):
            with open(score_file, 'r') as f:
                score_str = f.read().strip()
                scores[corr_type] = float(score_str)
        
        print(f"✅ Experiment already completed with scores:")
        for corr_type, score in scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
        return scores
    except (ValueError, IOError):
        print(f"⚠️  Score files exist but contain invalid data, re-running experiment")
        return None


def get_unique_directories(dataset_name: str, target_name: str, seed: int) -> tuple[str, str]:
    """Get unique cache and generated metrics directories for this experiment."""
    # Create unique identifiers
    experiment_id = f"{dataset_name}_{target_name}_{seed}"
    
    # Unique cache directory
    cache_dir = f"./autometrics_cache_{experiment_id}"
    
    # Unique generated metrics directory
    generated_metrics_dir = f"./generated_metrics_{experiment_id}"
    
    return cache_dir, generated_metrics_dir


def evaluate_regression_on_test(regression_metric, test_dataset: Dataset, target_measure: str, successful_metric_instances) -> Dict[str, float]:
    """Evaluate regression metric on test set and return correlation scores for all types."""
    # Simply use predict() which will handle all dependencies automatically
    print(f"📈 Evaluating regression metric on test set...")
    regression_metric.predict(test_dataset, update_dataset=True)
    
    # Use the existing calculate_correlation method from the codebase
    from autometrics.evaluate.correlation import calculate_correlation
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    correlations = {}
    
    # Calculate each correlation type using the existing method
    try:
        # Pearson correlation
        pearson_results = calculate_correlation(test_dataset, correlation=pearsonr)
        correlations['pearson'] = pearson_results[target_measure][regression_metric.get_name()]
        
        # Spearman correlation
        spearman_results = calculate_correlation(test_dataset, correlation=spearmanr)
        correlations['spearman'] = spearman_results[target_measure][regression_metric.get_name()]
        
        # Kendall correlation
        kendall_results = calculate_correlation(test_dataset, correlation=kendalltau)
        correlations['kendall'] = kendall_results[target_measure][regression_metric.get_name()]
        
    except Exception as e:
        print(f"⚠️ Warning: Error computing correlations: {e}")
        correlations = {'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0}
    
    return correlations


def run_autometrics_experiment(
    dataset_name: str,
    target_name: str,
    seed: int,
    output_dir: str,
    generator_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, float]:
    """Run a single autometrics experiment."""
    
    # Check if experiment already completed
    existing_scores = check_experiment_completed(output_dir, seed)
    if existing_scores is not None:
        return existing_scores
    
    print(f"🚀 Starting autometrics experiment:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Target: {target_name}")
    print(f"   Seed: {seed}")
    print(f"   Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique directories for this experiment
    cache_dir, generated_metrics_dir = get_unique_directories(dataset_name, target_name, seed)
    
    # Set environment variables for unique cache
    os.environ["AUTOMETRICS_CACHE_DIR"] = cache_dir
    
    try:
        # Load dataset with persistent splits
        print(f"📊 Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        train_dataset, val_dataset, test_dataset = dataset.load_permanent_splits()
        
        print(f"   Train: {len(train_dataset.get_dataframe())} examples")
        print(f"   Val: {len(val_dataset.get_dataframe())} examples")
        print(f"   Test: {len(test_dataset.get_dataframe())} examples")
        
        # Configure LLMs (CLI args take precedence; fall back to env vars; then defaults)
        print(f"🤖 Configuring LLMs...")
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base
        
        # Determine base model names
        generator_model_name_base = (
            generator_model_name
            or os.environ.get("AUTOMETRICS_LM_GENERATOR")
            or "openai/gpt-4o-mini"
        )
        judge_model_name_base = (
            judge_model_name
            or os.environ.get("AUTOMETRICS_LM_JUDGE")
            or generator_model_name_base
        )
        
        # Convert model names to proper litellm format if using local server
        def format_model_name(model_name: str) -> str:
            if os.environ.get("OPENAI_API_BASE") and "localhost" in os.environ.get("OPENAI_API_BASE", ""):
                # Local server - use litellm_proxy format
                if model_name.startswith("Qwen/"):
                    return f"litellm_proxy/{model_name}"
                elif "/" not in model_name and model_name.lower().startswith("qwen"):
                    return "litellm_proxy/Qwen/Qwen3-32B"
            return model_name
        
        generator_model_id = format_model_name(generator_model_name_base)
        judge_model_id = format_model_name(judge_model_name_base)
        
        print(f"   Generator LM: {generator_model_id}")
        print(f"   Judge LM: {judge_model_id}")
        
        # Create LLM instances with proper API key handling
        api_key = os.environ.get("OPENAI_API_KEY", "None")

        generator_llm = None
        judge_llm = None

        if "Qwen" in generator_model_id:
            generator_llm = dspy.LM(generator_model_id, api_key=api_key, max_tokens=8192)
        else:
            generator_llm = dspy.LM(generator_model_id, api_key=api_key)

        if "Qwen" in judge_model_id:
            judge_llm = dspy.LM(judge_model_id, api_key=api_key, max_tokens=8192)
        else:
            judge_llm = dspy.LM(judge_model_id, api_key=api_key)
        
        # Create autometrics with unique directories
        print(f"🔧 Creating autometrics pipeline...")
        autometrics = Autometrics(
            generated_metrics_dir=generated_metrics_dir,
            seed=seed
        )
        
        # Run autometrics pipeline on training data
        print(f"⚡ Running autometrics pipeline...")
        results = autometrics.run(
            dataset=train_dataset,
            target_measure=target_name,
            generator_llm=generator_llm,
            judge_llm=judge_llm
        )
        
        # Get regression metric
        regression_metric = results['regression_metric']
        if regression_metric is None:
            raise ValueError("No regression metric generated")
        
        # Evaluate on test set
        print(f"📈 Evaluating regression metric on test set...")
        test_scores = evaluate_regression_on_test(regression_metric, test_dataset, target_name, results['top_metrics'])
        
        print(f"✅ Test correlations:")
        for corr_type, score in test_scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
        
        # Save results
        print(f"💾 Saving results...")
        
        # Save individual score files for each correlation type
        for corr_type, score in test_scores.items():
            score_file = os.path.join(output_dir, f"score_{corr_type}_{seed}.txt")
            with open(score_file, 'w') as f:
                f.write(f"{score}")
        
        # Save detailed log as JSON
        log_file = os.path.join(output_dir, f"log_{seed}.json")
        log_data = {
            "dataset_name": dataset_name,
            "target_name": target_name,
            "seed": seed,
            "split_sizes": {
                "train": len(train_dataset.get_dataframe()),
                "val": len(val_dataset.get_dataframe()),
                "test": len(test_dataset.get_dataframe()),
            },
            "test_scores": test_scores,
            "report_card": results['report_card'],
            "top_metrics": [m.get_name() for m in results['top_metrics']],
            "importance_scores": [(float(score), name) for score, name in results['importance_scores'][:10]],
            "generated_metrics_count": len(results['all_generated_metrics']),
            "retrieved_metrics_count": len(results['retrieved_metrics']),
            "pipeline_config": results['pipeline_config']
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✅ Experiment completed successfully!")
        return test_scores
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        raise e


def main():
    """Main function to run autometrics experiment."""
    parser = argparse.ArgumentParser(description="Run Autometrics experiment")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    parser.add_argument("target_name", type=str, help="Target/measure name")
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--model-name", dest="model_name", type=str, default=None, help="LLM model name (e.g., openai/gpt-5-mini)")
    parser.add_argument("--api-base", dest="api_base", type=str, default=None, help="API base URL for OpenAI-compatible endpoints")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    try:
        scores = run_autometrics_experiment(
            dataset_name=args.dataset_name,
            target_name=args.target_name,
            seed=args.seed,
            output_dir=args.output_dir,
            generator_model_name=args.model_name,
            judge_model_name=args.model_name,
            api_base=args.api_base,
        )
        print(f"\n🎉 Final test correlations:")
        for corr_type, score in scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
    except Exception as e:
        print(f"\n💥 Experiment failed: {e}")
        raise e


if __name__ == "__main__":
    main()
