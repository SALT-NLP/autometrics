#!/usr/bin/env python3
"""
Run a single Autometrics ablation experiment.

Differences from main experiments:
- Trains on train split, evaluates final correlations on the persistent validation split (NOT test).
- Supports ablations via hyperparameters:
  - Metric bank mode: full | existing_only | generated_only
  - Retrieval amount k (num_to_retrieve)
  - Regression count n (num_to_regress)
  - No metric cards (retrieval uses description-only docs instead of full docstrings)
  - Force reindexing of retrievers (to avoid stale/cached indices), recommended for special ablations

Each run writes three score files (pearson/spearman/kendall) and a JSON log to the provided output directory.
Cache and generated metrics directories are made unique per run to avoid contamination across ablations.

Usage:
    python analysis/ablations/run_autometrics_ablation.py \
        <dataset_name> <target_name> <seed> <output_dir> \
        [--model-name MODEL] [--api-base API_BASE] \
        [--metricbank {full,existing_only,generated_only}] \
        [--k NUM_TO_RETRIEVE] [--n NUM_TO_REGRESS] \
        [--no-metric-cards] [--force-reindex] [--resized]
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict

# Add project root to path
sys.path.append('/nlp/scr2/nlp/personal-rm/autometrics')

import dspy
from autometrics.autometrics import Autometrics
from autometrics.dataset.Dataset import Dataset


def load_dataset(dataset_name: str) -> Dataset:
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
        # Use specific subclasses to preserve task descriptions and clear naming
        from autometrics.dataset.datasets.evalgen.evalgen import EvalGenProduct, EvalGenMedical
        if dataset_name == "EvalGenMedical":
            return EvalGenMedical()
        elif dataset_name == "EvalGenProduct":
            return EvalGenProduct()
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
    correlation_types = ['pearson', 'spearman', 'kendall']
    score_files = [os.path.join(output_dir, f"score_{corr_type}_{seed}.txt") for corr_type in correlation_types]
    log_file = os.path.join(output_dir, f"log_{seed}.json")
    if not (all(os.path.exists(f) for f in score_files) and os.path.exists(log_file)):
        return None
    try:
        scores = {}
        for corr_type, score_file in zip(correlation_types, score_files):
            with open(score_file, 'r') as f:
                scores[corr_type] = float(f.read().strip())
        print("✅ Experiment already completed with scores:")
        for corr_type, score in scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
        return scores
    except Exception:
        print("⚠️  Score files exist but are invalid; re-running experiment")
        return None


def get_unique_dirs(dataset_name: str, target_name: str, seed: int, k: Optional[int], n: Optional[int], metricbank_mode: str, no_metric_cards: bool, resized: bool = False) -> tuple[str, str]:
    ablation_bits = []
    if k is not None:
        ablation_bits.append(f"k{k}")
    if n is not None:
        ablation_bits.append(f"n{n}")
    if metricbank_mode != 'full':
        ablation_bits.append(metricbank_mode)
    if no_metric_cards:
        ablation_bits.append("desc")
    ablation_tag = "_".join(ablation_bits) if ablation_bits else "default"
    run_id = f"{dataset_name}_{target_name}_seed{seed}_{ablation_tag}"
    cache_dir = f"./autometrics_cache_ablations/autometrics_cache_{run_id}"
    gen_dir = f"./generated_metrics_ablations/generated_metrics_{run_id}{'_resized' if resized else ''}"
    return cache_dir, gen_dir


def evaluate_on_validation(regression_metric, val_dataset: Dataset, target_measure: str):
    from scipy.stats import pearsonr, spearmanr, kendalltau
    print("📈 Evaluating regression metric on validation set…")
    # Ensure constituent top metrics may be present; regression will compute its own output
    regression_metric.predict(val_dataset, update_dataset=True)
    # Direct correlation: only target vs regression metric
    try:
        df = val_dataset.get_dataframe()
        metric_name = regression_metric.get_name()
        total_rows = len(df)
        if target_measure not in df.columns or metric_name not in df.columns:
            print(f"⚠️ Correlation debug (val): missing columns. target_present={target_measure in df.columns}, metric_present={metric_name in df.columns}")
            return ({'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0},
                    {'pearson': None, 'spearman': None, 'kendall': None})
        pair_df = df[[target_measure, metric_name]].dropna()
        print(f"🔎 Correlation debug (val): rows={total_rows}, valid_pairs={len(pair_df)} for metric='{metric_name}' vs target='{target_measure}'")
        if len(pair_df) < 2:
            print("⚠️ Not enough valid pairs (<2) on val. Returning zeros.")
            return ({'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0},
                    {'pearson': None, 'spearman': None, 'kendall': None})
        pr, pp = pearsonr(pair_df[target_measure], pair_df[metric_name])
        sr, sp = spearmanr(pair_df[target_measure], pair_df[metric_name])
        kr, kp = kendalltau(pair_df[target_measure], pair_df[metric_name])
        scores = {'pearson': float(pr), 'spearman': float(sr), 'kendall': float(kr)}
        p_values = {'pearson': float(pp), 'spearman': float(sp), 'kendall': float(kp)}
        return scores, p_values
    except Exception as e:
        print(f"⚠️ Warning: Error computing direct correlations (val): {e}")
        return ({'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0},
                {'pearson': None, 'spearman': None, 'kendall': None})


def run_ablation(
    dataset_name: str,
    target_name: str,
    seed: int,
    output_dir: str,
    generator_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    metricbank_mode: str = 'full',
    k: Optional[int] = None,
    n: Optional[int] = None,
    no_metric_cards: bool = False,
    force_reindex: bool = False,
    resized: bool = False,
) -> Dict[str, float]:
    existing = check_experiment_completed(output_dir, seed)
    if existing is not None:
        return existing

    print("🚀 Starting Autometrics ablation:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Target: {target_name}")
    print(f"   Seed: {seed}")
    print(f"   Output: {output_dir}")
    print(f"   MetricBank: {metricbank_mode}")
    print(f"   k (retrieve): {k if k is not None else 'default'}")
    print(f"   n (regress): {n if n is not None else 'default'}")
    print(f"   No metric cards: {no_metric_cards}")
    print(f"   Force reindex: {force_reindex}")

    os.makedirs(output_dir, exist_ok=True)

    cache_dir, generated_metrics_dir = get_unique_dirs(dataset_name, target_name, seed, k, n, metricbank_mode, no_metric_cards, resized)
    os.environ["AUTOMETRICS_CACHE_DIR"] = cache_dir

    print("📊 Loading dataset…")
    dataset = load_dataset(dataset_name)
    train_dataset, val_dataset, test_dataset = dataset.load_permanent_splits(resized=resized)
    print(f"   Train: {len(train_dataset.get_dataframe())}")
    print(f"   Val:   {len(val_dataset.get_dataframe())}")
    print(f"   Test:  {len(test_dataset.get_dataframe())}")

    # LLM config
    print("🤖 Configuring LLMs…")
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    api_key = os.environ.get("OPENAI_API_KEY", "None")

    def format_model(model_name: Optional[str]) -> str:
        base = model_name or os.environ.get("AUTOMETRICS_LM_GENERATOR") or "openai/gpt-4o-mini"
        api_base_env = os.environ.get("OPENAI_API_BASE", "")
        # If we're pointing at any OpenAI-compatible endpoint (local or remote), use litellm_proxy for non-OpenAI providers like Qwen
        if api_base_env:
            if not base.startswith("litellm_proxy/"):
                if base.startswith("Qwen/"):
                    return f"litellm_proxy/{base}"
                elif "/" not in base and base.lower().startswith("qwen"):
                    return "litellm_proxy/Qwen/Qwen3-32B"
        return base

    generator_model_id = format_model(generator_model_name)
    judge_model_id = format_model(judge_model_name or generator_model_name)
    print(f"   Generator LM: {generator_model_id}")
    print(f"   Judge LM: {judge_model_id}")

    if "Qwen" in generator_model_id:
        generator_llm = dspy.LM(generator_model_id, api_key=api_key, max_tokens=4096)
    else:
        generator_llm = dspy.LM(generator_model_id, api_key=api_key)
    if "Qwen" in judge_model_id:
        judge_llm = dspy.LM(judge_model_id, api_key=api_key, max_tokens=4096)
    else:
        judge_llm = dspy.LM(judge_model_id, api_key=api_key)

    # Configure Autometrics per ablation
    print("🔧 Creating Autometrics pipeline…")

    # Metric bank modes
    metric_generation_configs = None
    metric_bank = None
    merge_generated_with_bank = False

    if metricbank_mode == 'full':
        # Use library defaults by omitting metric_generation_configs and metric_bank entirely.
        # Do NOT set them to None explicitly when constructing Autometrics.
        metric_generation_configs = 'USE_DEFAULTS'
        metric_bank = 'USE_DEFAULTS'
    elif metricbank_mode == 'existing_only':
        # Do not generate; only use existing metric bank
        metric_generation_configs = {}
        metric_bank = None
    elif metricbank_mode == 'generated_only':
        # Generate metrics only; start from empty bank
        from autometrics.autometrics import DEFAULT_GENERATOR_CONFIGS
        metric_generation_configs = DEFAULT_GENERATOR_CONFIGS
        metric_bank = []
    else:
        raise ValueError("metricbank_mode must be one of: full, existing_only, generated_only")

    # Retriever kwargs - start from defaults and augment
    from autometrics.autometrics import DEFAULT_RETRIEVER_KWARGS
    retriever_kwargs = DEFAULT_RETRIEVER_KWARGS.copy()
    # Add description-only and reindex flags; these are respected by validators and constructors
    retriever_kwargs['use_description_only'] = bool(no_metric_cards)
    retriever_kwargs['force_reindex'] = bool(force_reindex)

    # Construct Autometrics kwargs minimally to avoid passing None to generation configs
    autometrics_kwargs = dict(
        generated_metrics_dir=generated_metrics_dir,
        merge_generated_with_bank=merge_generated_with_bank,
        seed=seed,
        retriever_kwargs=retriever_kwargs,
    )
    if metricbank_mode == 'existing_only':
        autometrics_kwargs['metric_generation_configs'] = {}
    elif metricbank_mode == 'generated_only':
        autometrics_kwargs['metric_generation_configs'] = metric_generation_configs  # from defaults
        autometrics_kwargs['metric_bank'] = []
    # For 'full', we pass nothing so Autometrics uses its own defaults

    autometrics = Autometrics(**autometrics_kwargs)

    print("⚡ Running pipeline on training set…")
    results = autometrics.run(
        dataset=train_dataset,
        target_measure=target_name,
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        num_to_retrieve=(k if k is not None else 30),
        num_to_regress=(n if n is not None else 5),
        # Provide validation set for report card sections and set HTML output path
        eval_dataset=val_dataset,
        report_output_path=os.path.join(output_dir, f"report_{dataset_name}_{target_name}_{seed}.html"),
        verbose=True,
    )

    regression_metric = results['regression_metric']
    if regression_metric is None:
        raise ValueError("No regression metric generated")

    print("📈 Evaluating on validation split…")
    val_scores, val_p_values = evaluate_on_validation(regression_metric, val_dataset, target_name)

    print("✅ Validation correlations:")
    for corr_type, score in val_scores.items():
        print(f"   {corr_type.capitalize()}: {score:.4f}")

    print("💾 Saving results…")
    for corr_type, score in val_scores.items():
        score_file = os.path.join(output_dir, f"score_{corr_type}_{seed}.txt")
        with open(score_file, 'w') as f:
            f.write(f"{score}")

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
        "val_scores": val_scores,
        "val_p_values": val_p_values,
        "report_card": results['report_card'],
        "top_metrics": [m.get_name() for m in results['top_metrics']],
        "importance_scores": [(float(score), name) for score, name in results['importance_scores'][:10]],
        "generated_metrics_count": len(results['all_generated_metrics']),
        "retrieved_metrics_count": len(results['retrieved_metrics']),
        "pipeline_config": results['pipeline_config'],
        "ablation_config": {
            "metricbank_mode": metricbank_mode,
            "k": k,
            "n": n,
            "no_metric_cards": no_metric_cards,
            "force_reindex": force_reindex,
        }
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    print("✅ Ablation run completed successfully!")
    return val_scores


def main():
    parser = argparse.ArgumentParser(description="Run a single Autometrics ablation experiment (evaluates on validation)")
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("target_name", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--model-name", dest="model_name", type=str, default=None)
    parser.add_argument("--api-base", dest="api_base", type=str, default=None)

    parser.add_argument("--metricbank", type=str, choices=["full", "existing_only", "generated_only"], default="full")
    parser.add_argument("--k", dest="k", type=int, default=None)
    parser.add_argument("--n", dest="n", type=int, default=None)
    parser.add_argument("--no-metric-cards", action="store_true", help="Use description-only documents for retrieval and reranking (separate indices)")
    parser.add_argument("--force-reindex", action="store_true", help="Force retriever reindex (avoid cached indices)")
    parser.add_argument("--resized", action="store_true", help="Use resized dataset (for train and val splits of EvalGenProduct and CoGymTravelOutcome)")

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    try:
        scores = run_ablation(
            dataset_name=args.dataset_name,
            target_name=args.target_name,
            seed=args.seed,
            output_dir=args.output_dir,
            generator_model_name=args.model_name,
            judge_model_name=args.model_name,
            api_base=args.api_base,
            metricbank_mode=args.metricbank,
            k=args.k,
            n=args.n,
            no_metric_cards=args.no_metric_cards,
            force_reindex=args.force_reindex,
            resized=args.resized,
        )
        print("\n🎉 Final validation correlations:")
        for corr_type, score in scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
    except Exception as e:
        print(f"\n💥 Ablation run failed: {e}")
        raise e


if __name__ == "__main__":
    main()


