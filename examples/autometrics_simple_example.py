#!/usr/bin/env python3
"""
Autometrics on a real dataset, all defaults
===========================================

This script runs the full pipeline — metric generation + retrieval from the
built-in bank + PLS aggregation — on the HelpSteer dataset with no custom
configuration. It's the second step after `tutorial.py`: same `Autometrics()`,
but now on real data that exercises the metric bank and retrievers.

If you just want the minimal generated-only entry point, see `tutorial.py`.
For full customization, see `autometrics_example.py`.

Usage:
    export OPENAI_API_KEY="your-api-key-here"
    python autometrics_simple_example.py
"""

import os
import sys
import dspy

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autometrics.autometrics import Autometrics
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer

def main():
    """Run the Autometrics pipeline with minimal configuration."""
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # 1. Load a dataset (HelpSteer is a good default)
    print("📊 Loading HelpSteer dataset...")
    dataset = HelpSteer()
    target_measure = "helpfulness"  # Good default measure
    
    # 2. Configure LLMs (GPT-4o-mini is a good default)
    print("🤖 Configuring LLMs...")
    generator_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    judge_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    
    # 3. Create Autometrics with ALL defaults - no parameters needed!
    print("🔧 Creating Autometrics pipeline...")
    autometrics = Autometrics()  # Uses all meaningful defaults from method signature!
    # The method signature shows exactly what defaults are used:
    # - metric_generation_configs=DEFAULT_GENERATOR_CONFIGS
    # - retriever=PipelinedRec
    # - retriever_kwargs=DEFAULT_RETRIEVER_KWARGS (ColBERT→LLMRec)
    # - regression_strategy=Lasso (class, not instance)
    # - regression_kwargs=DEFAULT_REGRESSION_KWARGS (empty for now, dataset added automatically)
    # - metric_bank=all_metric_classes (auto-switches to reference_free if no reference columns)
    # - seed=42
    # - allowed_failed_metrics=0
    
    # 4. Run the pipeline with defaults
    print("🚀 Running Autometrics pipeline...")
    # Run the Autometrics pipeline
    # This will:
    # - Generate metrics using all configured generators
    # - Retrieve the most relevant metrics from the bank
    # - Evaluate metrics on the dataset
    # - Use regression to select the top 5 most important metrics
    # - Add the final regression metric to the dataset (hybrid approach: safe experimentation + user access)
    # - Generate a report card
    results = autometrics.run(
        dataset=dataset,
        target_measure=target_measure,
        generator_llm=generator_llm,
        judge_llm=judge_llm
    )
    
    # 5. Display results
    print("\n" + "="*60)
    print("🎉 AUTOMETRICS PIPELINE COMPLETE!")
    print("="*60)
    
    print(f"\n📈 Results Summary:")
    print(f"   Dataset: {results['dataset'].get_name()}")
    print(f"   Target: {results['target_measure']}")
    print(f"   Generated: {len(results['all_generated_metrics'])} metrics")
    print(f"   Retrieved: {len(results['retrieved_metrics'])} metrics")
    print(f"   Selected: {len(results['top_metrics'])} top metrics")
    
    if results['top_metrics']:
        print(f"\n🏆 Top Selected Metrics:")
        for i, metric in enumerate(results['top_metrics']):
            print(f"   {i+1}. {metric.get_name()}")
    
    if results['regression_metric']:
        print(f"\n📊 Final Regression Metric:")
        print(f"   Name: {results['regression_metric'].get_name()}")
        print(f"   Description: {results['regression_metric'].get_description()}")
    
    print(f"\n📋 Full Report:")
    print(results['report_card'])
    
    print("\n✅ Pipeline completed successfully!")
    print("💡 Check the 'generated_metrics' directory for generated metric files.")
    print("🎯 This example used ALL defaults - no hyperparameter tuning required!")

if __name__ == "__main__":
    main() 