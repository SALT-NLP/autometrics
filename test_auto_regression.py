#!/usr/bin/env python3
"""
Simple test script to verify AutoRegression implementation.
"""

import os
import sys
import dspy
import numpy as np
import pandas as pd

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autometrics.autometrics import Autometrics
from autometrics.aggregator.regression.AutoRegression import AutoRegression
from autometrics.dataset.Dataset import Dataset
from autometrics.dataset.datasets.simplification.simplification import SimpDA

def test_auto_regression():
    """Test the AutoRegression functionality."""
    
    print("🧪 Testing AutoRegression Implementation")
    print("=" * 50)
    
    # Create a simple test dataset
    print("📊 Creating test dataset...")
    
    # Use SimpDA dataset for testing
    dataset = SimpDA()
    train_dataset, val_dataset, test_dataset = dataset.get_splits(
        train_ratio=0.8, 
        val_ratio=0.2, 
        seed=42
    )
    
    print(f"   Train: {len(train_dataset.get_dataframe())} examples")
    print(f"   Validation: {len(val_dataset.get_dataframe())} examples")
    print(f"   Test: {len(test_dataset.get_dataframe())} examples")
    
    # Create Autometrics with AutoRegression
    print("\n🚀 Creating Autometrics with AutoRegression...")
    autometrics = Autometrics(
        regression_strategy=AutoRegression,
        metric_generation_configs={
            "llm_judge": {"metrics_per_trial": 2, "description": "Basic LLM Judge"},
        },
        num_to_retrieve=10,  # Small number for quick testing
        num_to_regress=3,    # Small number for quick testing
        generated_metrics_dir="test_generated_metrics"
    )
    
    # Configure LLMs (you'll need to set OPENAI_API_KEY)
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    generator_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    judge_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Run the pipeline
    print("\n🔍 Running Autometrics pipeline...")
    try:
        results = autometrics.run(
            dataset=train_dataset,
            target_measure="simplicity",
            generator_llm=generator_llm,
            judge_llm=judge_llm,
            regenerate_metrics=True
        )
        
        print("\n✅ Pipeline completed successfully!")
        print(f"   Top metrics: {[m.get_name() for m in results['top_metrics']]}")
        print(f"   Regression metric: {results['regression_metric'].get_name()}")
        
        # Check if AutoRegression results are available
        if 'best_strategy' in results:
            print(f"   Best strategy: {results['best_strategy']}")
            print(f"   Best alpha: {results['best_alpha']}")
            print(f"   Best score: {results['best_score']:.4f}")
        
        # Print report card
        print("\n📋 Report Card:")
        print(results['report_card'])
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_auto_regression()
