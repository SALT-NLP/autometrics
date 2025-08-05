#!/usr/bin/env python3
"""
Example script showing how to extend the realistic regression test for multiple datasets.
This demonstrates the modular extensibility of the test framework.
"""

import os
import sys
from test_regression_realistic import ExperimentConfig, run_realistic_experiment

def test_multiple_datasets():
    """Example of testing multiple datasets with the realistic framework."""
    
    # Define experiments for different datasets
    experiments = [
        ExperimentConfig(
            dataset_name="SimpDA",
            target_measure="simplicity",
            num_to_retrieve=30,
            num_to_regress=5,
            seed=42
        ),
        # Example: Add more datasets here
        # ExperimentConfig(
        #     dataset_name="AnotherDataset",
        #     target_measure="quality",
        #     num_to_retrieve=30,
        #     num_to_regress=5,
        #     seed=42
        # ),
    ]
    
    results = {}
    
    for i, config in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}: {config.dataset_name}")
        print(f"{'='*80}")
        
        try:
            experiment_results = run_realistic_experiment(config)
            results[config.dataset_name] = experiment_results
        except Exception as e:
            print(f"❌ Failed to run experiment for {config.dataset_name}: {e}")
            results[config.dataset_name] = None
    
    # Summary across all experiments
    print(f"\n{'='*80}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, experiment_results in results.items():
        if experiment_results is None:
            print(f"❌ {dataset_name}: FAILED")
            continue
            
        # Find best strategy for this dataset
        successful_strategies = [(name, result) for name, result in experiment_results.items() if result['success']]
        if successful_strategies:
            best_strategy = max(successful_strategies, key=lambda x: x[1]['validation_correlation'])
            print(f"✅ {dataset_name}: {best_strategy[0]} (corr: {best_strategy[1]['validation_correlation']:.4f})")
        else:
            print(f"❌ {dataset_name}: No successful strategies")

def test_with_custom_config():
    """Example of testing with custom configuration."""
    
    # Custom configuration with different settings
    custom_config = ExperimentConfig(
        dataset_name="SimpDA",
        target_measure="simplicity",
        num_to_retrieve=50,  # More metrics
        num_to_regress=10,   # More final metrics
        seed=123,            # Different seed
        regenerate_metrics=True  # Force regeneration
    )
    
    print("Testing with custom configuration...")
    results = run_realistic_experiment(custom_config)
    
    return results

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    print("🔍 MULTI-DATASET REGRESSION TESTING EXAMPLE")
    print("=" * 60)
    
    # Example 1: Test multiple datasets
    print("\n1. Testing multiple datasets...")
    test_multiple_datasets()
    
    # Example 2: Test with custom configuration
    print("\n2. Testing with custom configuration...")
    # Uncomment to run custom config test
    # test_with_custom_config()
    
    print("\n✅ Multi-dataset testing example completed!")
    print("\nTo add new datasets:")
    print("1. Add dataset loading method to DatasetLoader class")
    print("2. Add dataset name to load_dataset method")
    print("3. Create ExperimentConfig for your dataset")
    print("4. Run the experiment!") 