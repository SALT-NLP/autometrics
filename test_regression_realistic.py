#!/usr/bin/env python3
"""
Realistic regression strategy comparison test with default Autometrics settings.
This test uses the full metric bank, large metric generation budget, and tests all regression strategies
in a realistic setting (30 metrics retrieved, 5 selected via regression).
"""

import os
import sys
import time
import dspy
import numpy as np
from typing import List, Dict, Any, Type, Optional
from dataclasses import dataclass

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autometrics.autometrics import Autometrics
from autometrics.dataset.datasets.simplification.simplification import SimpDA
from autometrics.aggregator.regression.Linear import Linear
from autometrics.aggregator.regression.Ridge import Ridge
from autometrics.aggregator.regression.RandomForest import RandomForest
from autometrics.aggregator.regression.GradientBoosting import GradientBoosting
from autometrics.aggregator.regression.Lasso import Lasso
from autometrics.aggregator.regression.ElasticNet import ElasticNet
from autometrics.aggregator.regression.PLS import PLS
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.LLMRec import LLMRec

@dataclass
class ExperimentConfig:
    """Configuration for regression comparison experiments."""
    dataset_name: str
    target_measure: str
    num_to_retrieve: int = 30  # Default Autometrics setting
    num_to_regress: int = 5    # Default Autometrics setting
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    regenerate_metrics: bool = False

class DatasetLoader:
    """Modular dataset loader for extensibility."""
    
    @staticmethod
    def load_simpda(config: ExperimentConfig):
        """Load SimpDA dataset."""
        from autometrics.dataset.datasets.simplification.simplification import SimpDA
        dataset = SimpDA()
        train_dataset, val_dataset, test_dataset = dataset.get_splits(
            train_ratio=config.train_ratio, 
            val_ratio=config.val_ratio, 
            seed=config.seed
        )
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def load_dataset(config: ExperimentConfig):
        """Load dataset based on configuration."""
        if config.dataset_name.lower() == "simpda":
            return DatasetLoader.load_simpda(config)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

class RegressionStrategyTester:
    """Modular regression strategy tester."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.strategies = [
            ("Linear", Linear),
            ("Ridge", Ridge), 
            ("RandomForest", RandomForest),
            ("GradientBoosting", GradientBoosting),
            ("Lasso", Lasso),
            ("ElasticNet", ElasticNet),
            ("PLS", lambda **kwargs: PLS(n_components=1, **kwargs))
        ]
        
    def run_realistic_comparison(self, generator_llm: dspy.LM, judge_llm: dspy.LM):
        """Run realistic regression strategy comparison."""
        
        print("🔍 REALISTIC REGRESSION STRATEGY COMPARISON")
        print("=" * 80)
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Target Measure: {self.config.target_measure}")
        print(f"Retrieve: {self.config.num_to_retrieve} metrics")
        print(f"Regress to: {self.config.num_to_regress} metrics")
        print(f"Seed: {self.config.seed}")
        print("=" * 80)
        
        # Load dataset
        print(f"\n📊 Loading {self.config.dataset_name} dataset...")
        train_dataset, val_dataset, test_dataset = DatasetLoader.load_dataset(self.config)
        print(f"   Train: {len(train_dataset.get_dataframe())} examples")
        print(f"   Validation: {len(val_dataset.get_dataframe())} examples")
        print(f"   Test: {len(test_dataset.get_dataframe())} examples")
        
        # Create Autometrics with default settings
        print(f"\n🚀 Creating Autometrics with default settings...")
        autometrics = Autometrics(
            regression_strategy=Lasso,  # Default, will be overridden
            retriever_kwargs={
                "recommenders": [ColBERT, LLMRec],
                "top_ks": [60, 30],  # Default Autometrics settings
                "index_paths": [None, None]
            }
        )
        
        # Run pipeline up to evaluation step
        start_time = time.time()
        
        # Step 1: Generate/Load metrics
        print(f"\n[Realistic] Step 1: Generating/Loading Metrics")
        generated_metrics = autometrics._generate_or_load_metrics(
            train_dataset, self.config.target_measure, generator_llm, judge_llm, 
            self.config.regenerate_metrics
        )
        print(f"[Realistic] Generated/Loaded {len(generated_metrics)} metrics")
        
        # Step 2: Load metric bank and merge
        print(f"\n[Realistic] Step 2: Loading Metric Bank")
        metric_bank = autometrics._load_metric_bank(train_dataset)
        metric_bank = autometrics._merge_generated_with_bank(metric_bank, generated_metrics)
        print(f"[Realistic] Loaded {len(metric_bank)} metrics in bank")
        
        # Step 3: Configure retriever
        print("[Realistic] Configuring retriever...")
        retriever_kwargs = autometrics.retriever_kwargs.copy()
        retriever_kwargs["metric_classes"] = metric_bank
        retriever_kwargs["model"] = generator_llm
        retriever_kwargs = autometrics._validate_and_adjust_retriever_config(
            retriever_kwargs, train_dataset, metric_bank, self.config.num_to_retrieve
        )
        retriever_instance = autometrics.retriever(**retriever_kwargs)
        
        # Step 4: Retrieve metrics
        print(f"\n[Realistic] Step 4: Retrieving Top {self.config.num_to_retrieve} Metrics")
        retrieved_metrics = autometrics._retrieve_top_k_metrics(
            train_dataset, self.config.target_measure, self.config.num_to_retrieve, 
            retriever_instance, metric_bank
        )
        print(f"[Realistic] Retrieved {len(retrieved_metrics)} metrics")
        
        # Step 5: Evaluate metrics on training set
        print(f"\n[Realistic] Step 5: Evaluating {len(retrieved_metrics)} Metrics on Training Dataset")
        successful_metric_instances = autometrics._evaluate_metrics_on_dataset(train_dataset, retrieved_metrics)
        print(f"[Realistic] Successfully evaluated {len(successful_metric_instances)} metrics")
        
        # Step 6: Evaluate same metrics on validation set
        print(f"\n[Realistic] Step 6: Evaluating {len(successful_metric_instances)} Metrics on Validation Dataset")
        for metric in successful_metric_instances:
            val_dataset.add_metric(metric, update_dataset=True)
        print(f"[Realistic] Added {len(successful_metric_instances)} metrics to validation dataset")
        
        pipeline_time = time.time() - start_time
        print(f"\n⏱️  Pipeline setup completed in {pipeline_time:.2f}s")
        
        # Now run all regression strategies on the same evaluated metrics
        print(f"\n🧪 Testing {len(self.strategies)} regression strategies on same data...")
        
        results = {}
        
        for strategy_name, strategy_class in self.strategies:
            print(f"\n🧪 Testing {strategy_name}...")
            
            strategy_start_time = time.time()
            
            try:
                # Create regression instance
                regression_kwargs = autometrics.regression_kwargs.copy()
                regression_kwargs["dataset"] = train_dataset
                regression_instance = strategy_class(**regression_kwargs)
                
                # Run regression on the same evaluated metrics
                regression_results = autometrics._regress_and_select_top_n(
                    train_dataset, 
                    successful_metric_instances, 
                    self.config.target_measure, 
                    self.config.num_to_regress,
                    regression_instance
                )
                
                # Get the final regression metric
                final_regression = regression_results['regression_metric']
                
                # Give the regression metric a unique name to avoid overwriting
                final_regression.name = f"{strategy_name}_Regression_{self.config.target_measure}"
                
                # Evaluate on validation set
                print(f"   Evaluating {strategy_name} on validation set...")
                final_regression._predict_unsafe(val_dataset, update_dataset=True)
                
                # Calculate validation performance
                val_df = val_dataset.get_dataframe()
                y_true = val_df[self.config.target_measure]
                y_pred = val_df[final_regression.name]
                
                # Calculate correlation (primary metric)
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                
                # Calculate R² score
                from sklearn.metrics import r2_score
                r2 = r2_score(y_true, y_pred)
                
                # Calculate mean squared error
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(y_true, y_pred)
                
                strategy_time = time.time() - strategy_start_time
                
                # Get importance scores
                importance_scores = regression_results['importance_scores']
                
                # Find the LLM judge metric in importance scores
                llm_judge_importance = None
                llm_judge_rank = None
                
                for i, (score, metric_name) in enumerate(importance_scores):
                    if "Clarity" in metric_name or "LLMJudge" in metric_name:
                        llm_judge_importance = score
                        llm_judge_rank = i + 1
                        break
                
                results[strategy_name] = {
                    "importance_scores": importance_scores[:10],  # Top 10
                    "llm_judge_importance": llm_judge_importance,
                    "llm_judge_rank": llm_judge_rank,
                    "time": strategy_time,
                    "top_metrics": regression_results['top_metrics'],
                    "validation_correlation": correlation,
                    "validation_r2": r2,
                    "validation_mse": mse,
                    "success": True
                }
                
                print(f"   Top 10 importance scores:")
                for i, (score, metric_name) in enumerate(importance_scores[:10]):
                    marker = "🎯" if "Clarity" in metric_name or "LLMJudge" in metric_name else "  "
                    print(f"   {marker} {i+1}. {metric_name}: {score:.6f}")
                
                print(f"   📊 Validation Performance:")
                print(f"      Correlation: {correlation:.4f}")
                print(f"      R² Score: {r2:.4f}")
                print(f"      MSE: {mse:.4f}")
                
                if llm_judge_importance is not None:
                    print(f"   🎯 LLM Judge found at rank {llm_judge_rank} with importance {llm_judge_importance:.6f}")
                else:
                    print(f"   ⚠️  LLM Judge not found in top importance scores")
                    
            except Exception as e:
                print(f"   ❌ Error with {strategy_name}: {e}")
                results[strategy_name] = {
                    "importance_scores": [],
                    "llm_judge_importance": None,
                    "llm_judge_rank": None,
                    "time": time.time() - strategy_start_time,
                    "top_metrics": [],
                    "validation_correlation": None,
                    "validation_r2": None,
                    "validation_mse": None,
                    "success": False,
                    "error": str(e)
                }
        
        # Comprehensive analysis
        self._print_comprehensive_analysis(results, pipeline_time)
        
        return results
    
    def _print_comprehensive_analysis(self, results: Dict[str, Any], pipeline_time: float):
        """Print comprehensive analysis of results."""
        
        print(f"\n📊 REALISTIC REGRESSION STRATEGY ANALYSIS")
        print("=" * 120)
        print(f"Pipeline setup time: {pipeline_time:.2f}s")
        print(f"Total regression testing time: {sum(r['time'] for r in results.values() if r['success']):.2f}s")
        print(f"Total time: {pipeline_time + sum(r['time'] for r in results.values() if r['success']):.2f}s")
        print("=" * 120)
        
        # Detailed results table
        print(f"{'Strategy':<15} | {'Val Corr':<8} | {'Val R²':<8} | {'Val MSE':<8} | {'LLM Rank':<8} | {'Time':<8} | {'Success':<8} | {'Top Metrics'}")
        print("-" * 120)
        
        successful_strategies = []
        for strategy_name, result in results.items():
            if result['success']:
                correlation = result['validation_correlation']
                r2 = result['validation_r2']
                mse = result['validation_mse']
                llm_rank = result['llm_judge_rank'] or "Not found"
                success = "✓"
                top_metric_names = [m.get_name() for m in result['top_metrics']]
                
                print(f"{strategy_name:<15} | {correlation:<8.4f} | {r2:<8.4f} | {mse:<8.4f} | {str(llm_rank):<8} | {result['time']:<8.2f}s | {success:<8} | {', '.join(top_metric_names[:3])}")
                successful_strategies.append((strategy_name, result))
            else:
                print(f"{strategy_name:<15} | {'FAILED':<8} | {'FAILED':<8} | {'FAILED':<8} | {'FAILED':<8} | {result['time']:<8.2f}s | ✗ | FAILED")
        
        # Statistical analysis
        print(f"\n📈 STATISTICAL ANALYSIS")
        print("=" * 60)
        
        if successful_strategies:
            # Find best strategies by different criteria
            best_by_correlation = max(successful_strategies, key=lambda x: x[1]['validation_correlation'])
            best_by_r2 = max(successful_strategies, key=lambda x: x[1]['validation_r2'])
            best_by_llm_rank = min(successful_strategies, key=lambda x: x[1]['llm_judge_rank'] or 999)
            fastest = min(successful_strategies, key=lambda x: x[1]['time'])
            
            print(f"🏆 Best by Validation Correlation: {best_by_correlation[0]} (corr: {best_by_correlation[1]['validation_correlation']:.4f})")
            print(f"🏆 Best by Validation R²: {best_by_r2[0]} (R²: {best_by_r2[1]['validation_r2']:.4f})")
            print(f"🎯 Best by LLM Judge Rank: {best_by_llm_rank[0]} (rank: {best_by_llm_rank[1]['llm_judge_rank']})")
            print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['time']:.2f}s)")
            
            # Analyze strategy types
            linear_strategies = [s for s in successful_strategies if any(name in s[0].lower() for name in ['linear', 'ridge', 'lasso', 'elastic'])]
            tree_strategies = [s for s in successful_strategies if any(name in s[0].lower() for name in ['forest', 'gradient'])]
            
            print(f"\n📊 Strategy Type Analysis:")
            print(f"   Linear-based strategies: {len(linear_strategies)}")
            print(f"   Tree-based strategies: {len(tree_strategies)}")
            
            # Compare linear vs tree-based
            if linear_strategies and tree_strategies:
                linear_avg_corr = np.mean([s[1]['validation_correlation'] for s in linear_strategies])
                tree_avg_corr = np.mean([s[1]['validation_correlation'] for s in tree_strategies])
                print(f"   Linear strategies avg correlation: {linear_avg_corr:.4f}")
                print(f"   Tree strategies avg correlation: {tree_avg_corr:.4f}")
        
        # Error analysis
        failed_strategies = [name for name, result in results.items() if not result['success']]
        if failed_strategies:
            print(f"\n❌ Failed Strategies:")
            for name in failed_strategies:
                print(f"   - {name}: {results[name].get('error', 'Unknown error')}")

def run_realistic_experiment(config: ExperimentConfig):
    """Run a realistic regression comparison experiment."""
    
    # Configure LLMs
    generator_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    judge_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create tester and run experiment
    tester = RegressionStrategyTester(config)
    results = tester.run_realistic_comparison(generator_llm, judge_llm)
    
    return results

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Configure experiment for SimpDA
    config = ExperimentConfig(
        dataset_name="SimpDA",
        target_measure="simplicity",
        num_to_retrieve=30,  # Default Autometrics setting
        num_to_regress=5,    # Default Autometrics setting
        seed=42
    )
    
    # Run realistic experiment
    results = run_realistic_experiment(config)
    
    print(f"\n✅ Realistic regression strategy comparison completed!")
    print("Check the results above to find the best strategy for your use case.") 