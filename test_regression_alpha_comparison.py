#!/usr/bin/env python3
"""
Comprehensive regression strategy comparison with alpha parameter testing.
This test supports multiple datasets and tests different alpha configurations for Lasso, ElasticNet, and Ridge.
"""

import os
import sys
import time
import dspy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Type, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

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
class AlphaTestConfig:
    """Configuration for alpha parameter testing."""
    alpha: float
    description: str
    is_two_stage: bool = False
    selection_alpha: Optional[float] = None
    final_alpha: Optional[float] = None

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
    alpha_configs: Optional[List[AlphaTestConfig]] = None

class DatasetLoader:
    """Modular dataset loader for extensibility."""
    
    @staticmethod
    def load_simpda(config: ExperimentConfig):
        """Load SimpDA dataset."""
        dataset = SimpDA()
        train_dataset, val_dataset, test_dataset = dataset.get_splits(
            train_ratio=config.train_ratio, 
            val_ratio=config.val_ratio, 
            seed=config.seed
        )
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def load_evalgen_product(config: ExperimentConfig):
        """Load EvalGenProduct dataset."""
        from autometrics.dataset.datasets.evalgen.evalgen import EvalGen
        dataset = EvalGen('./autometrics/dataset/datasets/evalgen/product.csv')
        train_dataset, val_dataset, test_dataset = dataset.get_splits(
            train_ratio=config.train_ratio, 
            val_ratio=config.val_ratio, 
            seed=config.seed
        )
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def load_cogym_travel_outcome(config: ExperimentConfig):
        """Load CoGymTravelOutcome dataset."""
        from autometrics.dataset.datasets.cogym.cogym import CoGymTravelOutcome
        dataset = CoGymTravelOutcome()
        train_dataset, val_dataset, test_dataset = dataset.get_splits(
            train_ratio=config.train_ratio, 
            val_ratio=config.val_ratio, 
            seed=config.seed
        )
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def load_helpsteer(config: ExperimentConfig):
        """Load HelpSteer dataset."""
        from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer
        dataset = HelpSteer()
        train_dataset, val_dataset, test_dataset = dataset.get_splits(
            train_ratio=config.train_ratio, 
            val_ratio=config.val_ratio, 
            seed=config.seed,
            max_size=1000
        )
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def load_dataset(config: ExperimentConfig):
        """Load dataset based on configuration."""
        if config.dataset_name.lower() == "simpda":
            return DatasetLoader.load_simpda(config)
        elif config.dataset_name.lower() in ["evalgenproduct", "evalgen_product"]:
            return DatasetLoader.load_evalgen_product(config)
        elif config.dataset_name.lower() == "cogymtraveloutcome":
            return DatasetLoader.load_cogym_travel_outcome(config)
        elif config.dataset_name.lower() == "helpsteer":
            return DatasetLoader.load_helpsteer(config)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

class AlphaAwareRegression:
    """Wrapper class to allow alpha parameter modification for regression strategies."""
    
    def __init__(self, strategy_class, alpha: float, **kwargs):
        self.strategy_class = strategy_class
        self.alpha = alpha
        self.kwargs = kwargs
        self._instance = None
    
    def create_instance(self, dataset=None):
        """Create a regression instance with the specified alpha."""
        if self.strategy_class.__name__ in ['Lasso', 'Ridge', 'ElasticNet']:
            # For sklearn-based regressors, we need to modify the model parameters
            instance = self.strategy_class(dataset=dataset, **self.kwargs)
            
            # Update the underlying sklearn model with new alpha
            if hasattr(instance, 'model'):
                if self.strategy_class.__name__ == 'Lasso':
                    from sklearn.linear_model import Lasso as SklearnLasso
                    instance.model = SklearnLasso(alpha=self.alpha)
                elif self.strategy_class.__name__ == 'Ridge':
                    from sklearn.linear_model import Ridge as SklearnRidge
                    instance.model = SklearnRidge(alpha=self.alpha)
                elif self.strategy_class.__name__ == 'ElasticNet':
                    from sklearn.linear_model import ElasticNet as SklearnElasticNet
                    instance.model = SklearnElasticNet(alpha=self.alpha, l1_ratio=0.5)
            
            return instance
        else:
            # For other strategies, just return the original instance
            return self.strategy_class(dataset=dataset, **self.kwargs)

class TwoStageRegression:
    """Implements two-stage regression: alpha=0.01 for selection, then alpha=1 for final regression."""
    
    def __init__(self, strategy_class, selection_alpha: float = 0.01, final_alpha: float = 1.0, **kwargs):
        self.strategy_class = strategy_class
        self.selection_alpha = selection_alpha
        self.final_alpha = final_alpha
        self.kwargs = kwargs
    
    def create_selection_instance(self, dataset=None):
        """Create instance for metric selection (low alpha)."""
        return AlphaAwareRegression(self.strategy_class, self.selection_alpha, **self.kwargs).create_instance(dataset)
    
    def create_final_instance(self, dataset=None):
        """Create instance for final regression (high alpha)."""
        return AlphaAwareRegression(self.strategy_class, self.final_alpha, **self.kwargs).create_instance(dataset)

class AutoModeRegression:
    """Implements auto mode: alpha=1.0 for selection, then use only non-zero coefficients for final regression."""
    
    def __init__(self, strategy_class, selection_alpha: float = 1.0, **kwargs):
        self.strategy_class = strategy_class
        self.selection_alpha = selection_alpha
        self.kwargs = kwargs
    
    def create_selection_instance(self, dataset=None):
        """Create instance for metric selection (high alpha to get sparse selection)."""
        return AlphaAwareRegression(self.strategy_class, self.selection_alpha, **self.kwargs).create_instance(dataset)
    
    def create_final_instance(self, dataset=None):
        """Create instance for final regression (using same alpha as selection)."""
        return AlphaAwareRegression(self.strategy_class, self.selection_alpha, **self.kwargs).create_instance(dataset)

class AlphaComparisonTester:
    """Comprehensive alpha parameter comparison tester."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base_strategies = [
            ("Linear", Linear),
            ("RandomForest", RandomForest),
            ("GradientBoosting", GradientBoosting),
            ("PLS", lambda **kwargs: PLS(n_components=1, **kwargs))
        ]
        
        # Define alpha configurations to test
        if config.alpha_configs is None:
            config.alpha_configs = [
                AlphaTestConfig(alpha=0.01, description="Current (alpha=0.01)"),
                AlphaTestConfig(alpha=1.0, description="Old (alpha=1.0)"),
                AlphaTestConfig(alpha=0.01, description="Two-Stage (0.01→1.0)", 
                              is_two_stage=True, selection_alpha=0.01, final_alpha=1.0),
                AlphaTestConfig(alpha=1.0, description="Auto Mode (α=1.0→auto)", 
                              is_two_stage=True, selection_alpha=1.0, final_alpha=1.0)
            ]
        
        self.alpha_configs = config.alpha_configs
        
    def run_alpha_comparison(self, generator_llm: dspy.LM, judge_llm: dspy.LM):
        """Run comprehensive alpha parameter comparison."""
        
        print("🔍 ALPHA PARAMETER COMPARISON TEST")
        print("=" * 80)
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Target Measure: {self.config.target_measure}")
        print(f"Retrieve: {self.config.num_to_retrieve} metrics")
        print(f"Regress to: {self.config.num_to_regress} metrics")
        print(f"Seed: {self.config.seed}")
        print(f"Alpha Configurations: {len(self.alpha_configs)}")
        for i, config in enumerate(self.alpha_configs):
            print(f"  {i+1}. {config.description}")
        print("=" * 80)
        
        # Load dataset
        print(f"\n📊 Loading {self.config.dataset_name} dataset...")
        train_dataset, val_dataset, test_dataset = DatasetLoader.load_dataset(self.config)
        print(f"   Train: {len(train_dataset.get_dataframe())} examples")
        print(f"   Validation: {len(val_dataset.get_dataframe())} examples")
        print(f"   Test: {len(test_dataset.get_dataframe())} examples")
        
        # Create Autometrics with dataset-specific generated metrics directory
        safe_dataset_name = self.config.dataset_name.replace(" ", "_").replace("/", "_")
        generated_metrics_dir = f"generated_metrics_{safe_dataset_name}"
        
        print(f"\n🚀 Creating Autometrics with dataset-specific metrics directory: {generated_metrics_dir}")
        autometrics = Autometrics(
            regression_strategy=Lasso,  # Default, will be overridden
            retriever_kwargs={
                "recommenders": [ColBERT, LLMRec],
                "top_ks": [60, 30],  # Default Autometrics settings
                "index_paths": [None, None]
            },
            generated_metrics_dir=generated_metrics_dir
        )
        
        # Run pipeline up to evaluation step (same for all alpha tests)
        start_time = time.time()
        
        # Step 1: Generate/Load metrics
        print(f"\n[Alpha Test] Step 1: Generating/Loading Metrics")
        generated_metrics = autometrics._generate_or_load_metrics(
            train_dataset, self.config.target_measure, generator_llm, judge_llm, 
            self.config.regenerate_metrics
        )
        print(f"[Alpha Test] Generated/Loaded {len(generated_metrics)} metrics")
        
        # Step 2: Load metric bank and merge
        print(f"\n[Alpha Test] Step 2: Loading Metric Bank")
        metric_bank = autometrics._load_metric_bank(train_dataset)
        metric_bank = autometrics._merge_generated_with_bank(metric_bank, generated_metrics)
        print(f"[Alpha Test] Loaded {len(metric_bank)} metrics in bank")
        
        # Step 3: Configure retriever
        print("[Alpha Test] Configuring retriever...")
        retriever_kwargs = autometrics.retriever_kwargs.copy()
        retriever_kwargs["metric_classes"] = metric_bank
        retriever_kwargs["model"] = generator_llm
        retriever_kwargs = autometrics._validate_and_adjust_retriever_config(
            retriever_kwargs, train_dataset, metric_bank, self.config.num_to_retrieve
        )
        retriever_instance = autometrics.retriever(**retriever_kwargs)
        
        # Step 4: Retrieve metrics
        print(f"\n[Alpha Test] Step 4: Retrieving Top {self.config.num_to_retrieve} Metrics")
        retrieved_metrics = autometrics._retrieve_top_k_metrics(
            train_dataset, self.config.target_measure, self.config.num_to_retrieve, 
            retriever_instance, metric_bank
        )
        print(f"[Alpha Test] Retrieved {len(retrieved_metrics)} metrics")
        
        # Step 5: Evaluate metrics on training set
        print(f"\n[Alpha Test] Step 5: Evaluating {len(retrieved_metrics)} Metrics on Training Dataset")
        successful_metric_instances = autometrics._evaluate_metrics_on_dataset(train_dataset, retrieved_metrics)
        print(f"[Alpha Test] Successfully evaluated {len(successful_metric_instances)} metrics")
        
        # Step 6: Filter metrics to ensure they work on all datasets
        print(f"\n[Alpha Test] Step 6: Filtering metrics to ensure availability on all datasets")
        
        # Debug metric failures first
        self._debug_metric_failures(autometrics, train_dataset, val_dataset, test_dataset, successful_metric_instances)
        
        filtered_metric_instances = self._get_metrics_available_on_all_datasets(
            autometrics, train_dataset, val_dataset, test_dataset, successful_metric_instances
        )
        
        if len(filtered_metric_instances) == 0:
            raise ValueError("No metrics available on all datasets! Cannot proceed with regression.")
        
        pipeline_time = time.time() - start_time
        print(f"\n⏱️  Pipeline setup completed in {pipeline_time:.2f}s")
        
        # Now run all alpha configurations
        print(f"\n🧪 Testing {len(self.alpha_configs)} alpha configurations...")
        
        all_results = {}
        
        for alpha_config in self.alpha_configs:
            print(f"\n🧪 Testing {alpha_config.description}...")
            
            alpha_start_time = time.time()
            
            try:
                if alpha_config.is_two_stage:
                    # Two-stage approach
                    results = self._run_two_stage_regression(
                        autometrics, train_dataset, val_dataset, test_dataset,
                        filtered_metric_instances, alpha_config
                    )
                else:
                    # Single-stage approach
                    results = self._run_single_stage_regression(
                        autometrics, train_dataset, val_dataset, test_dataset,
                        filtered_metric_instances, alpha_config
                    )
                
                alpha_time = time.time() - alpha_start_time
                results['time'] = alpha_time
                results['success'] = True
                
                all_results[alpha_config.description] = results
                
                # Print results
                self._print_alpha_results(alpha_config, results)
                
            except Exception as e:
                print(f"   ❌ Error with {alpha_config.description}: {e}")
                all_results[alpha_config.description] = {
                    'time': time.time() - alpha_start_time,
                    'success': False,
                    'error': str(e)
                }
                raise e
        
        # Comprehensive analysis
        self._print_comprehensive_alpha_analysis(all_results, pipeline_time)
        
        # Create scatter plots for visualization
        self._create_scatter_plots(all_results, val_dataset, test_dataset, self.config.target_measure)
        
        return all_results
    
    def _run_single_stage_regression(self, autometrics, train_dataset, val_dataset, test_dataset,
                                   successful_metric_instances, alpha_config):
        """Run single-stage regression with specified alpha."""
        
        results = {}
        
        # Test all regression strategies with the specified alpha
        for strategy_name, strategy_class in self.base_strategies + [
            ("Lasso", Lasso), ("Ridge", Ridge), ("ElasticNet", ElasticNet)
        ]:
            print(f"   Testing {strategy_name} with alpha={alpha_config.alpha}...")
            
            try:
                # Create regression instance with alpha
                if strategy_name in ['Lasso', 'Ridge', 'ElasticNet']:
                    regression_instance = AlphaAwareRegression(
                        strategy_class, alpha_config.alpha
                    ).create_instance(dataset=train_dataset)
                else:
                    regression_kwargs = autometrics.regression_kwargs.copy()
                    regression_kwargs["dataset"] = train_dataset
                    regression_instance = strategy_class(**regression_kwargs)
                
                # Run regression
                regression_results = autometrics._regress_and_select_top_n(
                    train_dataset, 
                    successful_metric_instances, 
                    self.config.target_measure, 
                    self.config.num_to_regress,
                    regression_instance
                )
                
                # Evaluate on validation set
                final_regression = regression_results['regression_metric']
                final_regression.name = f"{strategy_name}_{alpha_config.alpha}_{self.config.target_measure}"
                final_regression._predict_unsafe(val_dataset, update_dataset=True)
                
                # Calculate validation performance
                val_df = val_dataset.get_dataframe()
                y_true_val = val_df[self.config.target_measure]
                y_pred_val = val_df[final_regression.name]
                
                # Filter out NaN/inf in y_true or y_pred
                valid_mask_val = ~(np.isnan(y_true_val) | np.isnan(y_pred_val) | np.isinf(y_true_val) | np.isinf(y_pred_val))
                y_true_valid_val = y_true_val[valid_mask_val]
                y_pred_valid_val = y_pred_val[valid_mask_val]
                
                correlation_val = np.corrcoef(y_true_valid_val, y_pred_valid_val)[0, 1] if len(y_true_valid_val) > 1 else 0.0
                # Handle NaN correlation values
                if np.isnan(correlation_val):
                    correlation_val = 0.0
                
                from sklearn.metrics import r2_score, mean_squared_error
                r2_val = r2_score(y_true_valid_val, y_pred_valid_val) if len(y_true_valid_val) > 1 else 0.0
                mse_val = mean_squared_error(y_true_valid_val, y_pred_valid_val) if len(y_true_valid_val) > 1 else 0.0
                
                # Evaluate on test set
                final_regression._predict_unsafe(test_dataset, update_dataset=True)
                
                # Calculate test performance
                test_df = test_dataset.get_dataframe()
                y_true_test = test_df[self.config.target_measure]
                y_pred_test = test_df[final_regression.name]
                
                # Filter out NaN/inf in y_true or y_pred
                valid_mask_test = ~(np.isnan(y_true_test) | np.isnan(y_pred_test) | np.isinf(y_true_test) | np.isinf(y_pred_test))
                y_true_valid_test = y_true_test[valid_mask_test]
                y_pred_valid_test = y_pred_test[valid_mask_test]
                
                correlation_test = np.corrcoef(y_true_valid_test, y_pred_valid_test)[0, 1] if len(y_true_valid_test) > 1 else 0.0
                # Handle NaN correlation values
                if np.isnan(correlation_test):
                    correlation_test = 0.0
                
                r2_test = r2_score(y_true_valid_test, y_pred_valid_test) if len(y_true_valid_test) > 1 else 0.0
                mse_test = mean_squared_error(y_true_valid_test, y_pred_valid_test) if len(y_true_valid_test) > 1 else 0.0
                
                results[strategy_name] = {
                    'correlation_val': correlation_val,
                    'r2_val': r2_val,
                    'mse_val': mse_val,
                    'correlation_test': correlation_test,
                    'r2_test': r2_test,
                    'mse_test': mse_test,
                    'importance_scores': regression_results['importance_scores'][:10],
                    'top_metrics': regression_results['top_metrics'],
                    'regression_column': final_regression.name  # Store the actual column name
                }
                
            except Exception as e:
                print(f"     ❌ Error with {strategy_name}: {e}")
                results[strategy_name] = {
                    'correlation_val': None,
                    'r2_val': None,
                    'mse_val': None,
                    'correlation_test': None,
                    'r2_test': None,
                    'mse_test': None,
                    'importance_scores': [],
                    'top_metrics': [],
                    'regression_column': None,
                    'error': str(e)
                }
                raise e

        return results
    
    def _run_two_stage_regression(self, autometrics, train_dataset, val_dataset, test_dataset,
                                successful_metric_instances, alpha_config):
        """Run two-stage regression: selection with low alpha, final with high alpha."""
        
        results = {}
        
        # Test all regression strategies with two-stage approach
        for strategy_name, strategy_class in self.base_strategies + [
            ("Lasso", Lasso), ("Ridge", Ridge), ("ElasticNet", ElasticNet)
        ]:
            print(f"   Testing {strategy_name} with two-stage (α={alpha_config.selection_alpha}→{alpha_config.final_alpha})...")
            
            try:
                if strategy_name in ['Lasso', 'Ridge', 'ElasticNet']:
                    # Stage 1: Selection with low alpha
                    selection_instance = AlphaAwareRegression(
                        strategy_class, alpha_config.selection_alpha
                    ).create_instance(dataset=train_dataset)
                    
                    # Run selection regression
                    selection_results = autometrics._regress_and_select_top_n(
                        train_dataset, 
                        successful_metric_instances, 
                        self.config.target_measure, 
                        self.config.num_to_regress,
                        selection_instance
                    )
                    
                    # Get selected metrics
                    selected_metrics = selection_results['top_metrics']
                    
                    # Check if this is auto mode (selection_alpha == final_alpha == 1.0)
                    is_auto_mode = (alpha_config.selection_alpha == 1.0 and alpha_config.final_alpha == 1.0)
                    
                    if is_auto_mode:
                        # Auto mode: count non-zero coefficients to determine budget
                        importance_scores = selection_results['importance_scores']
                        
                        # importance_scores is a list of tuples: (coefficient_value, metric_name)
                        non_zero_count = sum(1 for score, metric_name in importance_scores if abs(score) > 1e-6)
                        auto_budget = max(1, non_zero_count)  # At least 1 metric
                        print(f"     Auto mode: {non_zero_count} non-zero coefficients → budget = {auto_budget}")
                        
                        # Use only the top metrics based on auto budget
                        selected_metrics = selected_metrics[:auto_budget]
                    
                    # Stage 2: Final regression with high alpha on selected metrics
                    final_instance = AlphaAwareRegression(
                        strategy_class, alpha_config.final_alpha
                    ).create_instance(dataset=train_dataset)
                    
                    # Run final regression on selected metrics only
                    final_results = autometrics._regress_and_select_top_n(
                        train_dataset, 
                        selected_metrics, 
                        self.config.target_measure, 
                        len(selected_metrics),  # Use all selected metrics
                        final_instance
                    )
                    
                    # Evaluate on validation set
                    final_regression = final_results['regression_metric']
                    final_regression.name = f"{strategy_name}_2stage_{self.config.target_measure}"
                    final_regression._predict_unsafe(val_dataset, update_dataset=True)
                    
                    # Calculate validation performance
                    val_df = val_dataset.get_dataframe()
                    y_true_val = val_df[self.config.target_measure]
                    y_pred_val = val_df[final_regression.name]

                    # Filter out NaN/inf in y_true or y_pred
                    valid_mask_val = ~(np.isnan(y_true_val) | np.isnan(y_pred_val) | np.isinf(y_true_val) | np.isinf(y_pred_val))
                    y_true_valid_val = y_true_val[valid_mask_val]
                    y_pred_valid_val = y_pred_val[valid_mask_val]

                    correlation_val = np.corrcoef(y_true_valid_val, y_pred_valid_val)[0, 1] if len(y_true_valid_val) > 1 else 0.0
                    # Handle NaN correlation values
                    if np.isnan(correlation_val):
                        correlation_val = 0.0
                    
                    from sklearn.metrics import r2_score, mean_squared_error
                    r2_val = r2_score(y_true_valid_val, y_pred_valid_val) if len(y_true_valid_val) > 1 else 0.0
                    mse_val = mean_squared_error(y_true_valid_val, y_pred_valid_val) if len(y_true_valid_val) > 1 else 0.0
                    
                    # Evaluate on test set
                    final_regression._predict_unsafe(test_dataset, update_dataset=True)
                    
                    # Calculate test performance
                    test_df = test_dataset.get_dataframe()
                    y_true_test = test_df[self.config.target_measure]
                    y_pred_test = test_df[final_regression.name]

                    # Filter out NaN/inf in y_true or y_pred
                    valid_mask_test = ~(np.isnan(y_true_test) | np.isnan(y_pred_test) | np.isinf(y_true_test) | np.isinf(y_pred_test))
                    y_true_valid_test = y_true_test[valid_mask_test]
                    y_pred_valid_test = y_pred_test[valid_mask_test]

                    correlation_test = np.corrcoef(y_true_valid_test, y_pred_valid_test)[0, 1] if len(y_true_valid_test) > 1 else 0.0
                    # Handle NaN correlation values
                    if np.isnan(correlation_test):
                        correlation_test = 0.0
                    
                    r2_test = r2_score(y_true_valid_test, y_pred_valid_test) if len(y_true_valid_test) > 1 else 0.0
                    mse_test = mean_squared_error(y_true_valid_test, y_pred_valid_test) if len(y_true_valid_test) > 1 else 0.0
                    
                    # Add auto mode info if applicable
                    auto_info = {}
                    if is_auto_mode:
                        # Calculate non-zero coefficients with proper type handling
                        non_zero_coeffs = 0
                        for score, metric_name in selection_results['importance_scores']:
                            if abs(score) > 1e-6:
                                non_zero_coeffs += 1
                        
                        auto_info = {
                            'auto_budget': len(selected_metrics),
                            'non_zero_coefficients': non_zero_coeffs
                        }
                    
                    results[strategy_name] = {
                        'correlation_val': correlation_val,
                        'r2_val': r2_val,
                        'mse_val': mse_val,
                        'correlation_test': correlation_test,
                        'r2_test': r2_test,
                        'mse_test': mse_test,
                        'importance_scores': final_results['importance_scores'][:10],
                        'top_metrics': final_results['top_metrics'],
                        'selection_importance_scores': selection_results['importance_scores'][:10],
                        'regression_column': final_regression.name,  # Store the actual column name
                        **auto_info
                    }
                    
                else:
                    # For non-alpha strategies, just run normally
                    regression_kwargs = autometrics.regression_kwargs.copy()
                    regression_kwargs["dataset"] = train_dataset
                    regression_instance = strategy_class(**regression_kwargs)
                    
                    regression_results = autometrics._regress_and_select_top_n(
                        train_dataset, 
                        successful_metric_instances, 
                        self.config.target_measure, 
                        self.config.num_to_regress,
                        regression_instance
                    )
                    
                    # Evaluate on validation set
                    final_regression = regression_results['regression_metric']
                    final_regression.name = f"{strategy_name}_2stage_{self.config.target_measure}"
                    final_regression._predict_unsafe(val_dataset, update_dataset=True)
                    
                    # Calculate validation performance
                    val_df = val_dataset.get_dataframe()
                    y_true_val = val_df[self.config.target_measure]
                    y_pred_val = val_df[final_regression.name]

                    # Filter out NaN/inf in y_true or y_pred
                    valid_mask_val = ~(np.isnan(y_true_val) | np.isnan(y_pred_val) | np.isinf(y_true_val) | np.isinf(y_pred_val))
                    y_true_valid_val = y_true_val[valid_mask_val]
                    y_pred_valid_val = y_pred_val[valid_mask_val]

                    correlation_val = np.corrcoef(y_true_valid_val, y_pred_valid_val)[0, 1] if len(y_true_valid_val) > 1 else 0.0
                    # Handle NaN correlation values
                    if np.isnan(correlation_val):
                        correlation_val = 0.0
                    
                    from sklearn.metrics import r2_score, mean_squared_error
                    r2_val = r2_score(y_true_valid_val, y_pred_valid_val) if len(y_true_valid_val) > 1 else 0.0
                    mse_val = mean_squared_error(y_true_valid_val, y_pred_valid_val) if len(y_true_valid_val) > 1 else 0.0
                    
                    # Evaluate on test set
                    final_regression._predict_unsafe(test_dataset, update_dataset=True)
                    
                    # Calculate test performance
                    test_df = test_dataset.get_dataframe()
                    y_true_test = test_df[self.config.target_measure]
                    y_pred_test = test_df[final_regression.name]

                    # Filter out NaN/inf in y_true or y_pred
                    valid_mask_test = ~(np.isnan(y_true_test) | np.isnan(y_pred_test) | np.isinf(y_true_test) | np.isinf(y_pred_test))
                    y_true_valid_test = y_true_test[valid_mask_test]
                    y_pred_valid_test = y_pred_test[valid_mask_test]

                    correlation_test = np.corrcoef(y_true_valid_test, y_pred_valid_test)[0, 1] if len(y_true_valid_test) > 1 else 0.0
                    # Handle NaN correlation values
                    if np.isnan(correlation_test):
                        correlation_test = 0.0
                    
                    r2_test = r2_score(y_true_valid_test, y_pred_valid_test) if len(y_true_valid_test) > 1 else 0.0
                    mse_test = mean_squared_error(y_true_valid_test, y_pred_valid_test) if len(y_true_valid_test) > 1 else 0.0
                    
                    results[strategy_name] = {
                        'correlation_val': correlation_val,
                        'r2_val': r2_val,
                        'mse_val': mse_val,
                        'correlation_test': correlation_test,
                        'r2_test': r2_test,
                        'mse_test': mse_test,
                        'importance_scores': regression_results['importance_scores'][:10],
                        'top_metrics': regression_results['top_metrics'],
                        'regression_column': final_regression.name  # Store the actual column name
                    }
                
            except Exception as e:
                print(f"     ❌ Error with {strategy_name}: {e}")
                results[strategy_name] = {
                    'correlation_val': None,
                    'r2_val': None,
                    'mse_val': None,
                    'correlation_test': None,
                    'r2_test': None,
                    'mse_test': None,
                    'importance_scores': [],
                    'top_metrics': [],
                    'regression_column': None,
                    'error': str(e)
                }
                raise e
        
        return results
    
    def _print_alpha_results(self, alpha_config, results):
        """Print results for a specific alpha configuration."""
        
        print(f"   📊 Results for {alpha_config.description}:")
        
        # Filter out metadata keys (time, success, error) and only look at strategy results
        strategy_results = {k: v for k, v in results.items() 
                          if k not in ['time', 'success', 'error']}
        
        successful_strategies = []
        for strategy_name, result in strategy_results.items():
            if result.get('correlation_val') is not None:
                correlation_val = result['correlation_val']
                r2_val = result['r2_val']
                mse_val = result['mse_val']
                correlation_test = result['correlation_test']
                r2_test = result['r2_test']
                mse_test = result['mse_test']
                
                # Add auto mode info if available
                auto_info = ""
                if 'auto_budget' in result:
                    auto_info = f" (auto budget: {result['auto_budget']})"
                
                print(f"      {strategy_name:<15}: val_corr={correlation_val:.4f}, test_corr={correlation_test:.4f}, val_R²={r2_val:.4f}, test_R²={r2_test:.4f}{auto_info}")
                successful_strategies.append((strategy_name, result))
            else:
                print(f"      {strategy_name:<15}: FAILED")
        
        if successful_strategies:
            best_val_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_val'])
            best_test_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_test'])
            print(f"   🏆 Best Val: {best_val_strategy[0]} (corr: {best_val_strategy[1]['correlation_val']:.4f})")
            print(f"   🏆 Best Test: {best_test_strategy[0]} (corr: {best_test_strategy[1]['correlation_test']:.4f})")
            if best_val_strategy[0] == best_test_strategy[0]:
                print(f"   ✅ Validation best matches test best!")
            else:
                print(f"   ⚠️  Validation best differs from test best")
    
    def _print_comprehensive_alpha_analysis(self, all_results, pipeline_time):
        """Print comprehensive analysis across all alpha configurations."""
        
        print(f"\n📊 COMPREHENSIVE ALPHA ANALYSIS")
        print("=" * 120)
        print(f"Pipeline setup time: {pipeline_time:.2f}s")
        print(f"Total alpha testing time: {sum(r['time'] for r in all_results.values() if r['success']):.2f}s")
        print("=" * 120)
        
        # Results table
        print(f"{'Alpha Config':<20} | {'Best Strategy':<15} | {'Best Val Corr':<12} | {'Best Test Corr':<12} | {'Time':<8}")
        print("-" * 120)
        
        for alpha_desc, results in all_results.items():
            if results['success']:
                # Filter out metadata keys (time, success, error) and only look at strategy results
                strategy_results = {k: v for k, v in results.items() 
                                  if k not in ['time', 'success', 'error']}
                successful_strategies = [(name, result) for name, result in strategy_results.items() 
                                       if isinstance(result, dict) and result.get('correlation_val') is not None]
                if successful_strategies:
                    best_val_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_val'])
                    best_test_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_test'])
                    print(f"{alpha_desc:<20} | {best_val_strategy[0]:<15} | {best_val_strategy[1]['correlation_val']:<12.4f} | {best_test_strategy[1]['correlation_test']:<12.4f} | {results['time']:<8.2f}s")
                else:
                    print(f"{alpha_desc:<20} | {'NO SUCCESS':<15} | {'N/A':<12} | {'N/A':<12} | {results['time']:<8.2f}s")
            else:
                print(f"{alpha_desc:<20} | {'FAILED':<15} | {'N/A':<12} | {'N/A':<12} | {results['time']:<8.2f}s")
        
        # Find overall best configuration
        successful_configs = [(desc, results) for desc, results in all_results.items() 
                             if results['success']]
        
        if successful_configs:
            best_val_configs = []
            best_test_configs = []
            for desc, results in successful_configs:
                # Filter out metadata keys
                strategy_results = {k: v for k, v in results.items() 
                                  if k not in ['time', 'success', 'error']}
                successful_strategies = [(name, result) for name, result in strategy_results.items() 
                                       if isinstance(result, dict) and result.get('correlation_val') is not None]
                if successful_strategies:
                    best_val_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_val'])
                    best_test_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_test'])
                    best_val_configs.append((desc, best_val_strategy[0], best_val_strategy[1]['correlation_val']))
                    best_test_configs.append((desc, best_test_strategy[0], best_test_strategy[1]['correlation_test']))
            
            if best_val_configs:
                overall_best_val = max(best_val_configs, key=lambda x: x[2])
                overall_best_test = max(best_test_configs, key=lambda x: x[2])
                print(f"\n🏆 OVERALL BEST VAL: {overall_best_val[0]} with {overall_best_val[1]} (corr: {overall_best_val[2]:.4f})")
                print(f"🏆 OVERALL BEST TEST: {overall_best_test[0]} with {overall_best_test[1]} (corr: {overall_best_test[2]:.4f})")
        
        # Alpha-specific insights
        print(f"\n📈 ALPHA-SPECIFIC INSIGHTS:")
        for alpha_desc, results in all_results.items():
            if results['success']:
                # Filter out metadata keys
                strategy_results = {k: v for k, v in results.items() 
                                  if k not in ['time', 'success', 'error']}
                alpha_strategies = [(name, result) for name, result in strategy_results.items() 
                                  if isinstance(result, dict) and result.get('correlation_val') is not None]
                if alpha_strategies:
                    avg_val_correlation = np.mean([s[1]['correlation_val'] for s in alpha_strategies])
                    avg_test_correlation = np.mean([s[1]['correlation_test'] for s in alpha_strategies])
                    print(f"   {alpha_desc}: avg val corr = {avg_val_correlation:.4f}, avg test corr = {avg_test_correlation:.4f}")
        
        # Print comprehensive summary table
        self._print_comprehensive_summary_table(all_results)
        
        # Analyze validation vs test set agreement
        self._analyze_validation_test_agreement(all_results)
    
    def _print_comprehensive_summary_table(self, all_results):
        """Print a comprehensive summary table showing all correlations for all methods."""
        
        print(f"\n📋 COMPREHENSIVE SUMMARY TABLE")
        print("=" * 140)
        
        # Get all unique strategy names across all configurations
        all_strategies = set()
        for alpha_desc, results in all_results.items():
            if results['success']:
                strategy_results = {k: v for k, v in results.items() 
                                  if k not in ['time', 'success', 'error']}
                all_strategies.update(strategy_results.keys())
        
        all_strategies = sorted(list(all_strategies))
        
        # Print header
        header = f"{'Alpha Config':<20} |"
        for strategy in all_strategies:
            header += f" {strategy:<12} |"
        print(header)
        print("-" * len(header))
        
        # Print validation correlations
        val_row = f"{'Validation Corr':<20} |"
        for strategy in all_strategies:
            # Find the best validation correlation for this strategy across all configs
            best_val_corr = None
            for alpha_desc, results in all_results.items():
                if results['success']:
                    strategy_results = {k: v for k, v in results.items() 
                                      if k not in ['time', 'success', 'error']}
                    if strategy in strategy_results and strategy_results[strategy].get('correlation_val') is not None:
                        if best_val_corr is None or strategy_results[strategy]['correlation_val'] > best_val_corr:
                            best_val_corr = strategy_results[strategy]['correlation_val']
            
            if best_val_corr is not None:
                val_row += f" {best_val_corr:<12.4f} |"
            else:
                val_row += f" {'N/A':<12} |"
        print(val_row)
        
        # Print test correlations
        test_row = f"{'Test Corr':<20} |"
        for strategy in all_strategies:
            # Find the best test correlation for this strategy across all configs
            best_test_corr = None
            for alpha_desc, results in all_results.items():
                if results['success']:
                    strategy_results = {k: v for k, v in results.items() 
                                      if k not in ['time', 'success', 'error']}
                    if strategy in strategy_results and strategy_results[strategy].get('correlation_test') is not None:
                        if best_test_corr is None or strategy_results[strategy]['correlation_test'] > best_test_corr:
                            best_test_corr = strategy_results[strategy]['correlation_test']
            
            if best_test_corr is not None:
                test_row += f" {best_test_corr:<12.4f} |"
            else:
                test_row += f" {'N/A':<12} |"
        print(test_row)
        
        # Print validation R²
        val_r2_row = f"{'Validation R²':<20} |"
        for strategy in all_strategies:
            # Find the best validation R² for this strategy across all configs
            best_val_r2 = None
            for alpha_desc, results in all_results.items():
                if results['success']:
                    strategy_results = {k: v for k, v in results.items() 
                                      if k not in ['time', 'success', 'error']}
                    if strategy in strategy_results and strategy_results[strategy].get('r2_val') is not None:
                        if best_val_r2 is None or strategy_results[strategy]['r2_val'] > best_val_r2:
                            best_val_r2 = strategy_results[strategy]['r2_val']
            
            if best_val_r2 is not None:
                val_r2_row += f" {best_val_r2:<12.4f} |"
            else:
                val_r2_row += f" {'N/A':<12} |"
        print(val_r2_row)
        
        # Print test R²
        test_r2_row = f"{'Test R²':<20} |"
        for strategy in all_strategies:
            # Find the best test R² for this strategy across all configs
            best_test_r2 = None
            for alpha_desc, results in all_results.items():
                if results['success']:
                    strategy_results = {k: v for k, v in results.items() 
                                      if k not in ['time', 'success', 'error']}
                    if strategy in strategy_results and strategy_results[strategy].get('r2_test') is not None:
                        if best_test_r2 is None or strategy_results[strategy]['r2_test'] > best_test_r2:
                            best_test_r2 = strategy_results[strategy]['r2_test']
            
            if best_test_r2 is not None:
                test_r2_row += f" {best_test_r2:<12.4f} |"
            else:
                test_r2_row += f" {'N/A':<12} |"
        print(test_r2_row)
        
        print("=" * len(header))
        
        # Now print detailed results for each alpha configuration
        print(f"\n📊 DETAILED RESULTS BY ALPHA CONFIGURATION")
        print("=" * 140)
        
        for alpha_desc, results in all_results.items():
            if results['success']:
                print(f"\n{alpha_desc}:")
                strategy_results = {k: v for k, v in results.items() 
                                  if k not in ['time', 'success', 'error']}
                
                # Print header for this config
                config_header = f"{'Strategy':<15} | {'Val Corr':<10} | {'Test Corr':<10} | {'Val R²':<10} | {'Test R²':<10}"
                print(config_header)
                print("-" * len(config_header))
                
                # Print results for each strategy
                for strategy_name, result in strategy_results.items():
                    if result.get('correlation_val') is not None:
                        print(f"{strategy_name:<15} | {result['correlation_val']:<10.4f} | {result['correlation_test']:<10.4f} | {result['r2_val']:<10.4f} | {result['r2_test']:<10.4f}")
                    else:
                        print(f"{strategy_name:<15} | {'FAILED':<10} | {'FAILED':<10} | {'FAILED':<10} | {'FAILED':<10}")
            else:
                print(f"\n{alpha_desc}: FAILED")
    
    def _analyze_validation_test_agreement(self, all_results):
        """Analyze how often the best validation strategy would have been the best test strategy."""
        
        print(f"\n🔍 VALIDATION vs TEST SET AGREEMENT ANALYSIS")
        print("=" * 80)
        
        successful_configs = [(desc, results) for desc, results in all_results.items() 
                             if results['success']]
        
        if not successful_configs:
            print("No successful configurations to analyze")
            return
        
        agreement_count = 0
        total_configs = 0
        
        for alpha_desc, results in successful_configs:
            # Filter out metadata keys
            strategy_results = {k: v for k, v in results.items() 
                              if k not in ['time', 'success', 'error']}
            successful_strategies = [(name, result) for name, result in strategy_results.items() 
                                   if isinstance(result, dict) and result.get('correlation_val') is not None]
            
            if successful_strategies:
                total_configs += 1
                best_val_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_val'])
                best_test_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_test'])
                
                if best_val_strategy[0] == best_test_strategy[0]:
                    agreement_count += 1
                    print(f"✅ {alpha_desc}: Validation best ({best_val_strategy[0]}) matches test best ({best_test_strategy[0]})")
                else:
                    print(f"⚠️  {alpha_desc}: Validation best ({best_val_strategy[0]}, corr={best_val_strategy[1]['correlation_val']:.4f}) differs from test best ({best_test_strategy[0]}, corr={best_test_strategy[1]['correlation_test']:.4f})")
        
        if total_configs > 0:
            agreement_rate = agreement_count / total_configs
            print(f"\n📊 AGREEMENT SUMMARY:")
            print(f"   Total configurations: {total_configs}")
            print(f"   Agreements: {agreement_count}")
            print(f"   Disagreements: {total_configs - agreement_count}")
            print(f"   Agreement rate: {agreement_rate:.2%}")
            
            if agreement_rate >= 0.75:
                print(f"   🎉 High agreement! Validation set is predictive of test set performance.")
            elif agreement_rate >= 0.5:
                print(f"   🤔 Moderate agreement. Consider using validation set with caution.")
            else:
                print(f"   ⚠️  Low agreement! Validation set may not be predictive of test set performance.")
        else:
            print("No successful configurations to analyze")

    def _evaluate_built_metrics_on_dataset(self, autometrics, dataset, metric_instances):
        """
        Evaluate already-built metric instances on a dataset using the same smart parallelization
        logic as _evaluate_metrics_on_dataset.
        
        This method implements the same device_map="auto" detection and parallelization
        decision logic but works with already-built metric instances instead of classes.
        """
        if not metric_instances:
            print("[Alpha Test] No metrics to evaluate")
            return []
        
        # Convert to the format expected by the sequential/parallel methods
        metric_classes = [type(metric) for metric in metric_instances]
        valid_metrics = [(i, metric) for i, metric in enumerate(metric_instances)]
        
        # Check for device_map="auto" metrics that should be forced to sequential
        auto_device_map_metrics = []
        regular_metrics = []
        
        for i, metric in valid_metrics:
            # Check if this metric uses device_map="auto" by inspecting its attributes
            if hasattr(metric, 'device_map') and metric.device_map == "auto":
                auto_device_map_metrics.append((i, metric))
                print(f"  🔄 {metric.get_name()} uses device_map='auto' - will be evaluated sequentially")
            elif hasattr(metric, 'load_kwargs') and isinstance(metric.load_kwargs, dict):
                if metric.load_kwargs.get('device_map') == "auto":
                    auto_device_map_metrics.append((i, metric))
                    print(f"  🔄 {metric.get_name()} uses load_kwargs device_map='auto' - will be evaluated sequentially")
            else:
                regular_metrics.append((i, metric))
        
        successful_metrics = []
        
        # Phase 1: Evaluate regular metrics in parallel (if enabled and we have multiple)
        if autometrics.enable_parallel_evaluation and len(regular_metrics) > 1:
            print(f"[Alpha Test] Evaluating {len(regular_metrics)} regular metrics using parallel execution...")
            successful_metrics.extend(autometrics._evaluate_metrics_parallel(dataset, metric_classes, regular_metrics))
        elif len(regular_metrics) > 0:
            print(f"[Alpha Test] Evaluating {len(regular_metrics)} regular metrics using sequential execution...")
            successful_metrics.extend(autometrics._evaluate_metrics_sequential(dataset, metric_classes, regular_metrics))
        
        # Phase 2: Evaluate device_map="auto" metrics sequentially (if any)
        if len(auto_device_map_metrics) > 0:
            print(f"[Alpha Test] Evaluating {len(auto_device_map_metrics)} device_map='auto' metrics sequentially...")
            print(f"[Alpha Test] device_map='auto' metrics: {[metric.get_name() for _, metric in auto_device_map_metrics]}")
            successful_metrics.extend(autometrics._evaluate_metrics_sequential(dataset, metric_classes, auto_device_map_metrics))
        
        return successful_metrics

    def _get_metrics_available_on_all_datasets(self, autometrics, train_dataset, val_dataset, test_dataset, successful_metric_instances):
        """
        Filter metrics to only include those that are available on all datasets (train, val, test).
        This prevents the regression model from depending on metrics that fail on any dataset.
        """
        print(f"[Alpha Test] Filtering metrics to ensure availability on all datasets...")
        print(f"[Alpha Test] Starting with {len(successful_metric_instances)} metrics")
        
        # Evaluate metrics on all datasets with error handling
        train_metrics = []
        val_metrics = []
        test_metrics = []
        
        try:
            train_metrics = self._evaluate_built_metrics_on_dataset(autometrics, train_dataset, successful_metric_instances)
            print(f"[Alpha Test] Training evaluation: {len(train_metrics)} metrics")
        except Exception as e:
            print(f"[Alpha Test] ❌ Training evaluation failed: {e}")
        
        try:
            val_metrics = self._evaluate_built_metrics_on_dataset(autometrics, val_dataset, successful_metric_instances)
            print(f"[Alpha Test] Validation evaluation: {len(val_metrics)} metrics")
        except Exception as e:
            print(f"[Alpha Test] ❌ Validation evaluation failed: {e}")
        
        try:
            test_metrics = self._evaluate_built_metrics_on_dataset(autometrics, test_dataset, successful_metric_instances)
            print(f"[Alpha Test] Test evaluation: {len(test_metrics)} metrics")
        except Exception as e:
            print(f"[Alpha Test] ❌ Test evaluation failed: {e}")
        
        print(f"[Alpha Test] Successful evaluations:")
        print(f"  - Training: {len(train_metrics)} metrics")
        print(f"  - Validation: {len(val_metrics)} metrics") 
        print(f"  - Test: {len(test_metrics)} metrics")
        
        # Get metric names that succeeded on all datasets
        train_names = {metric.get_name() for metric in train_metrics}
        val_names = {metric.get_name() for metric in val_metrics}
        test_names = {metric.get_name() for metric in test_metrics}
        
        # Find intersection
        common_names = train_names & val_names & test_names
        print(f"[Alpha Test] Metrics available on all datasets: {len(common_names)}")
        
        if len(common_names) == 0:
            print(f"[Alpha Test] ⚠️  No metrics available on all datasets!")
            print(f"[Alpha Test] Training metrics: {sorted(train_names)}")
            print(f"[Alpha Test] Validation metrics: {sorted(val_names)}")
            print(f"[Alpha Test] Test metrics: {sorted(test_names)}")
            
            # Fallback: use metrics that work on at least training and validation
            fallback_names = train_names & val_names
            if len(fallback_names) > 0:
                print(f"[Alpha Test] 🛠️  Fallback: using {len(fallback_names)} metrics available on train+val")
                common_names = fallback_names
            else:
                print(f"[Alpha Test] ❌ No fallback available - no metrics work on train+val")
        
        # Filter original metric instances to only include those in the intersection
        filtered_metrics = []
        for metric in successful_metric_instances:
            if metric.get_name() in common_names:
                filtered_metrics.append(metric)
        
        print(f"[Alpha Test] Filtered to {len(filtered_metrics)} metrics available on all datasets")
        if len(filtered_metrics) < len(successful_metric_instances):
            missing_metrics = set(m.get_name() for m in successful_metric_instances) - common_names
            print(f"[Alpha Test] Missing metrics: {missing_metrics}")
        
        return filtered_metrics

    def _debug_metric_failures(self, autometrics, train_dataset, val_dataset, test_dataset, successful_metric_instances):
        """
        Debug method to understand which metrics are failing on which datasets and why.
        """
        print(f"\n🔍 DEBUGGING METRIC FAILURES")
        print("=" * 60)
        
        datasets = {
            'train': train_dataset,
            'val': val_dataset, 
            'test': test_dataset
        }
        
        results = {}
        for dataset_name, dataset in datasets.items():
            print(f"\n📊 Evaluating on {dataset_name} dataset...")
            try:
                successful_metrics = self._evaluate_built_metrics_on_dataset(autometrics, dataset, successful_metric_instances)
                results[dataset_name] = {
                    'successful': [m.get_name() for m in successful_metrics],
                    'count': len(successful_metrics)
                }
                print(f"   ✅ {len(successful_metrics)} metrics succeeded")
            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                results[dataset_name] = {
                    'successful': [],
                    'count': 0,
                    'error': str(e)
                }
        
        # Find metrics that failed on any dataset
        all_metric_names = {m.get_name() for m in successful_metric_instances}
        failed_metrics = {}
        
        for metric_name in all_metric_names:
            failed_on = []
            for dataset_name, result in results.items():
                if metric_name not in result['successful']:
                    failed_on.append(dataset_name)
            if failed_on:
                failed_metrics[metric_name] = failed_on
        
        if failed_metrics:
            print(f"\n❌ METRICS THAT FAILED ON SOME DATASETS:")
            for metric_name, failed_datasets in failed_metrics.items():
                print(f"   {metric_name}: failed on {failed_datasets}")
        else:
            print(f"\n✅ All metrics succeeded on all datasets!")
        
        # Show intersection
        successful_on_all = set.intersection(*[set(result['successful']) for result in results.values() if 'successful' in result])
        print(f"\n📈 METRICS AVAILABLE ON ALL DATASETS: {len(successful_on_all)}")
        if successful_on_all:
            print(f"   {sorted(successful_on_all)}")
        
        return results

    def _create_scatter_plots(self, all_results, val_dataset, test_dataset, target_measure):
        """Create scatter plots for the best configuration from each alpha test."""
        
        # Create temp_alpha directory if it doesn't exist
        temp_dir = "temp_alpha"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a safe filename based on dataset and target
        safe_dataset_name = self.config.dataset_name.replace(" ", "_").replace("/", "_")
        safe_target_name = target_measure.replace(" ", "_").replace("/", "_")
        timestamp = int(time.time())
        
        # Find the best configuration for each alpha test
        best_configs = {}
        for alpha_desc, results in all_results.items():
            if results['success']:
                # Filter out metadata keys
                strategy_results = {k: v for k, v in results.items() 
                                  if k not in ['time', 'success', 'error']}
                successful_strategies = [(name, result) for name, result in strategy_results.items() 
                                       if isinstance(result, dict) and result.get('correlation_val') is not None]
                if successful_strategies:
                    best_strategy = max(successful_strategies, key=lambda x: x[1]['correlation_val'])
                    best_configs[alpha_desc] = {
                        'strategy': best_strategy[0],
                        'result': best_strategy[1]
                    }
        
        if not best_configs:
            print("No successful configurations to plot")
            return
        
        # Create subplots
        n_configs = len(best_configs)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        val_df = val_dataset.get_dataframe()
        test_df = test_dataset.get_dataframe()
        
        for i, (alpha_desc, config) in enumerate(best_configs.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            strategy_name = config['strategy']
            regression_column = config['result'].get('regression_column')
            
            # Verify the regression column exists in the dataframe
            if not regression_column or regression_column not in val_df.columns:
                ax.text(0.5, 0.5, f"Regression column '{regression_column}' not found for {strategy_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{alpha_desc}\n{strategy_name} - Column Not Found")
                continue
            
            # Get true and predicted values using the exact column name
            y_true_val = val_df[target_measure]
            y_pred_val = val_df[regression_column]
            
            # Filter out NaN/inf
            valid_mask_val = ~(np.isnan(y_true_val) | np.isnan(y_pred_val) | np.isinf(y_true_val) | np.isinf(y_pred_val))
            y_true_valid_val = y_true_val[valid_mask_val]
            y_pred_valid_val = y_pred_val[valid_mask_val]
            
            if len(y_true_valid_val) == 0:
                ax.text(0.5, 0.5, f"No valid data for {strategy_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{alpha_desc}\n{strategy_name} - No Valid Data")
                continue
            
            # Create scatter plot for validation
            ax.scatter(y_true_valid_val, y_pred_valid_val, alpha=0.6, s=20, label='Validation', color='blue')
            
            # Get test data
            y_true_test = test_df[target_measure]
            y_pred_test = test_df[regression_column]

            valid_mask_test = ~(np.isnan(y_true_test) | np.isnan(y_pred_test) | np.isinf(y_true_test) | np.isinf(y_pred_test))
            y_true_valid_test = y_true_test[valid_mask_test]
            y_pred_valid_test = y_pred_test[valid_mask_test]

            if len(y_true_valid_test) > 0:
                # Create scatter plot for test
                ax.scatter(y_true_valid_test, y_pred_valid_test, alpha=0.6, s=20, label='Test', color='orange')
            
            # Add perfect prediction line
            all_y_true = np.concatenate([y_true_valid_val, y_true_valid_test]) if len(y_true_valid_test) > 0 else y_true_valid_val
            all_y_pred = np.concatenate([y_pred_valid_val, y_pred_valid_test]) if len(y_pred_valid_test) > 0 else y_pred_valid_val
            
            min_val = min(all_y_true.min(), all_y_pred.min())
            max_val = max(all_y_true.max(), all_y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            # Add correlation and R² to title
            correlation_val = config['result']['correlation_val']
            r2_val = config['result']['r2_val']
            correlation_test = config['result']['correlation_test']
            r2_test = config['result']['r2_test']
            
            title = f"{alpha_desc}\n{strategy_name}\nVal: Corr={correlation_val:.4f}, R²={r2_val:.4f}\nTest: Corr={correlation_test:.4f}, R²={r2_test:.4f}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f'True {target_measure}')
            ax.set_ylabel(f'Predicted {target_measure}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(best_configs), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot with descriptive filename
        filename = f"{safe_dataset_name}_{safe_target_name}_{timestamp}_alpha_comparison.png"
        filepath = os.path.join(temp_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Scatter plots saved to: {filepath}")
        print(f"   Shows best configuration from each alpha test")
        print(f"   Red dashed line = perfect prediction")
        
        return filepath


def run_alpha_comparison_experiment(config: ExperimentConfig):
    """Run an alpha comparison experiment."""
    
    # Configure LLMs
    generator_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    judge_llm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create tester and run experiment
    tester = AlphaComparisonTester(config)
    results = tester.run_alpha_comparison(generator_llm, judge_llm)
    
    return results

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Choose dataset
    import argparse
    parser = argparse.ArgumentParser(description="Run alpha comparison experiment")
    parser.add_argument("--dataset", choices=["SimpDA", "EvalGenProduct", "CoGymTravelOutcome", "HelpSteer"], 
                       default="SimpDA", help="Dataset to use")
    parser.add_argument("--target-measure", default=None, 
                       help="Target measure (auto-detected if not specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--regenerate-metrics", action="store_true", 
                       help="Force regeneration of metrics")
    
    args = parser.parse_args()
    
    # Auto-detect target measure based on dataset
    if args.target_measure is None:
        if args.dataset == "SimpDA":
            target_measure = "simplicity"
        elif args.dataset == "EvalGenProduct":
            target_measure = "grade"
        elif args.dataset == "CoGymTravelOutcome":
            target_measure = "outcomeRating"
        elif args.dataset == "HelpSteer":
            target_measure = "helpfulness"
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    else:
        target_measure = args.target_measure
    
    # Configure experiment
    config = ExperimentConfig(
        dataset_name=args.dataset,
        target_measure=target_measure,
        num_to_retrieve=30,  # Default Autometrics setting
        num_to_regress=5,    # Default Autometrics setting
        seed=args.seed,
        regenerate_metrics=args.regenerate_metrics
    )
    
    print(f"🔍 Running alpha comparison experiment:")
    print(f"   Dataset: {config.dataset_name}")
    print(f"   Target Measure: {config.target_measure}")
    print(f"   Seed: {config.seed}")
    print(f"   Regenerate Metrics: {config.regenerate_metrics}")
    
    # Run alpha comparison experiment
    results = run_alpha_comparison_experiment(config)
    
    print(f"\n✅ Alpha comparison experiment completed!")
    print("Check the results above to find the best alpha configuration for your use case.") 