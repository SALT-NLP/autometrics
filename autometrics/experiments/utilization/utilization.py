#!/usr/bin/env python3
# File: utilization/utilization.py

import os
import time
import statistics
import psutil
import random
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import traceback
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    HAS_PYNVML = True
    pynvml.nvmlInit()
except (ImportError, pynvml.NVMLError):
    HAS_PYNVML = False

try:
    import nltk
    nltk.download('words')
    from nltk.corpus import words as nltk_words
    NLTK_WORDS = set(w.lower() for w in nltk_words.words() if 3 <= len(w) <= 10 and w.isalpha())
    if not NLTK_WORDS:  # Fallback if NLTK data not downloaded
        raise ImportError("NLTK words not available")
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from autometrics.experiments.experiment import Experiment
from autometrics.experiments.results import TabularResult, JSONResult, FigureResult
from autometrics.metrics.Metric import Metric


class ResourceTracker:
    """Tracks and records resource usage during metric execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.end_time = None
        self.start_memory_usage = 0
        self.peak_memory_usage = 0
        self.start_gpu_memory = 0
        self.peak_gpu_memory = 0
        self.disk_before = None
        self.disk_after = None
        self.device_count = 0
        
        if HAS_TORCH and torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            torch.cuda.reset_peak_memory_stats()
        
        if HAS_PYNVML:
            try:
                self.device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                pass
    
    def start(self):
        """Start tracking resources."""
        # Capture baseline memory usage
        self.start_memory_usage = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        self.peak_memory_usage = self.start_memory_usage
        
        # Capture baseline GPU memory
        self.start_gpu_memory = 0
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            try:
                self.start_gpu_memory = sum([
                    torch.cuda.memory_allocated(i) / (1024 * 1024) for i in range(self.device_count)
                ]) if self.device_count > 0 else 0
            except Exception:
                pass
        elif HAS_PYNVML:
            try:
                total_memory = 0
                for i in range(self.device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory += info.used / (1024 * 1024)
                self.start_gpu_memory = total_memory
            except Exception:
                pass
                
        self.disk_before = psutil.disk_usage('/').used
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop tracking resources and record final metrics."""
        self.end_time = time.time()
        current_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory_usage = current_memory
        
        # Get GPU memory if available - SUM across all GPUs
        current_gpu_memory = 0
        if HAS_TORCH and torch.cuda.is_available():
            try:
                current_gpu_memory = sum(
                    [torch.cuda.max_memory_allocated(i) / (1024 * 1024) for i in range(self.device_count)]
                ) if self.device_count > 0 else 0
                self.peak_gpu_memory = current_gpu_memory
            except Exception:
                pass
        elif HAS_PYNVML:
            try:
                total_memory = 0
                for i in range(self.device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory += info.used / (1024 * 1024)
                self.peak_gpu_memory = total_memory
            except Exception:
                pass
                
        self.disk_after = psutil.disk_usage('/').used
        return self
    
    def get_results(self):
        """Return the recorded metrics as a dictionary."""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        # Convert duration to milliseconds
        duration_ms = duration * 1000
        
        # Calculate incremental RAM usage (what was added during the metric execution)
        incremental_ram = max(0, self.peak_memory_usage - self.start_memory_usage)
        
        # Calculate incremental GPU memory usage
        incremental_gpu = max(0, self.peak_gpu_memory - self.start_gpu_memory)
        
        disk_delta = (self.disk_after - self.disk_before) / (1024 * 1024) if self.disk_after and self.disk_before else 0
        
        return {
            'duration_milliseconds': duration_ms,
            'cpu_ram_mb': incremental_ram,
            'gpu_ram_mb': incremental_gpu,
            'disk_usage_change_mb': disk_delta,
            'baseline_cpu_ram_mb': self.start_memory_usage,
            'baseline_gpu_ram_mb': self.start_gpu_memory,
            'total_cpu_ram_mb': self.peak_memory_usage,
            'total_gpu_ram_mb': self.peak_gpu_memory
        }


@contextmanager
def track_resources():
    """Context manager for tracking resource usage."""
    tracker = ResourceTracker().start()
    try:
        yield tracker
    finally:
        tracker.stop()


def generate_synthetic_text(length_category: str) -> Tuple[str, str, List[str]]:
    """Generate synthetic text samples for testing.
    
    Args:
        length_category: "short", "medium", or "long"
        
    Returns:
        Tuple containing (input_text, output_text, reference_texts)
    """
    if HAS_NLTK:
        # Use NLTK words if available
        vocab = list(NLTK_WORDS)
        if len(vocab) > 1000:  # Limit to a reasonable subset for efficiency
            vocab = random.sample(vocab, 1000)
    else:
        # Fallback vocabulary
        vocab = [
            "apple", "banana", "cat", "dog", "elephant", "frog", "guitar", 
            "house", "igloo", "jungle", "koala", "lemon", "mango", "night",
            "orange", "penguin", "queen", "rabbit", "strawberry", "tiger",
            "umbrella", "violet", "whale", "xylophone", "yellow", "zebra",
            "book", "chair", "desk", "ear", "flower", "garden", "hat", 
            "island", "jacket", "key", "lamp", "mountain", "notebook", 
            "ocean", "phone", "quilt", "river", "sun", "tree", "university",
            "village", "window", "xerox", "yard", "zebra", "airplane", 
            "bicycle", "computer", "dictionary", "engine", "forest"
        ]
    
    if length_category == "short":
        input_len = random.randint(3, 10)
        output_len = random.randint(3, 10)
        ref_count = random.randint(1, 3)
        ref_lens = [random.randint(3, 10) for _ in range(ref_count)]
    elif length_category == "medium":
        input_len = random.randint(80, 120)
        output_len = random.randint(80, 120)
        ref_count = random.randint(1, 3)
        ref_lens = [random.randint(80, 120) for _ in range(ref_count)]
    else:  # long
        input_len = random.randint(800, 1200)
        output_len = random.randint(800, 1200)
        ref_count = random.randint(1, 3)
        ref_lens = [random.randint(800, 1200) for _ in range(ref_count)]
    
    input_text = " ".join(random.choices(vocab, k=input_len))
    output_text = " ".join(random.choices(vocab, k=output_len))
    reference_texts = [" ".join(random.choices(vocab, k=ref_len)) for ref_len in ref_lens]
    
    return input_text, output_text, reference_texts


def measure_current_memory():
    """Get the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    gpu_memory = 0
    if HAS_TORCH and torch.cuda.is_available():
        try:
            gpu_memory = sum([
                torch.cuda.memory_allocated(i) / (1024 * 1024) 
                for i in range(torch.cuda.device_count())
            ]) if torch.cuda.device_count() > 0 else 0
        except Exception:
            pass
    elif HAS_PYNVML:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory += info.used / (1024 * 1024)
        except Exception:
            pass
            
    return {
        'cpu_ram_mb': cpu_memory,
        'gpu_ram_mb': gpu_memory
    }


class UtilizationExperiment(Experiment):
    """Experiment class to measure resource utilization of metrics."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        metrics: List[Metric], 
        output_dir: str,
        dataset=None,
        num_examples: int = 30,
        num_burn_in: int = 5,
        lengths: List[str] = None,
        use_synthetic: bool = True,
        seed: int = 42,
        **kwargs
    ):
        # Measure initial memory footprint before loading anything
        self.initial_memory = measure_current_memory()
        
        # Create a minimal dataset if none provided for the parent constructor
        if dataset is None and use_synthetic:
            from autometrics.dataset.Dataset import Dataset
            class MinimalDataset(Dataset):
                def __init__(self): pass
                def get_splits(self, seed=42): return self, self, self
            dataset = MinimalDataset()
            
        # Update output directory based on synthetic vs real data
        self.use_synthetic = use_synthetic  # Store for potential future use
        if use_synthetic:
            output_dir = os.path.join(output_dir, "synthetic")
            
        super().__init__(name, description, metrics, output_dir, dataset, seed, False, **kwargs)
        
        self.num_examples = num_examples
        self.num_burn_in = num_burn_in
        self.lengths = lengths or ["short", "medium", "long"]
        self.results = {}
        
    def run(self, print_results: bool = False):
        """Run the utilization experiment."""
        if print_results:
            print(f"Running utilization experiment with {len(self.metrics)} metrics")
            if self.use_synthetic:
                print(f"Testing {self.num_examples} synthetic examples per length category: {', '.join(self.lengths)}")
            else:
                print(f"Testing using real data from provided dataset")
            
            # Print initial memory footprint
            print(f"Initial memory footprint (before experiment):")
            print(f"  CPU RAM: {self.initial_memory['cpu_ram_mb']:.2f} MB")
            print(f"  GPU RAM: {self.initial_memory['gpu_ram_mb']:.2f} MB")
        
        # Store all results
        all_results = {
            "metrics": [],
            "results_by_metric": {},
            "initial_memory": self.initial_memory
        }
        
        # For real dataset, we don't use length categories
        categories_to_test = self.lengths if self.use_synthetic else ["dataset"]
        
        # Run experiment for each metric
        for metric in self.metrics:
            metric_name = metric.get_name()
            if print_results:
                print(f"Testing metric: {metric_name}")
            
            # Disable cache for this metric if possible
            try:
                original_cache_state = metric.use_cache
                metric.use_cache = False
            except AttributeError:
                pass
            
            metric_results = {"name": metric_name, "results": {}}
            
            # For each length category (if synthetic) or just once (if real data)
            for category in categories_to_test:
                if print_results:
                    if self.use_synthetic:
                        print(f"  Testing with {category} inputs/outputs")
                    else:
                        print(f"  Testing with real dataset examples")
                
                # Get test examples
                test_examples = self._get_test_examples(category)
                length_results = []
                
                # Track metadata for results
                result_prefix = f"{metric_name}/{category}" if self.use_synthetic else f"{metric_name}"
                
                # Check if this metric has model loading/unloading methods
                has_model_methods = hasattr(metric, '_load_model') and hasattr(metric, '_unload_model')
                model_was_loaded = False
                
                # If the metric has load/unload methods, check if model is already loaded
                if has_model_methods:
                    # Store whether model was loaded before we unload it
                    model_was_loaded = hasattr(metric, 'model') and metric.model is not None
                    # Unload model to ensure we measure full memory usage
                    try:
                        metric._unload_model()
                    except Exception as e:
                        if print_results:
                            print(f"  ⚠️ Warning: Failed to unload model: {str(e)}")

                    try:
                        metric._load_model()
                    except Exception as e:
                        if print_results:
                            print(f"  ⚠️ Warning: Failed to load model: {str(e)}")
                
                # Perform burn-in samples first
                for i in range(min(self.num_burn_in, len(test_examples))):
                    input_text, output_text, reference_texts = test_examples[i]
                    try:
                        metric.calculate(input_text, output_text, reference_texts)
                    except Exception as e:
                        if print_results:
                            print(f"  ⚠️ Error during burn-in: {str(e)}")
                
                # Now run the actual experiments
                for i in range(min(self.num_examples, len(test_examples))):
                    input_text, output_text, reference_texts = test_examples[i]
                    
                    try:
                        with track_resources() as tracker:
                            result = metric.calculate(input_text, output_text, reference_texts)
                        
                        resources = tracker.get_results()
                        length_results.append(resources)
                    except Exception as e:
                        if print_results:
                            print(f"  ⚠️ Error on example {i}: {str(e)}")
                            traceback.print_exc()
                
                # Calculate statistics
                if length_results:
                    stats = self._calculate_statistics(length_results)
                    metric_results["results"][category] = {
                        "raw_data": length_results,
                        "summary": stats
                    }
                    
                    # Save raw data
                    raw_df = pd.DataFrame(length_results)
                    raw_df.insert(0, 'example_id', range(len(raw_df)))
                    self.results[f"{result_prefix}/raw_data"] = TabularResult(raw_df)
                    
                    # Save summary
                    summary_df = pd.DataFrame([stats])
                    self.results[f"{result_prefix}/summary"] = TabularResult(summary_df)
                    
                    # Generate plots
                    self._generate_resource_plots(raw_df, result_prefix)
            
            # Restore cache setting
            try:
                metric.use_cache = original_cache_state
            except AttributeError:
                pass

            # Unload the model if it was not loaded initially and has unload methods
            if not model_was_loaded and has_model_methods:
                try:
                    metric._unload_model()
                except Exception as e:
                    if print_results:
                        print(f"  ⚠️ Warning: Failed to unload model: {str(e)}")
            
            all_results["metrics"].append(metric_name)
            all_results["results_by_metric"][metric_name] = metric_results
        
        # Save combined results if we have multiple metrics
        if len(self.metrics) > 1:
            self._generate_comparison_plots(all_results)
        
        # Save full results JSON
        self.results["full_results"] = JSONResult(all_results)
        
        if print_results:
            print("\nUtilization Experiment Results Summary:")
            for metric_name in all_results["metrics"]:
                for category, data in all_results["results_by_metric"][metric_name]["results"].items():
                    summary = data["summary"]
                    print(f"\n{metric_name} - {category}:")
                    print(f"  Duration (ms): {summary['mean_duration_milliseconds']:.2f} " +
                          f"±{summary['ci_upper_duration'] - summary['mean_duration_milliseconds']:.2f}")
                    
                    # Print both incremental and total memory usage
                    incremental_ram = summary['mean_cpu_ram_mb']
                    baseline_ram = summary['mean_baseline_cpu_ram_mb']
                    total_ram = summary['mean_total_cpu_ram_mb']
                    
                    print(f"  CPU RAM - Baseline: {baseline_ram:.2f} MB")
                    print(f"  CPU RAM - Used by metric: {incremental_ram:.2f} MB " +
                          f"±{summary['ci_upper_cpu_ram'] - summary['mean_cpu_ram_mb']:.2f}")
                    print(f"  CPU RAM - Total: {total_ram:.2f} MB")
                    
                    if summary['mean_gpu_ram_mb'] > 0:
                        print(f"  GPU RAM (MB): {summary['mean_gpu_ram_mb']:.2f} " +
                              f"±{summary['ci_upper_gpu_ram'] - summary['mean_gpu_ram_mb']:.2f}")
    
    def _get_test_examples(self, category):
        """Get test examples for the experiment.
        
        Args:
            category: The category name ("short", "medium", "long", or "dataset")
            
        Returns:
            List of (input_text, output_text, reference_texts) tuples
        """
        examples = []
        
        if self.use_synthetic:
            # Generate synthetic examples based on length category
            for _ in range(self.num_examples + self.num_burn_in):
                examples.append(generate_synthetic_text(category))
        else:
            # Use examples from dataset without filtering by length
            df = self.test_dataset.get_dataframe()
            
            # If dataset is too large, take a subset
            if len(df) > (self.num_examples + self.num_burn_in):
                test_dataset = self.test_dataset.get_subset(self.num_examples + self.num_burn_in, self.seed)
                # Use random seed for consistent subsetting
                df = test_dataset.get_dataframe()
            
            # Convert each row to the expected format
            for _, row in df.iterrows():
                input_text = row[self.test_dataset.get_input_column()]
                output_text = row[self.test_dataset.get_output_column()]
                references = [row[ref_col] for ref_col in self.test_dataset.get_reference_columns()]
                examples.append((input_text, output_text, references))
        
        return examples
    
    def _calculate_statistics(self, results):
        """Calculate statistics from a list of resource measurements."""
        durations = [r["duration_milliseconds"] for r in results]
        cpu_ram = [r["cpu_ram_mb"] for r in results]
        gpu_ram = [r["gpu_ram_mb"] for r in results]
        disk_usage = [r["disk_usage_change_mb"] for r in results]
        baseline_cpu_ram = [r["baseline_cpu_ram_mb"] for r in results]
        baseline_gpu_ram = [r["baseline_gpu_ram_mb"] for r in results]
        total_cpu_ram = [r["total_cpu_ram_mb"] for r in results]
        total_gpu_ram = [r["total_gpu_ram_mb"] for r in results]
        
        mean_duration = statistics.mean(durations)
        mean_cpu_ram = statistics.mean(cpu_ram)
        mean_gpu_ram = statistics.mean(gpu_ram)
        mean_disk = statistics.mean(disk_usage)
        mean_baseline_cpu_ram = statistics.mean(baseline_cpu_ram)
        mean_baseline_gpu_ram = statistics.mean(baseline_gpu_ram)
        mean_total_cpu_ram = statistics.mean(total_cpu_ram)
        mean_total_gpu_ram = statistics.mean(total_gpu_ram)
        
        # Calculate 95% confidence intervals
        if len(durations) > 1:
            std_duration = statistics.stdev(durations)
            std_cpu_ram = statistics.stdev(cpu_ram)
            std_gpu_ram = statistics.stdev(gpu_ram)
            std_disk = statistics.stdev(disk_usage)
            std_baseline_cpu_ram = statistics.stdev(baseline_cpu_ram)
            std_baseline_gpu_ram = statistics.stdev(baseline_gpu_ram)
            std_total_cpu_ram = statistics.stdev(total_cpu_ram)
            std_total_gpu_ram = statistics.stdev(total_gpu_ram)
            
            n = len(durations)
            # Use t-distribution with n-1 degrees of freedom for small samples
            # For simplicity, approximating with 1.96 * std/sqrt(n) for 95% CI
            margin_duration = 1.96 * std_duration / (n ** 0.5)
            margin_cpu_ram = 1.96 * std_cpu_ram / (n ** 0.5)
            margin_gpu_ram = 1.96 * std_gpu_ram / (n ** 0.5)
            margin_disk = 1.96 * std_disk / (n ** 0.5)
            margin_baseline_cpu_ram = 1.96 * std_baseline_cpu_ram / (n ** 0.5)
            margin_baseline_gpu_ram = 1.96 * std_baseline_gpu_ram / (n ** 0.5)
            margin_total_cpu_ram = 1.96 * std_total_cpu_ram / (n ** 0.5)
            margin_total_gpu_ram = 1.96 * std_total_gpu_ram / (n ** 0.5)
        else:
            margin_duration = margin_cpu_ram = margin_gpu_ram = margin_disk = 0
            margin_baseline_cpu_ram = margin_baseline_gpu_ram = margin_total_cpu_ram = margin_total_gpu_ram = 0
        
        return {
            "mean_duration_milliseconds": mean_duration,
            "ci_lower_duration": max(0, mean_duration - margin_duration),
            "ci_upper_duration": mean_duration + margin_duration,
            
            "mean_cpu_ram_mb": mean_cpu_ram,
            "ci_lower_cpu_ram": max(0, mean_cpu_ram - margin_cpu_ram),
            "ci_upper_cpu_ram": mean_cpu_ram + margin_cpu_ram,
            
            "mean_gpu_ram_mb": mean_gpu_ram,
            "ci_lower_gpu_ram": max(0, mean_gpu_ram - margin_gpu_ram),
            "ci_upper_gpu_ram": mean_gpu_ram + margin_gpu_ram,
            
            "mean_disk_usage_change_mb": mean_disk,
            "ci_lower_disk": mean_disk - margin_disk,
            "ci_upper_disk": mean_disk + margin_disk,
            
            "mean_baseline_cpu_ram_mb": mean_baseline_cpu_ram,
            "ci_lower_baseline_cpu_ram": max(0, mean_baseline_cpu_ram - margin_baseline_cpu_ram),
            "ci_upper_baseline_cpu_ram": mean_baseline_cpu_ram + margin_baseline_cpu_ram,
            
            "mean_baseline_gpu_ram_mb": mean_baseline_gpu_ram,
            "ci_lower_baseline_gpu_ram": max(0, mean_baseline_gpu_ram - margin_baseline_gpu_ram),
            "ci_upper_baseline_gpu_ram": mean_baseline_gpu_ram + margin_baseline_gpu_ram,
            
            "mean_total_cpu_ram_mb": mean_total_cpu_ram,
            "ci_lower_total_cpu_ram": max(0, mean_total_cpu_ram - margin_total_cpu_ram),
            "ci_upper_total_cpu_ram": mean_total_cpu_ram + margin_total_cpu_ram,
            
            "mean_total_gpu_ram_mb": mean_total_gpu_ram,
            "ci_lower_total_gpu_ram": max(0, mean_total_gpu_ram - margin_total_gpu_ram),
            "ci_upper_total_gpu_ram": mean_total_gpu_ram + margin_total_gpu_ram,
        }
    
    def _generate_resource_plots(self, df, prefix):
        """Generate plots for a single metric/category."""
        # Time series plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = df['example_id']
        ax.plot(x, df['duration_milliseconds'], 'o-', label='Duration (ms)')
        ax.set_xlabel('Example')
        ax.set_ylabel('Duration (milliseconds)')
        ax.set_title('Runtime Performance')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.results[f"{prefix}/duration_timeseries"] = FigureResult(fig)
        
        # Memory usage plot - incremental vs baseline
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the baseline memory (which should be constant across runs)
        baseline_memory = df['baseline_cpu_ram_mb'].mean()
        
        # Plot stacked bar chart to show baseline and incremental memory
        ax.bar(x, df['baseline_cpu_ram_mb'], label='Baseline RAM', alpha=0.5, color='lightgray')
        ax.bar(x, df['cpu_ram_mb'], bottom=df['baseline_cpu_ram_mb'], 
               label='Additional RAM used by metric', color='blue', alpha=0.7)
        
        # If GPU memory was used, plot it similarly
        if df['gpu_ram_mb'].sum() > 0:
            ax2 = ax.twinx()
            ax2.plot(x, df['gpu_ram_mb'], 'r--', label='GPU RAM')
            ax2.set_ylabel('GPU Memory (MB)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax.legend()
            
        ax.set_xlabel('Example')
        ax.set_ylabel('CPU Memory (MB)')
        ax.set_title('Memory Utilization')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.results[f"{prefix}/memory_timeseries"] = FigureResult(fig)
        
        # Histogram of duration - but only if we have enough different values
        fig, ax = plt.subplots(figsize=(10, 6))
        durations = df['duration_milliseconds'].values
        if len(durations) > 1 and not np.all(durations == durations[0]):
            # Only create a histogram if we have at least 2 different values
            bin_count = min(20, max(1, len(df) // 2))
            ax.hist(durations, bins=bin_count, alpha=0.7)
        else:
            # If all values are identical, just make a single bar
            ax.bar([durations[0]], [len(durations)], alpha=0.7, width=durations[0]*0.1 if durations[0] > 0 else 0.1)
        
        ax.axvline(df['duration_milliseconds'].mean(), color='red', linestyle='dashed', linewidth=2, 
                  label=f'Mean: {df["duration_milliseconds"].mean():.2f} ms')
        ax.set_xlabel('Duration (milliseconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Duration Distribution')
        ax.legend()
        fig.tight_layout()
        self.results[f"{prefix}/duration_histogram"] = FigureResult(fig)
        
        # Histogram of memory usage - but only if we have enough different values
        fig, ax = plt.subplots(figsize=(10, 6))
        memory_values = df['cpu_ram_mb'].values
        if len(memory_values) > 1 and not np.all(memory_values == memory_values[0]):
            # Only create a histogram if we have at least 2 different values
            bin_count = min(10, max(1, len(df) // 3))
            ax.hist(memory_values, bins=bin_count, alpha=0.7, 
                    label=f'Mean: {df["cpu_ram_mb"].mean():.2f} MB')
        else:
            # If all values are identical, just make a single bar
            ax.bar([memory_values[0]], [len(memory_values)], alpha=0.7, 
                   width=memory_values[0]*0.1 if memory_values[0] > 0 else 0.1,
                   label=f'Mean: {df["cpu_ram_mb"].mean():.2f} MB')
            
        ax.axvline(df['cpu_ram_mb'].mean(), color='red', linestyle='dashed', linewidth=2)
        ax.set_xlabel('CPU RAM Usage (MB)')
        ax.set_ylabel('Frequency')
        ax.set_title('Memory Usage Distribution')
        ax.legend()
        fig.tight_layout()
        self.results[f"{prefix}/memory_histogram"] = FigureResult(fig)
    
    def _generate_comparison_plots(self, all_results):
        """Generate comparison plots across all metrics."""
        if not self.use_synthetic:
            # For real datasets, create summary comparison across metrics
            summary_data = []
            for metric_name in all_results["metrics"]:
                data = all_results["results_by_metric"][metric_name]["results"]["dataset"]["summary"]
                summary_data.append({
                    "metric": metric_name,
                    "duration_ms": data["mean_duration_milliseconds"],
                    "duration_ci": data["ci_upper_duration"] - data["mean_duration_milliseconds"],
                    "cpu_ram_mb": data["mean_cpu_ram_mb"],  # This is now incremental RAM
                    "cpu_ram_ci": data["ci_upper_cpu_ram"] - data["mean_cpu_ram_mb"],
                    "gpu_ram_mb": data["mean_gpu_ram_mb"],
                    "gpu_ram_ci": data["ci_upper_gpu_ram"] - data["mean_gpu_ram_mb"],
                    "baseline_cpu_ram_mb": data.get("mean_baseline_cpu_ram_mb", 0),  # Include baseline
                })
            
            df = pd.DataFrame(summary_data)
            self.results["metric_comparison/summary"] = TabularResult(df)
            
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            metrics = df["metric"].values
            x_pos = np.arange(len(metrics))
            
            ax.bar(x_pos, df["duration_ms"], yerr=df["duration_ci"], capsize=10)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Duration (milliseconds)')
            ax.set_title('Average Runtime by Metric')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            fig.tight_layout()
            self.results["metric_comparison/duration"] = FigureResult(fig)
            
            # Box and whisker plot for timing
            if len(all_results["metrics"]) > 1:
                # Collect all timing data across all metrics
                timing_data = []
                metric_labels = []
                for metric_name in all_results["metrics"]:
                    raw_data = all_results["results_by_metric"][metric_name]["results"]["dataset"]["raw_data"]
                    durations = [d["duration_milliseconds"] for d in raw_data]
                    timing_data.append(durations)
                    metric_labels.append(metric_name)
                
                # Create box plot
                fig, ax = plt.subplots(figsize=(12, 8))
                box = ax.boxplot(timing_data, patch_artist=True, labels=metric_labels)
                
                # Add some styling
                for patch in box['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title('Runtime Distribution by Metric')
                ax.set_xticklabels(metric_labels, rotation=45, ha='right')
                plt.grid(True, axis='y', alpha=0.3)
                fig.tight_layout()
                self.results["metric_comparison/duration_boxplot"] = FigureResult(fig)
            
            # Memory comparison - stacked bar to show baseline and incremental
            fig, ax = plt.subplots(figsize=(12, 8))
            x_pos = np.arange(len(metrics))
            
            # Plot baseline memory
            ax.bar(x_pos, df["baseline_cpu_ram_mb"], label='Baseline RAM', alpha=0.5, color='lightgray')
            
            # Plot incremental memory on top
            ax.bar(x_pos, df["cpu_ram_mb"], bottom=df["baseline_cpu_ram_mb"],
                   yerr=df["cpu_ram_ci"], capsize=10, label='Incremental RAM', color='blue', alpha=0.7)
            
            # If any GPU memory was used, add it as a separate set of bars
            if df["gpu_ram_mb"].sum() > 0:
                ax2 = ax.twinx()
                ax2.bar(x_pos + 0.2, df["gpu_ram_mb"], width=0.3, yerr=df["gpu_ram_ci"], 
                        capsize=10, color='red', alpha=0.7, label='GPU RAM')
                ax2.set_ylabel('GPU Memory (MB)', color='r')
                
                # Combine legends
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left')
            else:
                ax.legend()
                
            ax.set_xlabel('Metrics')
            ax.set_ylabel('CPU Memory Usage (MB)')
            ax.set_title('Memory Usage by Metric')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            fig.tight_layout()
            self.results["metric_comparison/memory"] = FigureResult(fig)
        else:
            # For synthetic data, compare within each length category
            for length in self.lengths:
                summary_data = []
                for metric_name in all_results["metrics"]:
                    if length in all_results["results_by_metric"][metric_name]["results"]:
                        data = all_results["results_by_metric"][metric_name]["results"][length]["summary"]
                        summary_data.append({
                            "metric": metric_name,
                            "length": length,
                            "duration_ms": data["mean_duration_milliseconds"],
                            "duration_ci": data["ci_upper_duration"] - data["mean_duration_milliseconds"],
                            "cpu_ram_mb": data["mean_cpu_ram_mb"],
                            "cpu_ram_ci": data["ci_upper_cpu_ram"] - data["mean_cpu_ram_mb"],
                            "gpu_ram_mb": data["mean_gpu_ram_mb"],
                            "gpu_ram_ci": data["ci_upper_gpu_ram"] - data["mean_gpu_ram_mb"],
                        })
                
                if not summary_data:
                    continue
                    
                df = pd.DataFrame(summary_data)
                self.results[f"metric_comparison/{length}/summary"] = TabularResult(df)
                
                # Bar chart comparison - duration
                fig, ax = plt.subplots(figsize=(12, 8))
                metrics = df["metric"].values
                x_pos = np.arange(len(metrics))
                
                ax.bar(x_pos, df["duration_ms"], yerr=df["duration_ci"], capsize=10)
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title(f'Average Runtime by Metric ({length})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                fig.tight_layout()
                self.results[f"metric_comparison/{length}/duration"] = FigureResult(fig)
                
                # Box and whisker plot for timing
                if len(all_results["metrics"]) > 1:
                    # Collect all timing data across all metrics
                    timing_data = []
                    metric_labels = []
                    for metric_name in all_results["metrics"]:
                        if length in all_results["results_by_metric"][metric_name]["results"]:
                            raw_data = all_results["results_by_metric"][metric_name]["results"][length]["raw_data"]
                            durations = [d["duration_milliseconds"] for d in raw_data]
                            timing_data.append(durations)
                            metric_labels.append(metric_name)
                    
                    # Create box plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    box = ax.boxplot(timing_data, patch_artist=True, labels=metric_labels)
                    
                    # Add some styling
                    for patch in box['boxes']:
                        patch.set_facecolor('lightblue')
                    
                    ax.set_xlabel('Metrics')
                    ax.set_ylabel('Duration (milliseconds)')
                    ax.set_title(f'Runtime Distribution by Metric ({length})')
                    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
                    plt.grid(True, axis='y', alpha=0.3)
                    fig.tight_layout()
                    self.results[f"metric_comparison/{length}/duration_boxplot"] = FigureResult(fig)
                
                # Bar chart comparison - memory
                fig, ax = plt.subplots(figsize=(12, 8))
                width = 0.35
                
                ax.bar(x_pos - width/2, df["cpu_ram_mb"], width, yerr=df["cpu_ram_ci"], capsize=10, label='CPU RAM')
                if df["gpu_ram_mb"].sum() > 0:
                    ax.bar(x_pos + width/2, df["gpu_ram_mb"], width, yerr=df["gpu_ram_ci"], capsize=10, label='GPU RAM')
                    
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title(f'Average Memory Usage by Metric ({length})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend()
                fig.tight_layout()
                self.results[f"metric_comparison/{length}/memory"] = FigureResult(fig)
            
            # Also create a comparison across all lengths for each metric
            for metric_name in all_results["metrics"]:
                length_data = []
                metric_results = all_results["results_by_metric"][metric_name]["results"]
                
                for length in self.lengths:
                    if length in metric_results:
                        data = metric_results[length]["summary"]
                        length_data.append({
                            "metric": metric_name,
                            "length": length,
                            "duration_ms": data["mean_duration_milliseconds"],
                            "duration_ci": data["ci_upper_duration"] - data["mean_duration_milliseconds"],
                            "cpu_ram_mb": data["mean_cpu_ram_mb"],
                            "cpu_ram_ci": data["ci_upper_cpu_ram"] - data["mean_cpu_ram_mb"],
                            "gpu_ram_mb": data["mean_gpu_ram_mb"],
                            "gpu_ram_ci": data["ci_upper_gpu_ram"] - data["mean_gpu_ram_mb"],
                        })
                
                if not length_data:
                    continue
                    
                df = pd.DataFrame(length_data)
                self.results[f"length_comparison/{metric_name}/summary"] = TabularResult(df)
                
                # Bar chart comparison - duration vs length
                fig, ax = plt.subplots(figsize=(10, 6))
                lengths = df["length"].values
                x_pos = np.arange(len(lengths))
                
                ax.bar(x_pos, df["duration_ms"], yerr=df["duration_ci"], capsize=10)
                ax.set_xlabel('Text Length')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title(f'Runtime vs Text Length ({metric_name})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(lengths)
                fig.tight_layout()
                self.results[f"length_comparison/{metric_name}/duration"] = FigureResult(fig)
                
                # Box plot comparison - duration vs length
                # Collect timing data across all lengths
                timing_data = []
                length_labels = []
                for length in self.lengths:
                    if length in metric_results:
                        raw_data = metric_results[length]["raw_data"]
                        durations = [d["duration_milliseconds"] for d in raw_data]
                        timing_data.append(durations)
                        length_labels.append(length)
                
                # Create box plot
                fig, ax = plt.subplots(figsize=(10, 6))
                box = ax.boxplot(timing_data, patch_artist=True, labels=length_labels)
                
                # Add some styling
                for patch in box['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax.set_xlabel('Text Length')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title(f'Runtime Distribution vs Text Length ({metric_name})')
                plt.grid(True, axis='y', alpha=0.3)
                fig.tight_layout()
                self.results[f"length_comparison/{metric_name}/duration_boxplot"] = FigureResult(fig)
                
                # Bar chart comparison - memory vs length
                fig, ax = plt.subplots(figsize=(10, 6))
                width = 0.35
                
                ax.bar(x_pos - width/2, df["cpu_ram_mb"], width, yerr=df["cpu_ram_ci"], capsize=10, label='CPU RAM')
                if df["gpu_ram_mb"].sum() > 0:
                    ax.bar(x_pos + width/2, df["gpu_ram_mb"], width, yerr=df["gpu_ram_ci"], capsize=10, label='GPU RAM')
                    
                ax.set_xlabel('Text Length')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title(f'Memory Usage vs Text Length ({metric_name})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(lengths)
                ax.legend()
                fig.tight_layout()
                self.results[f"length_comparison/{metric_name}/memory"] = FigureResult(fig)


def main():
    """Example usage of the UtilizationExperiment."""
    from autometrics.metrics.reference_based.BLEU import BLEU
    from autometrics.metrics.reference_based.ROUGE import ROUGE
    from autometrics.metrics.reference_based.LENS import LENSMetric
    
    # Example metrics - including one that supports model loading/unloading
    metrics = [BLEU(), ROUGE()]
    
    # Optionally add a LENS metric if available
    try:
        lens = LENSMetric()
        # LENS has _load_model and _unload_model methods which will be used
        # by the experiment to properly measure GPU memory usage
        metrics.append(lens)
        print("Added LENS metric which supports model unloading/loading")
    except (ImportError, Exception) as e:
        print(f"LENS metric not available: {str(e)}")
    
    print("\n--- Running experiment with synthetic data ---")
    # Create and run the experiment with synthetic data
    # Output will go to outputs/utilization/synthetic
    experiment = UtilizationExperiment(
        name="Metric Utilization Experiment",
        description="Measuring resource usage of metrics on synthetic data",
        metrics=metrics,
        output_dir="outputs/utilization",  # Will be appended with /synthetic
        num_examples=10,  # Fewer examples for demonstration
        num_burn_in=2,
        use_synthetic=True
    )
    
    experiment.run(print_results=True)
    experiment.save_results()
    
    print("\n--- Running experiment with real dataset ---")
    # Example with real dataset
    # Output will go to outputs/utilization
    try:
        from autometrics.dataset.datasets.simplification.simplification import SimpDA
        dataset = SimpDA()
        
        experiment_real = UtilizationExperiment(
            name="Metric Utilization on Real Data",
            description="Measuring resource usage of metrics on real data",
            metrics=metrics,
            output_dir="outputs/utilization", 
            dataset=dataset,
            num_examples=10,
            num_burn_in=2,
            use_synthetic=False
        )
        
        experiment_real.run(print_results=True)
        experiment_real.save_results()
        print("\nExperiments complete. Results saved to:")
        print("  - Synthetic data: outputs/utilization/synthetic")
        print("  - Real data: outputs/utilization/real_data")
    except ImportError:
        print("SimpDA dataset not available for demonstration")
        print("\nSynthetic data experiment complete. Results saved to:")
        print("  - outputs/utilization/synthetic")


if __name__ == "__main__":
    main()

