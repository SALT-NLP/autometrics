#!/usr/bin/env python
"""
Test script for verifying caching with different initialization parameters.

This script ensures metrics with different initialization parameters use separate caches,
and that parameters are automatically included in the cache key without explicit registration.
"""

import time
import os
import shutil
import sys
from typing import Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.metrics.reference_based.BERTScore import BERTScore


def measure_execution_time(func: Callable, *args, **kwargs):
    """Measure the execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def test_different_param_caching():
    """Test caching for metrics with different initialization parameters"""
    print("\nTesting automatic initialization parameter caching:")
    
    # Clean any existing cache
    cache_dir = "./autometrics_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Test data
    input_text = "The cat sat on the mat."
    output_text = "A cat was sitting on a mat."
    references = ["The feline was on the carpet.", "A cat was resting on the mat."]
    
    # Create two BERTScore metrics with different models
    metric1 = BERTScore(model="distilbert-base-uncased")
    metric2 = BERTScore(model="roberta-base")
    
    # First run with metric1 (cache miss)
    print("First metric (distilbert), first run (cache miss):")
    result1, time1 = measure_execution_time(
        metric1.calculate, input_text, output_text, references
    )
    print(f"  Result: {result1}")
    print(f"  Time: {time1:.4f} seconds")
    
    # Second run with metric1 should use cache
    print("First metric (distilbert), second run (cache hit):")
    result2, time2 = measure_execution_time(
        metric1.calculate, input_text, output_text, references
    )
    print(f"  Result: {result2}")
    print(f"  Time: {time2:.4f} seconds")
    print(f"  Speedup: {time1/time2:.2f}x")
    
    # First run with metric2 should be a cache miss despite same input
    print("Second metric (roberta), first run (should be cache miss):")
    result3, time3 = measure_execution_time(
        metric2.calculate, input_text, output_text, references
    )
    print(f"  Result: {result3}")
    print(f"  Time: {time3:.4f} seconds")
    
    # Second run with metric2 should be a cache hit
    print("Second metric (roberta), second run (cache hit):")
    result4, time4 = measure_execution_time(
        metric2.calculate, input_text, output_text, references
    )
    print(f"  Result: {result4}")
    print(f"  Time: {time4:.4f} seconds")
    print(f"  Speedup: {time3/time4:.2f}x")
    
    # Verify results are different between the two metrics (different models)
    print("\nVerifying results:")
    print(f"  First metric results:  {result1}")
    print(f"  Second metric results: {result3}")
    
    assert result1 != result3, "Results should be different with different models"
    assert result1 == result2, "Results should be the same for same metric"
    assert result3 == result4, "Results should be the same for same metric"

    print("\nCACHING TEST PASSED: Different initialization parameters automatically use separate caches!")
    print("No explicit parameter registration needed - model parameter was automatically included in cache key.")
    
    return True


if __name__ == "__main__":
    test_different_param_caching() 