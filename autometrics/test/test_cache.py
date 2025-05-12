#!/usr/bin/env python
"""
Test script for metric caching functionality.

This script demonstrates how caching works and measures the performance improvement.
"""

import time
import os
import shutil
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.SARI import SARI

def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def test_metric_caching(metric_class, input_text, output_text, references):
    """Test caching for a specific metric"""
    print(f"\nTesting {metric_class.__name__}:")
    
    # First run with caching enabled
    print("First run (cache miss):")
    metric = metric_class(use_cache=True)
    result1, time1 = measure_execution_time(
        metric.calculate, input_text, output_text, references
    )
    print(f"  Result: {result1}")
    print(f"  Time: {time1:.4f} seconds")
    
    # Second run with same inputs should use cache
    print("Second run (cache hit):")
    result2, time2 = measure_execution_time(
        metric.calculate, input_text, output_text, references
    )
    print(f"  Result: {result2}")
    print(f"  Time: {time2:.4f} seconds")
    print(f"  Speedup: {time1/time2:.2f}x")
    
    # Test with cache disabled
    print("Run with cache disabled:")
    metric_no_cache = metric_class(use_cache=False)
    result3, time3 = measure_execution_time(
        metric_no_cache.calculate, input_text, output_text, references
    )
    print(f"  Result: {result3}")
    print(f"  Time: {time3:.4f} seconds")
    
    # Verify results are the same
    assert result1 == result2 == result3, "Results should be identical"
    
    return result1, time1, time2, time3

def test_batch_caching(metric_class, inputs, outputs, references):
    """Test batch caching for a specific metric"""
    print(f"\nTesting batch mode for {metric_class.__name__}:")
    
    # First batch run (cache miss for all)
    print("First batch run (all cache misses):")
    metric = metric_class()
    result1, time1 = measure_execution_time(
        metric.calculate_batched, inputs, outputs, references
    )
    print(f"  Results: {result1}")
    print(f"  Time: {time1:.4f} seconds")
    
    # Second batch run (all cache hits)
    print("Second batch run (all cache hits):")
    result2, time2 = measure_execution_time(
        metric.calculate_batched, inputs, outputs, references
    )
    print(f"  Results: {result2}")
    print(f"  Time: {time2:.4f} seconds")
    print(f"  Speedup: {time1/time2:.2f}x")
    
    # Third batch run with partial overlap
    new_inputs = inputs[1:] + ["This is a new input not in cache"]
    new_outputs = outputs[1:] + ["This is a new output not in cache"]
    new_refs = references[1:] + [["This is a new reference not in cache"]]
    
    print("Third batch run (partial cache hits):")
    result3, time3 = measure_execution_time(
        metric.calculate_batched, new_inputs, new_outputs, new_refs
    )
    print(f"  Results: {result3}")
    print(f"  Time: {time3:.4f} seconds")
    
    # Verify results
    assert result1 == result2, "First and second batch results should be identical"
    assert result3[:-1] == result2[1:], "Overlapping results should be identical"
    
    return result1, result2, result3

def main():
    """Main test function"""
    # Clean any existing cache
    cache_dir = "./autometrics_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Test data
    input_text = "The cat sat on the mat."
    output_text = "A cat was sitting on a mat."
    references = ["The feline was on the carpet.", "A cat was resting on the mat."]
    
    # Batch test data
    inputs = [
        "The cat sat on the mat.",
        "The dog barked loudly.",
        "The sun is shining brightly."
    ]
    outputs = [
        "A cat was sitting on a mat.",
        "A loud dog was barking.",
        "It's a bright and sunny day."
    ]
    refs = [
        ["The feline was on the carpet.", "A cat was resting on the mat."],
        ["The dog made a loud noise.", "A noisy dog was heard."],
        ["The weather is nice today.", "The sun shines bright."]
    ]
    
    # Test BLEU
    bleu_results = test_metric_caching(BLEU, input_text, output_text, references)
    
    # Test SARI
    sari_results = test_metric_caching(SARI, input_text, output_text, references)
    
    # Test batch caching
    batch_bleu = test_batch_caching(BLEU, inputs, outputs, refs)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 