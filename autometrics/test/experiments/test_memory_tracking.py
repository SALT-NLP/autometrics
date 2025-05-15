#!/usr/bin/env python3
"""Unit tests for memory tracking functionality."""

import os
import sys
import time
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import pytest
from unittest.mock import patch, MagicMock

from autometrics.experiments.utilization.utilization import ResourceTracker, track_resources, measure_current_memory


def allocate_memory(size_mb):
    """Allocate a specified amount of memory."""
    # Create a list of byte arrays to allocate memory
    # Each element is 1MB
    byte_arrays = []
    for _ in range(size_mb):
        # Allocate 1MB
        byte_arrays.append(bytearray(1024 * 1024))
    return byte_arrays


@pytest.fixture
def setup_output_dir():
    """Create output directory for test results."""
    output_dir = os.path.join(os.path.dirname(__file__), '../../../outputs/test/')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@pytest.mark.parametrize("allocation_size", [10, 50, 100])
def test_memory_tracking_allocation(allocation_size):
    """Test the ResourceTracker correctly measures specific memory allocations."""
    # Force garbage collection before test
    gc.collect()
    time.sleep(0.1)  # Allow system to stabilize
    
    # Track memory usage during allocation
    with track_resources() as tracker:
        # Allocate memory
        memory_blocks = allocate_memory(allocation_size)
        # Hold the memory for a moment
        time.sleep(0.1)
    
    # Get the results
    metrics = tracker.get_results()
    
    # Clean up memory
    del memory_blocks
    gc.collect()
    
    # Allow for system variance with higher tolerance for small allocations
    # Small allocations have higher proportional overhead and compression
    if allocation_size <= 10:
        # 30% tolerance for small allocations
        lower_bound = allocation_size * 0.7
        upper_bound = allocation_size * 1.3
    else:
        # 20% tolerance for larger allocations
        lower_bound = allocation_size * 0.8
        upper_bound = allocation_size * 1.2
    
    # The incremental memory should be approximately the allocated amount
    assert metrics['cpu_ram_mb'] >= lower_bound, f"Measured memory {metrics['cpu_ram_mb']}MB is too low for {allocation_size}MB allocation"
    assert metrics['cpu_ram_mb'] <= upper_bound, f"Measured memory {metrics['cpu_ram_mb']}MB is too high for {allocation_size}MB allocation"
    
    # Check that durations are positive
    assert metrics['duration_milliseconds'] > 0, "Duration should be positive"


def test_correlation_across_allocations(setup_output_dir):
    """Test the correlation between allocated and measured memory."""
    results = []
    allocation_sizes = [10, 20, 50, 100]
    
    for size in allocation_sizes:
        # Force garbage collection before test
        gc.collect()
        time.sleep(0.1)  # Allow system to stabilize
        
        # Track memory usage during allocation
        with track_resources() as tracker:
            # Allocate memory
            memory_blocks = allocate_memory(size)
            # Hold the memory for a moment
            time.sleep(0.1)
        
        # Get the results
        metrics = tracker.get_results()
        results.append({
            'size': size,
            'measured': metrics['cpu_ram_mb']
        })
        
        # Clear the allocated memory
        del memory_blocks
        gc.collect()
    
    # Calculate correlation coefficient
    sizes = np.array([r['size'] for r in results])
    measured = np.array([r['measured'] for r in results])
    correlation = np.corrcoef(sizes, measured)[0, 1]
    
    # Plot results for visual inspection (optional in CI)
    if 'CI' not in os.environ:
        plt.figure(figsize=(10, 6))
        plt.scatter(sizes, measured, s=100, alpha=0.7)
        plt.plot(sizes, sizes, 'r--', label='Ideal (1:1)')
        
        # Add linear fit
        z = np.polyfit(sizes, measured, 1)
        p = np.poly1d(z)
        plt.plot(sizes, p(sizes), 'b-', label=f'Fit (slope={z[0]:.2f})')
        
        plt.title('Memory Tracking Validation')
        plt.xlabel('Allocated Memory (MB)')
        plt.ylabel('Measured Incremental Memory (MB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.annotate(f'Correlation: {correlation:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        # Save the plot
        plt.savefig(os.path.join(setup_output_dir, 'memory_validation.pdf'))
    
    # Correlation should be strong (>0.95)
    assert correlation > 0.95, f"Correlation between allocated and measured memory is too low: {correlation:.4f}"


def test_baseline_memory_tracking():
    """Test that baseline memory is properly tracked."""
    # First measure with an empty tracker
    tracker_1 = ResourceTracker().start()
    baseline_1 = tracker_1.start_memory_usage
    tracker_1.stop()
    
    # Allocate some memory
    memory_block = allocate_memory(50)
    
    # Now measure again
    tracker_2 = ResourceTracker().start()
    baseline_2 = tracker_2.start_memory_usage
    tracker_2.stop()
    
    # The baseline of the second tracker should be higher
    assert baseline_2 > baseline_1, "Baseline memory should increase after allocation"
    
    # Cleanup
    del memory_block
    gc.collect()


def test_incremental_calculation():
    """Test that incremental memory is correctly calculated."""
    # Start with a clean state
    gc.collect()
    
    with track_resources() as tracker:
        # Do nothing - this should record minimal memory usage
        pass
    
    empty_results = tracker.get_results()
    
    # Now allocate memory
    with track_resources() as tracker:
        memory_block = allocate_memory(50)
        time.sleep(0.1)  # Allow memory to register
    
    memory_results = tracker.get_results()
    
    # The incremental memory should be significantly higher in the second test
    assert memory_results['cpu_ram_mb'] > empty_results['cpu_ram_mb'] + 10, "Incremental memory should be higher when allocating"
    
    # Cleanup
    del memory_block
    gc.collect()


if __name__ == "__main__":
    # Allow running as a standalone script for debugging
    pytest.main(["-xvs", __file__]) 