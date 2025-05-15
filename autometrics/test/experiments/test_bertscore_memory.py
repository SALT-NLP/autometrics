#!/usr/bin/env python3
"""Test for BERTScore memory tracking."""

import os
import pytest
import gc
import time
import tempfile
import shutil
import torch
from unittest.mock import patch, MagicMock

from autometrics.metrics.reference_based.BERTScore import BERTScore
from autometrics.experiments.utilization.utilization import (
    ResourceTracker, 
    track_resources,
    UtilizationExperiment
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_bertscore_memory_tracking(temp_dir):
    """Test that BERTScore memory usage is properly tracked."""
    # Force garbage collection before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # First, measure the amount of memory allocated by BERTScore
    bertscore = BERTScore(persistent=False)
    
    # Track memory during first calculation
    with track_resources() as tracker:
        bertscore.calculate(
            "This is a test input.",
            "This is a generated output.",
            ["This is a reference."]
        )
    
    first_metrics = tracker.get_results()
    
    # The BERTScore should use a significant amount of memory
    assert first_metrics['cpu_ram_mb'] > 0.1, "BERTScore should use measurable memory for its model"
    
    # Track memory during second calculation - should be non-zero
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with track_resources() as tracker:
        bertscore.calculate(
            "This is another test.",
            "This is another output.",
            ["This is another reference."]
        )
    
    second_metrics = tracker.get_results()
    
    # Both memory usages should be measurable
    assert second_metrics['cpu_ram_mb'] > 0.1, "Second run should use measurable memory"
    
    # Print info for debugging
    print(f"First run memory: {first_metrics['cpu_ram_mb']:.2f} MB")
    print(f"Second run memory: {second_metrics['cpu_ram_mb']:.2f} MB")


def test_bertscore_in_experiment(temp_dir):
    """Test BERTScore inside a UtilizationExperiment."""
    # Create a simple experiment with just BERTScore
    experiment = UtilizationExperiment(
        name="BERTScore Memory Test",
        description="Testing memory tracking for BERTScore",
        metrics=[BERTScore(persistent=False)],
        output_dir=temp_dir,
        num_examples=2,  # Small number for quick test
        num_burn_in=1,
        lengths=["short"],  # Just test short inputs
        use_synthetic=True
    )
    
    # Run the experiment
    experiment.run(print_results=False)
    
    # Check results existence
    assert "BERTScore_roberta-large/short/raw_data" in experiment.results
    assert "BERTScore_roberta-large/short/summary" in experiment.results
    
    # Get raw data to check memory measurements
    raw_data = experiment.results["BERTScore_roberta-large/short/raw_data"].dataframe
    
    # Verify memory measurements are positive and reasonable
    assert all(raw_data['cpu_ram_mb'] > 0), "Incremental CPU RAM should be positive"
    assert all(raw_data['baseline_cpu_ram_mb'] > 0), "Baseline CPU RAM should be positive"
    
    # Check that incremental memory isn't zero (which would indicate improper tracking)
    assert all(raw_data['cpu_ram_mb'] > 0.1), "Incremental memory should be measurable"


if __name__ == "__main__":
    # Allow running as a standalone script for debugging
    pytest.main(["-xvs", __file__]) 