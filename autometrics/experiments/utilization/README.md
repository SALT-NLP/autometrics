# Metric Utilization Experiment

This module provides tools to benchmark and analyze the resource utilization of evaluation metrics in the `autometrics` library.

## Features

- Measures multiple resource aspects:
  - Runtime performance (milliseconds)
  - CPU memory usage (MB)
  - GPU memory usage (summed across all GPUs, in MB)
  - Disk usage changes (MB)
  
- Supports two data sources:
  - Synthetic text generation with configurable lengths
  - Real data from existing datasets (uses all available examples without filtering)
  
- Tests with configurable settings:
  - For synthetic data: short, medium, and long text lengths
  - For real data: uses the actual dataset examples as-is
  - Configurable number of examples and burn-in runs
  
- Produces comprehensive analysis:
  - Raw data for every test run
  - Statistical summary with means and confidence intervals
  - Visualization plots for easy comparison
  - JSON export for programmatic analysis

## Requirements

- Python 3.6+
- `psutil` for process monitoring
- `matplotlib` and `numpy` for plotting and analysis
- `pandas` for data manipulation
- Optional: `nltk` for better vocabulary in synthetic data generation
- Optional: `torch` for GPU memory monitoring with PyTorch
- Optional: `pynvml` for GPU memory monitoring with NVIDIA Management Library

## Usage

### Using Synthetic Data

```python
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.experiments.utilization import UtilizationExperiment

# Create and configure the experiment with synthetic data
experiment = UtilizationExperiment(
    name="Metric Utilization Benchmark",
    description="Measuring resource usage for NLG metrics",
    metrics=[BLEU(), ROUGE()],
    output_dir="outputs/utilization_synthetic",
    num_examples=30,  # Number of test examples per length category
    num_burn_in=5,    # Number of warm-up runs to avoid cold start effects
    lengths=["short", "medium", "long"],  # Text length categories to test
    use_synthetic=True  # Use synthetic data (default)
)

# Run the experiment
experiment.run(print_results=True)

# Save results to the output directory
experiment.save_results()
```

### Using Real Dataset

```python
from autometrics.dataset.datasets.simplification.simplification import SimpDA
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.experiments.utilization import UtilizationExperiment

# Load an actual dataset
dataset = SimpDA()

# Create and configure the experiment with real data
experiment = UtilizationExperiment(
    name="Real Data Utilization Benchmark",
    description="Measuring resource usage on real data",
    metrics=[BLEU(), ROUGE()],
    output_dir="outputs/utilization_real_data",
    dataset=dataset,
    num_examples=30,
    num_burn_in=5,
    use_synthetic=False  # Use the provided dataset
)

# Run the experiment
experiment.run(print_results=True)

# Save results to the output directory
experiment.save_results()
```

### Command-line Interface

The module provides a command-line interface through `run_utilization.py`:

```bash
# Run with synthetic data
python run_utilization.py --output-dir=outputs/my_experiment --num-examples=50 --burn-in=10 --metrics=BLEU,ROUGE,BERTScore --lengths=short,medium,long --synthetic

# Run with real dataset
python run_utilization.py --output-dir=outputs/my_experiment --num-examples=50 --burn-in=10 --metrics=BLEU,ROUGE,BERTScore --dataset=SimpDA
```

## Output Format

The experiment produces a well-organized directory structure of output files:

### For Synthetic Data

```
outputs/utilization_synthetic/
├── full_results.json                  # Complete results in JSON format
├── BLEU/
│   ├── short/                         # Results for short text inputs
│   │   ├── raw_data.csv               # Raw measurements for each run
│   │   ├── summary.csv                # Statistical summary
│   │   ├── duration_timeseries.pdf    # Time series plot of durations
│   │   ├── memory_timeseries.pdf      # Time series plot of memory usage
│   │   └── duration_histogram.pdf     # Histogram of durations
│   ├── medium/...                     # Similar files for medium text inputs
│   └── long/...                       # Similar files for long text inputs
├── ROUGE/...                          # Similar structure for ROUGE metric
├── metric_comparison/                 # Cross-metric comparisons
│   ├── short/                         # Comparisons for short texts
│   │   ├── summary.csv                # Summary statistics
│   │   ├── duration.pdf               # Duration comparison
│   │   └── memory.pdf                 # Memory usage comparison
│   ├── medium/...                     # Similar files for medium texts
│   └── long/...                       # Similar files for long texts
└── length_comparison/                 # Length impact analysis
    ├── BLEU/                          # Analysis for BLEU
    │   ├── summary.csv                # Summary across lengths
    │   ├── duration.pdf               # Duration vs length
    │   └── memory.pdf                 # Memory usage vs length
    └── ROUGE/...                      # Similar files for ROUGE
```

### For Real Dataset

```
outputs/utilization_real_data/
├── full_results.json                  # Complete results in JSON format
├── BLEU/                              # Results for BLEU metric
│   ├── raw_data.csv                   # Raw measurements
│   ├── summary.csv                    # Statistical summary
│   ├── duration_timeseries.pdf        # Time series plot
│   ├── memory_timeseries.pdf          # Memory usage plot
│   └── duration_histogram.pdf         # Distribution of durations
├── ROUGE/...                          # Similar structure for ROUGE
└── metric_comparison/                 # Cross-metric comparisons
    ├── summary.csv                    # Summary statistics
    ├── duration.pdf                   # Duration comparison
    └── memory.pdf                     # Memory usage comparison
```

## Synthetic Text Length Categories

When using synthetic data, these categories are used:
- **Short**: 3-10 words
- **Medium**: 80-120 words
- **Long**: 800-1200 words

## Resource Tracking API

The module provides a standalone resource tracking API that can be used outside the experiment:

```python
from autometrics.experiments.utilization import track_resources

# Track resources for any operation
with track_resources() as tracker:
    # Do some work here
    result = my_function()

# Get resource usage statistics
stats = tracker.get_results()
print(f"CPU RAM: {stats['cpu_ram_mb']} MB")
print(f"Duration: {stats['duration_milliseconds']} ms")
print(f"Total GPU Memory: {stats['gpu_ram_mb']} MB")
``` 