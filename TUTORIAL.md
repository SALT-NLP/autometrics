# Autometrics Tutorial

## Introduction

Autometrics automatically finds the best evaluation metrics for your NLP task by:
1. **Generating** task-specific metrics using LLMs
2. **Retrieving** relevant metrics from a bank of 60+ built-in metrics  
3. **Evaluating** all metrics on your dataset
4. **Selecting** the top metrics using regression
5. **Aggregating** into a single optimized metric

**Intended Use**: Evaluate text generation quality (summarization, translation, dialogue, etc.) with human-aligned metrics.

## Prerequisites: System Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Java Requirements
```bash
# Ubuntu/Debian
sudo apt install openjdk-21-jdk

# macOS
brew install openjdk@21

# Verify
java -version  # Should show Java 21
```

### GPU Requirements
Some metrics require GPUs. Check requirements:

```python
from autometrics.metrics.MetricBank import all_metric_classes

# Check GPU requirements
for metric_class in all_metric_classes:
    gpu_mem = getattr(metric_class, 'gpu_mem', 0)
    if gpu_mem > 0:
        print(f"{metric_class.__name__}: {gpu_mem:.0f}MB GPU")
```

**Examples:**
- `PRMRewardModel`: 130,000MB GPU (requires high-end GPU)
- `BERTScore`: 8MB GPU (works on most GPUs)
- `BLEU`: 0MB GPU (CPU-only)

## Step 1: Adding your Dataset

### Using Built-in Datasets
```python
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer
from autometrics.dataset.datasets.simplification.simplification import SimpDA

# Load built-in dataset
dataset = HelpSteer()  # or SimpDA(), etc.
target_measure = "helpfulness"  # column name with human scores
```

### Creating Custom Datasets
```python
import pandas as pd
from autometrics.dataset.Dataset import Dataset

# Your data
df = pd.DataFrame({
    'id': ['1', '2', '3'],
    'input': ['prompt 1', 'prompt 2', 'prompt 3'],
    'output': ['response 1', 'response 2', 'response 3'],
    'reference': ['ref 1', 'ref 2', 'ref 3'],  # optional
    'human_score': [4.5, 3.2, 4.8]  # target measure
})

# Create dataset
dataset = Dataset(
    dataframe=df,
    target_columns=['human_score'],
    ignore_columns=['id'],
    metric_columns=[],  # will be populated automatically
    name="MyCustomDataset",
    data_id_column="id",
    input_column="input", 
    output_column="output",
    reference_columns=['reference'],  # optional
    task_description="Evaluate response quality"
)
```

## Step 2: Running the Autometrics Pipeline

### Basic Usage (All Defaults)
```python
import os
import dspy
from autometrics.autometrics import Autometrics

# Set API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Configure LLMs
generator_llm = dspy.LM("openai/gpt-4o-mini")
judge_llm = dspy.LM("openai/gpt-4o-mini")

# Create pipeline with defaults
autometrics = Autometrics()

# Run pipeline
results = autometrics.run(
    dataset=dataset,
    target_measure="human_score",
    generator_llm=generator_llm,
    judge_llm=judge_llm
)

print(f"Top metrics: {[m.get_name() for m in results['top_metrics']]}")
print(f"Regression metric: {results['regression_metric'].get_name()}")
```

### Advanced Configuration
```python
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.LLMRec import LLMRec

# Custom configuration
autometrics = Autometrics(
    # Generate fewer metrics per type
    metric_generation_configs={
        "llm_judge": {"metrics_per_trial": 3},
        "codegen": {"metrics_per_trial": 2}
    },
    
    # Use specific retriever pipeline
    retriever_kwargs={
        "recommenders": [ColBERT, LLMRec],
        "top_ks": [20, 10]
    },
    
    # Include specific metrics upfront
    metric_priors=[LDLRewardModel, BLEU],
    
    # Allow some metric failures
    allowed_failed_metrics=2,
    
    # Custom output directory
    generated_metrics_dir="my_metrics"
)

results = autometrics.run(
    dataset=dataset,
    target_measure="human_score", 
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_to_retrieve=20,  # fewer metrics
    num_to_regress=3     # fewer final metrics
)
```

<details>
<summary><strong>Hyperparameters</strong></summary>

### Metric Generation Configs
```python
# Available generators
FULL_GENERATOR_CONFIGS = {
    "llm_judge": {"metrics_per_trial": 10},           # Basic LLM judge
    "rubric_prometheus": {"metrics_per_trial": 10},   # Prometheus rubric
    "rubric_dspy": {"metrics_per_trial": 5},          # DSPy rubric  
    "geval": {"metrics_per_trial": 10},               # G-Eval
    "codegen": {"metrics_per_trial": 10},             # Code generation
    "llm_judge_optimized": {"metrics_per_trial": 1},  # Optimized judge
    "finetune": {"metrics_per_trial": 1},             # Fine-tuned model
    "llm_judge_examples": {"metrics_per_trial": 1}    # Example-based
}
```

### Retriever Options
```python
# Single retrievers
from autometrics.recommend.BM25 import BM25
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.LLMRec import LLMRec

# Pipeline retrievers (recommended)
from autometrics.recommend.PipelinedRec import PipelinedRec

# Hardware-adaptive defaults:
# GPU: ColBERT → LLMRec
# CPU: BM25 → LLMRec
```

### Regression Strategies
```python
from autometrics.aggregator.regression.Lasso import Lasso
from autometrics.aggregator.regression.Ridge import Ridge
from autometrics.aggregator.regression.ElasticNet import ElasticNet

# Default: Lasso (sparse selection)
```

### Parallelization Settings
```python
autometrics = Autometrics(
    enable_parallel_evaluation=True,  # Speed up evaluation
    max_parallel_workers=20           # Adjust based on resources
)
```

</details>

## Step 3: Using your Metrics

### Understanding Results
```python
# Access results
top_metrics = results['top_metrics']           # List of selected metrics
regression_metric = results['regression_metric']  # Final aggregated metric
importance_scores = results['importance_scores']  # Metric importance
report_card = results['report_card']           # Summary report

# Use metrics on new data
for metric in top_metrics:
    scores = metric.predict(new_dataset)
    print(f"{metric.get_name()}: {scores.mean():.3f}")

# Use final regression metric
final_scores = regression_metric.predict(new_dataset)
```

### Metric Report Card
```python
print(results['report_card'])
```

**Example Output:**
```
# Autometrics Report Card

## Dataset Information
- Dataset: HelpSteer
- Target Measure: helpfulness
- Dataset Size: 1000 examples

## Top Metrics Selected
- 1. LDLRewardModel (MultiMetric: helpfulness, safety)
- 2. BERTScore
- 3. CustomLLMJudge_helpfulness

## Regression Aggregator
- Type: Lasso
- Name: Autometrics_Regression_helpfulness
```

### Adding to Metric Bank
```python
# Save generated metrics for reuse
autometrics = Autometrics(
    merge_generated_with_bank=True,  # Save to metric bank
    generated_metrics_dir="my_metric_bank"
)

# Load custom metric bank
from autometrics.metrics.MetricBank import all_metric_classes
custom_bank = all_metric_classes + [MyCustomMetric]
autometrics = Autometrics(metric_bank=custom_bank)
```

### Best Practices

1. **Start Simple**: Use defaults first, then customize
2. **Check GPU Requirements**: Some metrics need significant GPU memory
3. **Use Appropriate LLMs**: GPT-4o-mini works well, larger models for complex tasks
4. **Validate Results**: Check correlation with human scores
5. **Cache Results**: Metrics cache automatically in `./autometrics_cache/`

```python
# Quick validation
import numpy as np
from scipy.stats import pearsonr

human_scores = dataset.get_dataframe()['human_score']
predicted_scores = regression_metric.predict(dataset)

correlation, p_value = pearsonr(human_scores, predicted_scores)
print(f"Correlation: {correlation:.3f} (p={p_value:.3f})")
``` 