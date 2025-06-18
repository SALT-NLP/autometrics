# autometrics
Research Repo for the AutoMetrics library

This README is in progress!  Right now it will give useful pointers for navigating the repo!

The pipeline for recommending metrics works as follows:
1. Accept Human Labelled outputs and open ended feedback
2. Retrieve relevant metrics from 100s in our metric bank
3. Generate LLM as a Judge Rubrics based on human feedback
4. Aggregate into a single metric using Regression
5. Output Top-k relevant metrics, a single metric regression, and a metric report card

The parts of the repo are organized as follows:
- `inputs`: Right now these come in the form of `datasets`
    - Located at `autometrics/dataset/datasets` with the main class `autometrics/dataset/Dataset.py`
- `metrics`: All the metrics in the bank that we will retreive
    - Located at `autometrics/metrics`
- `LLM as a Judge`: The code for generating LLM-as-a-judge rubrics based on feedback
    - Located at `autometrics/metrics/llm_judge`.  There are several types to experiment with
- `Aggregate`: The regression code for taking multiple metrics and learning a regression
    - Located at `autometrics/aggregator`.  Specifically `autometrics/aggregator/regression` for the regression based methods.
- `Evaluate`: For some tasks we compute accuracy of scores (i.e. pairwise) and some we compute correlation (i.e. scalar human labels).  Eventually some more evaluations of our metrics will go here, notably this is not for metrics that measure text quality themselves.
    - Located at `autometrics/evaluate`.
- `Test`: Unit tests and functionality tests
    - Located at `autometrics/test`. Contains tests for caching and other features.

# Getting started

Make sure to install the necessary packages listed in `requirements.txt`.  Notably there could be some requirements missing so it would be amazing to collaborate on updating `requirements.txt` if anything is found to be lacking!

## Java Requirements

This package requires Java Development Kit (JDK) 21 for some of its search functionality. You can install it using one of these methods:

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install openjdk-21-jdk
```

### macOS (using Homebrew)
```bash
brew install openjdk@21
```

### Windows
Download and install from [Oracle's website](https://www.oracle.com/java/technologies/downloads/#java21) or use [Chocolatey](https://chocolatey.org/):
```bash
choco install openjdk21
```

After installation, verify your Java version:
```bash
java -version
```

You should see something like:
```
openjdk version "21.0.x"
OpenJDK Runtime Environment ...
OpenJDK 64-Bit Server VM ...
```

Note: Java 17 or lower versions will not work as Pyserini requires Java 21.

A nice simple starting point to working with this library would be to checkout the notebook `simpda.ipynb`.  This notebook shows computing metric correlations without introducing any LLM as a Judge complexity.  Just computing all metrics and aggregating (so skipping step 2)

For a more in depth introduction it would be useful to check out `simpda_dspy.ipynb` which will serve as an introduction to the LLM as a Judge components of the repo.

# Disk Caching

The library implements disk caching for all metrics to improve performance when running scripts multiple times. Key features:

- All metrics cache results by default in the `./autometrics_cache` directory
- Cache keys are generated based on:
  - Input/output/references passed to the metric
  - All initialization parameters (automatically included by default)
  - Any additional keyword arguments passed to the calculate method
- All initialization parameters automatically affect caching
  - No need to explicitly register parameters
  - Different parameter values create separate caches
  - For example, BERTScore with different models or LLMJudge with different prompts will use different caches
- The following parameters are automatically excluded from the cache key:
  - `name` and `description` (don't affect output, just labeling)
  - `use_cache` and `cache_dir` (cache configuration, not behavior)
- You can exclude additional parameters that don't affect results using `self.exclude_from_cache_key()`
  - For example, debug flags or verbosity settings
- Caching can be disabled per-metric-instance by passing `use_cache=False` during initialization
- Some simple metrics like BLEU and SARI have caching disabled by default (`DEFAULT_USE_CACHE=False`) since their computation is faster than cache lookup

To implement caching in your own metrics, you only need to:
1. Call the parent constructor with `super().__init__(...)`
2. Exclude any additional parameters that don't affect results with `self.exclude_from_cache_key('param1', 'param2', ...)`

See examples in `autometrics/test/custom_metric_caching_example.py`.

## Citation

If you use this software, please cite it as below.

```
@software{Ryan_Autometrics_2025,
  author = {Ryan, Michael J. and Zhang, Yanzhe and Salunkhe, Amol and Chu, Yi and Rahman, Emily and Xu, Di and Yang, Diyi},
  license = {MIT},
  title = {{Autometrics}},
  url = {https://github.com/XenonMolecule/autometrics},
  version = {1.0.0},
  year = {2025}
}
```