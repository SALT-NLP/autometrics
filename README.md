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


# Getting started

Make sure to install the necessary packages listed in `requirements.txt`.  Notably there could be some requirements missing so it would be amazing to collaborate on updating `requirements.txt` if anything is found to be lacking!

A nice simple starting point to working with this library would be to checkout the notebook `simpda.ipynb`.  This notebook shows computing metric correlations without introducing any LLM as a Judge complexity.  Just computing all metrics and aggregating (so skipping step 2)

For a more in depth introduction it would be useful to check out `simpda_dspy.ipynb` which will serve as an introduction to the LLM as a Judge components of the repo.