#!/usr/bin/env python3
"""
Autometrics Quickstart — generated-only
=======================================

The dead-simple entry point. Bring a small dataset of (input, output, human_score)
rows and Autometrics will generate task-specific LLM-judge metrics, fit them to
your human scores with PLS, and hand you back one aggregated metric you can call
on new data.

Everything in this tutorial runs in generated-only mode: no metric bank, no
retrieval, no Java, no GPU. All you need is an OPENAI_API_KEY.

Usage:
    export OPENAI_API_KEY="sk-..."
    python tutorial.py
"""

import os
import pandas as pd
import dspy

from autometrics.autometrics import Autometrics
from autometrics.dataset.Dataset import Dataset


# 1. Build a tiny dataset.
#    Columns are up to you — just name the input, output, and target-score
#    columns when you construct the Dataset. With <= 100 rows and the default
#    metric bank, Autometrics skips retrieval entirely and uses only the
#    metrics it generates for your task (generated-only mode).
df = pd.DataFrame({
    "id":     ["1", "2", "3", "4", "5", "6", "7", "8"],
    "input":  [
        "Explain photosynthesis to a 10-year-old.",
        "What is 17 * 23?",
        "Summarize the plot of Hamlet in one sentence.",
        "Give me a recipe for pancakes.",
        "Translate 'good morning' to Spanish.",
        "What causes rainbows?",
        "Write a haiku about autumn.",
        "List three benefits of exercise.",
    ],
    "output": [
        "Plants use sunlight to turn water and air into food. It's how they eat!",
        "17 times 23 equals 391.",
        "A Danish prince seeks revenge for his father's murder and everything ends badly.",
        "Mix flour, eggs, milk; cook on a hot pan.",
        "Buenos días.",
        "Light bending through water droplets.",
        "crisp leaves underfoot / the maple tree lets them fall / sweater weather now",
        "Better mood, stronger heart, more energy.",
    ],
    "helpfulness": [5, 5, 4, 3, 5, 2, 4, 5],
})

dataset = Dataset(
    dataframe=df,
    target_columns=["helpfulness"],
    ignore_columns=["id"],
    metric_columns=[],
    name="QuickstartDemo",
    data_id_column="id",
    input_column="input",
    output_column="output",
    reference_columns=[],
    task_description="Answer the user's question helpfully.",
)

# 2. Pick an LLM for generating and judging. GPT-4o-mini is cheap and good enough.
generator_llm = dspy.LM("openai/gpt-4o-mini")
judge_llm = dspy.LM("openai/gpt-4o-mini")

# 3. Generate a handful of LLM-judge metrics. PLS is the default aggregator
#    (the one used in the paper) — no need to pass it explicitly.
autometrics = Autometrics(
    metric_generation_configs={"llm_judge": {"metrics_per_trial": 3}},
    generated_metrics_dir="quickstart_metrics",
    seed=42,
)

results = autometrics.run(
    dataset=dataset,
    target_measure="helpfulness",
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_to_regress=2,
)

# 4. Inspect what came out.
print(f"\nGenerated {len(results['all_generated_metrics'])} metrics; "
      f"kept the top {len(results['top_metrics'])}:")
for m in results["top_metrics"]:
    print(f"  - {m.get_name()}")

print(f"\nAggregated metric: {results['regression_metric'].get_name()}")

# 5. Use it on new data. The regression_metric is a regular Metric, so you can
#    call .predict(dataset) on any Dataset with the same input/output schema.
scores = results["regression_metric"].predict(dataset)
print(f"\nPredicted vs. human scores:")
for pred, human in zip(scores, df["helpfulness"]):
    print(f"  predicted={pred:.2f}  human={human}")
