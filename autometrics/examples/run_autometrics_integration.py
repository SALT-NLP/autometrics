#!/usr/bin/env python3
"""
Full Autometrics integration demo on SimpDA with Qwen3-32B (litellm_proxy on port 8123).

- Generates 2 LLM-judge metrics and 1 Example Rubric metric
- Includes classic metrics (SARI, BLEU, ROUGE) as priors
- Runs retrieval, evaluation, regression (using Regression aggregator), and report card
- The report card HTML includes aggregated regression feedback by default

Usage:
  python autometrics/examples/run_autometrics_integration.py

Requirements:
  - litellm proxy serving Qwen3-32B at http://localhost:8123/v1
"""

import os
import argparse
import dspy

from autometrics.autometrics import Autometrics
from autometrics.metrics.reference_based.SARI import SARI
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.ROUGE import ROUGE


def _make_qwen32b_lm(port: int) -> dspy.LM:
    base_url = f"http://sphinx7.stanford.edu:{port}/v1"
    return dspy.LM(
        "litellm_proxy/Qwen/Qwen3-32B",
        api_base=base_url,
        temperature=0.0,
        max_tokens=4096,
        api_key=None,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Autometrics integration with Qwen3-32B on SimpDA")
    parser.add_argument("--port", type=int, default=8123, help="litellm_proxy port for Qwen3-32B")
    parser.add_argument("--rows", type=int, default=20, help="Number of SimpDA rows to evaluate (small for quick demo)")
    parser.add_argument("--retrieve", type=int, default=5, help="Number of metrics to retrieve")
    parser.add_argument("--top_n", type=int, default=4, help="Number of metrics to select for regression")
    parser.add_argument("--out", type=str, default=os.path.join("artifacts", "report_card", "autometrics_simpda_integration.html"), help="Output report card HTML path")
    args = parser.parse_args()

    # Build LMs for generation/evaluation
    generator_llm = _make_qwen32b_lm(args.port)
    judge_llm = _make_qwen32b_lm(args.port)

    # Load dataset and downsample for a quick run
    from autometrics.dataset.datasets.simplification.simplification import SimpDA
    ds = SimpDA()
    df = ds.get_dataframe().head(args.rows).copy()
    ds.set_dataframe(df)
    target_col = ds.get_target_columns()[0]

    # Configure Autometrics to use our Regression aggregator (which aggregates feedback)
    am = Autometrics(
        metric_bank=[SARI, BLEU, ROUGE],
        metric_generation_configs={
            # Generate 2 basic LLM Judges and 1 Example-based rubric metric
            "llm_judge": {"metrics_per_trial": 2},
            "llm_judge_examples": {"metrics_per_trial": 1},
        },
        enable_parallel_evaluation=False,  # safer for single-GPU/CPU; change to True for speed
        max_parallel_workers=32,
        seed=43,
        allowed_failed_metrics=0,
    )

    # Run the full pipeline
    results = am.run(
        dataset=ds,
        target_measure=target_col,
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        num_to_retrieve=args.retrieve,
        num_to_regress=args.top_n,
        regenerate_metrics=False,
        eval_dataset=ds,
        report_output_path=args.out,
        verbose=True,
    )

    print("\n=== Autometrics Integration Complete ===")
    print(f"Report card path: {results.get('report_card_path')}")
    print("Top metrics:")
    try:
        for m in results.get('top_metrics') or []:
            nm = m.get_name() if hasattr(m, 'get_name') else type(m).__name__
            print(f" - {nm}")
    except Exception:
        pass


if __name__ == "__main__":
    main()


