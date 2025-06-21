import os
from pathlib import Path
import pprint
import argparse

import dspy

# Datasets -----------------------------------------------------------------
from autometrics.dataset.datasets.simplification import SimpDA
from autometrics.dataset.datasets.cogym.cogym import CoGymTravelOutcome

# Generators ---------------------------------------------------------------
from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer
from autometrics.generator.GEvalJudgeProposer import GEvalJudgeProposer


# ----------------------------------------------------------------------------
# Helper to pretty-print a separator
# ----------------------------------------------------------------------------

def banner(text: str):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# 1.  Configure the *generator LLM* (for axis discovery) & *judge LLM*
# ----------------------------------------------------------------------------

def configure_gpt4o_mini():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please export OPENAI_API_KEY before running the demo.")

    generator_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    judge_lm = generator_lm  # Using the same model for judging in this demo

    return generator_lm, judge_lm

def configure_qwen():
    generator_lm = dspy.LM("litellm_proxy/Qwen/Qwen3_32B", api_base="http://iliad-hgx-1:7410/v1")
    judge_lm = generator_lm
    return generator_lm, judge_lm


# ----------------------------------------------------------------------------
# 2.  Run the full pipeline for a dataset
# ----------------------------------------------------------------------------

def run_pipeline(dataset, generator_lm, judge_lm, n_metrics: int = 3, use_geval: bool = False):
    banner(f"DATASET: {dataset.get_name()}")
    print("Task description:\n", dataset.get_task_description())

    # Instantiate the proposer ------------------------------------------------
    if use_geval:
        proposer = GEvalJudgeProposer(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
        )
        print("Using G-Eval metrics...")
    else:
        proposer = BasicLLMJudgeProposer(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
        )
        print("Using Basic LLM Judge metrics...")

    banner("Generating metric axes …")
    metrics = proposer.generate(dataset, target_measure=dataset.get_target_columns()[0], n_metrics=n_metrics)

    for m in metrics:
        print("➤", m.name, "-", m.description)

    # Evaluate the first metric on a tiny subset ------------------------------
    metric_type = "G-Eval" if use_geval else "LLM Judge"
    banner(f"Scoring first 5 examples with first generated {metric_type} metric …")
    first_metric = metrics[0] if len(metrics) > 0 else None
    
    if first_metric:
        df = dataset.get_dataframe().head(5)

        scores = first_metric.calculate_batched(
            inputs=df[dataset.get_input_column()].tolist(),
            outputs=df[dataset.get_output_column()].tolist(),
        )
        pprint.pprint(list(zip(range(len(scores)), scores)))

        # Optionally save the metric as standalone python file --------------------
        out_dir = Path("generated_metrics")
        out_dir.mkdir(exist_ok=True)
        
        # Clean filename
        safe_filename = first_metric.name.replace(" ", "_").replace("/", "_").replace(":", "_")
        first_metric.save_python_code(out_dir / f"{safe_filename}.py")
        print("Saved standalone metric to", out_dir / f"{safe_filename}.py")
        
        # Show a sample of the metric card
        if hasattr(first_metric, 'metric_card') and first_metric.metric_card:
            banner("Sample Metric Card (first 500 chars)")
            print(first_metric.metric_card[:500] + "..." if len(first_metric.metric_card) > 500 else first_metric.metric_card)
    else:
        print("No metrics were generated!")


# ----------------------------------------------------------------------------
# Main entrypoint
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo metric generation with Basic LLM Judge or G-Eval")
    parser.add_argument("--geval", action="store_true", help="Use G-Eval metrics instead of Basic LLM Judge")
    parser.add_argument("--model", choices=["gpt4o_mini", "qwen"], default="gpt4o_mini", 
                       help="Choose the model to use (default: gpt4o_mini)")
    parser.add_argument("--n-metrics", type=int, default=3, help="Number of metrics to generate (default: 3)")
    args = parser.parse_args()

    # Configure models based on choice
    if args.model == "qwen":
        generator_lm, judge_lm = configure_qwen()
        print("Using Qwen models...")
    else:
        generator_lm, judge_lm = configure_gpt4o_mini()
        print("Using GPT-4o-mini models...")

    # Display configuration
    print(f"Metric type: {'G-Eval' if args.geval else 'Basic LLM Judge'}")
    print(f"Number of metrics: {args.n_metrics}")
    print()

    for ds in [CoGymTravelOutcome(), SimpDA()]:
        run_pipeline(ds, generator_lm, judge_lm, n_metrics=args.n_metrics, use_geval=args.geval)

    # print(generator_lm.inspect_history(n=2))