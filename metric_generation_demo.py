import os
from pathlib import Path
import pprint

import dspy

# Datasets -----------------------------------------------------------------
from autometrics.dataset.datasets.simplification import SimpDA
from autometrics.dataset.datasets.cogym.cogym import CoGymTravelOutcome

# Generator ----------------------------------------------------------------
from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer


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

def run_pipeline(dataset, generator_lm, judge_lm, n_metrics: int = 3):
    banner(f"DATASET: {dataset.get_name()}")
    print("Task description:\n", dataset.get_task_description())

    # Instantiate the proposer ------------------------------------------------
    proposer = BasicLLMJudgeProposer(
        generator_llm=generator_lm,
        executor_kwargs={"model": judge_lm},
    )

    banner("Generating metric axes …")
    metrics = proposer.generate(dataset, target_measure=dataset.get_target_columns()[0], n_metrics=n_metrics)

    for m in metrics:
        print("➤", m.name, "-", m.description)

    # Evaluate the first metric on a tiny subset ------------------------------
    banner("Scoring first 5 examples with first generated metric …")
    first_metric = metrics[2]
    df = dataset.get_dataframe().head(5)

    scores = first_metric.calculate_batched(
        inputs=df[dataset.get_input_column()].tolist(),
        outputs=df[dataset.get_output_column()].tolist(),
    )
    pprint.pprint(list(zip(range(len(scores)), scores)))

    # Optionally save the metric as standalone python file --------------------
    out_dir = Path("generated_metrics")
    out_dir.mkdir(exist_ok=True)
    first_metric.save_python_code(out_dir / f"{first_metric.name}.py")
    print("Saved standalone metric to", out_dir / f"{first_metric.name}.py")


# ----------------------------------------------------------------------------
# Main entrypoint
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generator_lm, judge_lm = configure_gpt4o_mini()

    for ds in [CoGymTravelOutcome(), SimpDA()]:
        run_pipeline(ds, generator_lm, judge_lm)

    # print(generator_lm.inspect_history(n=2))