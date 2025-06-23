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
from autometrics.generator.CodeGenerator import CodeGenerator
from autometrics.generator.RubricGenerator import RubricGenerator
from autometrics.generator.FinetuneGenerator import FinetuneGenerator


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


def configure_prometheus():
    """Configure Prometheus model for evaluation."""
    try:
        from prometheus_eval.litellm import LiteLLM
        print("DEBUG: Successfully imported LiteLLM from prometheus_eval")
        # Use correct LiteLLM constructor: LiteLLM(name, api_base)
        model = LiteLLM(
            "litellm_proxy/Unbabel/M-Prometheus-14B",
            api_base="http://pasteur-hgx-1:7410/v1"
        )
        print(f"DEBUG: Successfully created LiteLLM model: {type(model)}")
        return model
    except ImportError as e:
        print(f"WARNING: Could not import prometheus_eval: {e}")
        print("DEBUG: Falling back to DSPy LM model pointing to Prometheus endpoint")
        # Create a DSPy LM pointing to the Prometheus endpoint as fallback
        try:
            prometheus_lm = dspy.LM(
                model="litellm_proxy/Unbabel/M-Prometheus-14B",
                api_base="http://pasteur-hgx-1:7410/v1",
                api_key="None"
            )
            print(f"DEBUG: Successfully created DSPy LM fallback: {type(prometheus_lm)}")
            return prometheus_lm
        except Exception as e2:
            print(f"ERROR: Failed to create DSPy LM fallback: {e2}")
            raise RuntimeError(f"Could not configure Prometheus model. Original error: {e}, Fallback error: {e2}")
    except Exception as e:
        print(f"WARNING: Could not create LiteLLM model: {e}")
        print("DEBUG: Falling back to DSPy LM model pointing to Prometheus endpoint")
        # Create a DSPy LM pointing to the Prometheus endpoint as fallback
        try:
            prometheus_lm = dspy.LM(
                model="litellm_proxy/Unbabel/M-Prometheus-14B",
                api_base="http://pasteur-hgx-1:7410/v1",
                api_key="None"
            )
            print(f"DEBUG: Successfully created DSPy LM fallback: {type(prometheus_lm)}")
            return prometheus_lm
        except Exception as e2:
            print(f"ERROR: Failed to create DSPy LM fallback: {e2}")
            raise RuntimeError(f"Could not configure Prometheus model. Original error: {e}, Fallback error: {e2}")


# ----------------------------------------------------------------------------
# 2.  Run the full pipeline for a dataset
# ----------------------------------------------------------------------------

def run_pipeline(dataset, generator_lm, judge_lm, n_metrics: int = 3, metric_type: str = "llm_judge", model_save_dir: str = None):
    banner(f"DATASET: {dataset.get_name()}")
    print("Task description:\n", dataset.get_task_description())

    # Instantiate the proposer ------------------------------------------------
    if metric_type == "geval":
        proposer = GEvalJudgeProposer(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
        )
        print("Using G-Eval metrics...")
    elif metric_type == "codegen":
        proposer = CodeGenerator(
            generator_llm=generator_lm,
        )
        print("Using Code Generation metrics...")
    elif metric_type == "rubric_prometheus":
        proposer = RubricGenerator(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
            use_prometheus=True,
        )
        print("Using Rubric Generator with Prometheus metrics...")
    elif metric_type == "rubric_dspy":
        proposer = RubricGenerator(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
            use_prometheus=False,
        )
        print("Using Rubric Generator with DSPy metrics...")
    elif metric_type == "finetune":
        proposer = FinetuneGenerator(
            generator_llm=generator_lm,
            model_save_dir=model_save_dir,
        )
        print("Using Fine-tune Generator with ModernBERT...")
        if model_save_dir:
            print(f"   Custom model directory: {model_save_dir}")
    else:  # llm_judge
        proposer = BasicLLMJudgeProposer(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
        )
        print("Using Basic LLM Judge metrics...")

    banner("Generating metric axes …")
    
    # For fine-tuning, default to 1 metric if not specified
    if metric_type == "finetune" and n_metrics == 3:  # 3 is the default
        n_metrics = 1
        print("Note: Fine-tuning is expensive, defaulting to n_metrics=1")
    
    metrics = proposer.generate(dataset, target_measure=dataset.get_target_columns()[0], n_metrics=n_metrics)

    for m in metrics:
        print("➤", m.name, "-", m.description)

    # Evaluate the first metric on a tiny subset ------------------------------
    eval_type_display = {
        "geval": "G-Eval", 
        "codegen": "Code Generation", 
        "llm_judge": "LLM Judge",
        "rubric_prometheus": "Rubric (Prometheus)",
        "rubric_dspy": "Rubric (DSPy)",
        "finetune": "Fine-tuned ModernBERT"
    }[metric_type]
    banner(f"Scoring first 10 examples with first generated {eval_type_display} metric …")
    first_metric = metrics[0] if len(metrics) > 0 else None
    
    if first_metric:
        df = dataset.get_dataframe().head(10)

        # Prepare references if the dataset has them
        references = None
        if dataset.get_reference_columns():
            references = []
            for _, row in df.iterrows():
                row_refs = [row[col] for col in dataset.get_reference_columns() if row[col] is not None]
                references.append(row_refs)

        scores = first_metric.calculate_batched(
            inputs=df[dataset.get_input_column()].tolist(),
            outputs=df[dataset.get_output_column()].tolist(),
            references=references,
        )
        pprint.pprint(list(zip(range(len(scores)), scores)))

        # Optionally save the metric as standalone python file --------------------
        out_dir = Path("generated_metrics")
        out_dir.mkdir(exist_ok=True)
        
        # Clean filename - much simpler now that names are clean from source
        safe_filename = (first_metric.name
                        .replace(" ", "_")
                        .replace("/", "_") 
                        .replace(":", "_")
                        .replace("-", "_"))
        
        first_metric.save_python_code(out_dir / f"{safe_filename}_Metric.py")
        print("Saved standalone metric to", out_dir / f"{safe_filename}_Metric.py")
        
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
    parser = argparse.ArgumentParser(description="Demo metric generation with Basic LLM Judge, G-Eval, Code Generation, Rubric Generator, or Fine-tuned ModernBERT")
    parser.add_argument("--metric-type", choices=["llm_judge", "geval", "codegen", "rubric_prometheus", "rubric_dspy", "finetune"], default="llm_judge", 
                       help="Choose the metric type to use (default: llm_judge)")
    parser.add_argument("--model", choices=["gpt4o_mini", "qwen"], default="gpt4o_mini", 
                       help="Choose the model to use (default: gpt4o_mini)")
    parser.add_argument("--n-metrics", type=int, default=3, help="Number of metrics to generate (default: 3)")
    parser.add_argument("--model-save-dir", type=str, default="/sphinx/u/salt-checkpoints/autometrics/models", 
                       help="Custom directory to save fine-tuned models (only used for finetune metric type)")
    args = parser.parse_args()

    # Configure models based on choice
    if args.model == "qwen":
        generator_lm, judge_lm = configure_qwen()
        print("Using Qwen models...")
    else:
        generator_lm, judge_lm = configure_gpt4o_mini()
        print("Using GPT-4o-mini models...")

    # Display configuration
    metric_type_display = {
        "llm_judge": "Basic LLM Judge", 
        "geval": "G-Eval", 
        "codegen": "Code Generation",
        "rubric_prometheus": "Rubric Generator (Prometheus)",
        "rubric_dspy": "Rubric Generator (DSPy)",
        "finetune": "Fine-tuned ModernBERT"
    }
    print(f"Metric type: {metric_type_display[args.metric_type]}")
    print(f"Number of metrics: {args.n_metrics}")
    print()

    # For Prometheus metrics, we need to use the Prometheus model as the evaluator
    if args.metric_type == "rubric_prometheus":
        prometheus_lm = configure_prometheus()
        for ds in [CoGymTravelOutcome(), SimpDA()]:
            run_pipeline(ds, generator_lm, prometheus_lm, n_metrics=args.n_metrics, metric_type=args.metric_type, model_save_dir=args.model_save_dir)
    elif args.metric_type == "finetune":
        # For fine-tuning, we don't need a judge model, just the generator model for metric cards
        for ds in [CoGymTravelOutcome(), SimpDA()]:
            run_pipeline(ds, generator_lm, None, n_metrics=args.n_metrics, metric_type=args.metric_type, model_save_dir=args.model_save_dir)
    else:
        for ds in [CoGymTravelOutcome(), SimpDA()]:
            run_pipeline(ds, generator_lm, judge_lm, n_metrics=args.n_metrics, metric_type=args.metric_type, model_save_dir=args.model_save_dir)

    # print(generator_lm.inspect_history(n=2))
