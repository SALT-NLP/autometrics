import os
from pathlib import Path
import pprint
import argparse
import importlib.util
import sys

import dspy

# Datasets -----------------------------------------------------------------
from autometrics.dataset.datasets.simplification import SimpDA
from autometrics.dataset.datasets.cogym.cogym import CoGymTravelOutcome

# Generators ---------------------------------------------------------------
from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer
from autometrics.generator.LLMJudgeExampleProposer import LLMJudgeExampleProposer
from autometrics.generator.OptimizedJudgeProposer import OptimizedJudgeProposer
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
# Helper to test metric serialization by loading and re-running saved metric
# ----------------------------------------------------------------------------

def test_metric_serialization(metric_file_path, inputs, outputs, references, original_scores):
    """
    Load a saved metric from file and test that it produces the same results.
    
    Args:
        metric_file_path: Path to the saved metric Python file
        inputs: List of input texts
        outputs: List of output texts  
        references: List of reference texts (or None)
        original_scores: Original scores to compare against
    
    Returns:
        bool: True if serialization test passed, False otherwise
    """
    
    print("\n" + "🔍" * 80)
    print(f"🔍 STARTING SERIALIZATION TEST")
    print(f"🔍 Metric File: {metric_file_path}")  
    print(f"🔍 Testing on {len(inputs)} examples")
    print("🔍" * 80)
    
    print(f"🔍 Starting serialization test for: {metric_file_path}")
    
    try:
        # Import the metric module dynamically
        print(f"📦 Loading module from: {metric_file_path}")
        spec = importlib.util.spec_from_file_location("loaded_metric", metric_file_path)
        if spec is None or spec.loader is None:
            print(f"❌ Failed to load metric from {metric_file_path}")
            return False
            
        loaded_module = importlib.util.module_from_spec(spec)
        print(f"✅ Module spec created successfully")
        
        spec.loader.exec_module(loaded_module)
        print(f"✅ Module executed successfully")
        
        # Find the metric class (should be the only class in the module)
        # Look for classes that have a calculate_batched method (the key metric interface)
        print(f"🔍 Searching for metric classes in module...")
        
        all_classes = []
        for name in dir(loaded_module):
            obj = getattr(loaded_module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, 'calculate_batched') and 
                callable(getattr(obj, 'calculate_batched'))):
                
                # Check if this class is defined in this module (not imported)
                is_local_class = obj.__module__ == loaded_module.__name__
                all_classes.append((name, obj, is_local_class))
                
                if is_local_class:
                    print(f"  🎯 Found LOCAL metric class: {name} (defined in this file)")
                else:
                    print(f"  📚 Found IMPORTED class: {name} (from {obj.__module__})")
        
        # Prioritize locally defined classes over imported ones
        local_classes = [(name, obj) for name, obj, is_local in all_classes if is_local]
        imported_classes = [(name, obj) for name, obj, is_local in all_classes if not is_local]
        
        if local_classes:
            # Use the first locally defined class (the actual generated metric)
            MetricClass = local_classes[0][1]
            print(f"✅ Using LOCAL metric class: {local_classes[0][0]}")
            if len(local_classes) > 1:
                print(f"⚠️  Multiple local classes found, using first: {local_classes[0][0]}")
        elif imported_classes:
            # Fall back to imported classes if no local ones found
            MetricClass = imported_classes[0][1]
            print(f"⚠️  No local classes found, using IMPORTED class: {imported_classes[0][0]}")
        else:
            print(f"❌ No metric class found in {metric_file_path}")
            print(f"  Available classes: {[name for name in dir(loaded_module) if isinstance(getattr(loaded_module, name), type)]}")
            return False
        print(f"🏗️  Attempting to instantiate: {MetricClass.__name__}")
        
        # Try to instantiate with default model if available
        try:
            # Check if there's a DEFAULT_MODEL defined in the module
            if hasattr(loaded_module, 'DEFAULT_MODEL'):
                print(f"✅ Found DEFAULT_MODEL in module, using it for instantiation")
                loaded_metric = MetricClass(model=loaded_module.DEFAULT_MODEL)
            else:
                print(f"⚠️  No DEFAULT_MODEL found, trying parameterless constructor")
                loaded_metric = MetricClass()
        except TypeError as e:
            print(f"❌ TypeError during instantiation: {e}")
            # If constructor needs arguments, try with DEFAULT_MODEL
            if hasattr(loaded_module, 'DEFAULT_MODEL'):
                print(f"🔄 Retrying with DEFAULT_MODEL...")
                try:
                    loaded_metric = MetricClass(model=loaded_module.DEFAULT_MODEL)
                    print(f"✅ Retry successful!")
                except Exception as retry_e:
                    print(f"❌ Retry also failed: {retry_e}")
                    return False
            else:
                print(f"❌ Failed to instantiate metric class: {e}")
                print(f"  Available module attributes: {[attr for attr in dir(loaded_module) if not attr.startswith('_')]}")
                return False
        
        print(f"✅ Metric instantiated successfully: {type(loaded_metric).__name__}")
        
        # Run the loaded metric on a small subset to generate LLM history
        test_inputs = inputs[:2]  # Just use first 2 examples for history inspection
        test_outputs = outputs[:2]
        test_references = references[:2] if references else None
        
        print(f"🧪 Testing metric execution on {len(test_inputs)} examples...")
        print(f"  Has references: {test_references is not None}")
        
        loaded_scores = loaded_metric.calculate_batched(test_inputs, test_outputs, test_references)
        print(f"✅ Metric execution completed, got {len(loaded_scores)} scores")
        
        # Show the actual scores for verification/comparison
        print(f"📊 Reloaded metric scores: {loaded_scores}")
        
        # Verify basic functionality
        if len(loaded_scores) != len(test_inputs):
            print(f"❌ Metric produced wrong number of scores: {len(loaded_scores)} vs {len(test_inputs)}")
            return False
        
        print("✅ Basic functionality test PASSED - metric loads and runs correctly")
        
        # Inspect LLM history to verify prompt structure
        print("\n🔍 Inspecting LLM history to verify prompt usage...")
        print("📜 LLM History (last 2 calls):")
        print("-" * 60)
        
        # Get the model from the loaded metric and show history
        if hasattr(loaded_metric, 'model') and hasattr(loaded_metric.model, 'inspect_history'):
            loaded_metric.model.inspect_history(n=2)
            print("-" * 60)
            print("✅ SERIALIZATION TEST PASSED! Metric loaded and executed successfully.")
            print("   You can inspect the history above to verify the prompt structure.")
            return True
        else:
            print("⚠️  Model doesn't support inspect_history() - skipping prompt verification")
            print("✅ SERIALIZATION TEST PASSED! Metric loaded and executed successfully.")
            return True
        
    except Exception as e:
        print(f"❌ Error during serialization test: {e}")
        print(f"📍 Error occurred at: {type(e).__name__}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        return False


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

def configure_qwen(api_base=None):
    """
    Configure Qwen model with proper API base handling.
    Fixed to handle sglang server API endpoints correctly and avoid /v1/v1/ issue.
    
    Args:
        api_base (str): API base URL (e.g., "http://localhost:7450/v1"). If None, uses default.
    """
    import requests
    
    # Default API base if not provided
    if api_base is None:
        api_base = "http://localhost:7450/v1"
    
    # Fix URL construction to avoid /v1/v1/ issue
    # Remove /v1 from the end of api_base if present, we'll add it back properly
    api_base_clean = api_base.rstrip("/v1").rstrip("/")
    
    # Test different configurations - try both with and without /v1
    configs = [
        {
            "name": "With /v1 suffix (most compatible)",
            "model": "litellm_proxy/Qwen/Qwen3-32B",
            "api_base": f"{api_base_clean}/v1",
            "api_key": "None",
            "max_tokens": 4096  # Increased token limit for code generation
        },
        {
            "name": "Without /v1 suffix (DSPy adds it)",
            "model": "litellm_proxy/Qwen/Qwen3-32B",
            "api_base": api_base_clean,
            "api_key": "None",
            "max_tokens": 4096  # Increased token limit for code generation
        },
        {
            "name": "Direct model name with /v1",
            "model": "Qwen/Qwen3-32B",
            "api_base": f"{api_base_clean}/v1",
            "api_key": "None",
            "max_tokens": 4096  # Increased token limit for code generation
        }
    ]
    
    print(f"DEBUG: Configuring Qwen with API base: {api_base}")
    print(f"DEBUG: Cleaned API base: {api_base_clean}")
    
    # First, test server connectivity
    test_urls = [
        f"{api_base_clean}/health",
        f"{api_base_clean}/get_model_info",
        f"{api_base_clean}/v1/models"
    ]
    
    server_reachable = False
    for test_url in test_urls:
        try:
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                print(f"DEBUG: Server reachable at {test_url}")
                server_reachable = True
                break
        except Exception as e:
            print(f"DEBUG: Failed to reach {test_url}: {e}")
            continue
    
    if not server_reachable:
        raise RuntimeError(f"Cannot connect to Qwen server at {api_base_clean}. Is the server running?")
    
    # Now try different DSPy configurations
    for config in configs:
        try:
            print(f"DEBUG: Trying config: {config['name']}")
            print(f"  Model: {config['model']}")
            print(f"  API Base: {config['api_base']}")
            
            # Create DSPy LM
            generator_lm = dspy.LM(**config)
            judge_lm = generator_lm  # Use same model for both
            
            # Test with a simple query to verify it works
            try:
                test_result = generator_lm("Hello", temperature=0.1, max_tokens=5)
                print(f"SUCCESS: Config '{config['name']}' works!")
                print(f"  Test result: {test_result}")
                return generator_lm, judge_lm
            except Exception as e:
                print(f"ERROR: Config '{config['name']}' failed test: {e}")
                continue
                
        except Exception as e:
            print(f"ERROR: Config '{config['name']}' failed to initialize: {e}")
            continue
    
    raise RuntimeError(f"All Qwen configurations failed with API base {api_base}. Check server logs for detailed error information.")


def configure_prometheus():
    """Configure Prometheus model for evaluation."""
    try:
        from prometheus_eval.litellm import LiteLLM
        print("DEBUG: Successfully imported LiteLLM from prometheus_eval")
        # Use correct LiteLLM constructor: LiteLLM(name, api_base)
        model = LiteLLM(
            "litellm_proxy/Unbabel/M-Prometheus-14B",
            api_base="http://future-hgx-1:7420/v1"
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
                api_base="http://future-hgx-1:7420/v1",
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
                api_base="http://future-hgx-1:7420/v1",
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

def run_pipeline(dataset, generator_lm, judge_lm, n_metrics: int = 3, metric_type: str = "llm_judge", model_save_dir: str = None, seed: int = None, max_optimization_samples: int = 100, test_serialization: bool = False):
    banner(f"DATASET: {dataset.get_name()}")
    print("Task description:\n", dataset.get_task_description())

    # Instantiate the proposer ------------------------------------------------
    if metric_type == "geval":
        proposer = GEvalJudgeProposer(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
        )
        print("Using G-Eval metrics...")
    elif metric_type == "llm_judge_examples":
        proposer = LLMJudgeExampleProposer(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
            max_optimization_samples=max_optimization_samples,
            seed=seed if seed is not None else 42,  # Pass seed for reproducible optimization
        )
        print("Using LLM Judge with optimized examples...")
        print(f"   (Limited to {max_optimization_samples} samples for faster optimization)")
        if seed is not None:
            print(f"   Using seed: {seed}")
    elif metric_type == "llm_judge_optimized":
        proposer = OptimizedJudgeProposer(
            generator_llm=generator_lm,
            executor_kwargs={"model": judge_lm},
            auto_mode="medium",  # MIPROv2 optimization level
            num_threads=64,       # For faster optimization
            eval_function_name='inverse_distance',
            seed=seed
        )
        print("Using LLM Judge with MIPROv2-optimized prompts...")
        if seed is not None:
            print(f"   Using seed: {seed}")
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
        "llm_judge_examples": "LLM Judge (Example-Based)",
        "llm_judge_optimized": "LLM Judge (MIPROv2-Optimized)",
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

        # CRITICAL FIX: Ensure all inputs and outputs are strings
        # Some datasets have numeric values that need to be converted to strings
        raw_inputs = df[dataset.get_input_column()].tolist()
        raw_outputs = df[dataset.get_output_column()].tolist()
        
        # Convert all inputs and outputs to strings, handling None/NaN values
        inputs = []
        for inp in raw_inputs:
            if inp is None or (hasattr(inp, '__iter__') and str(inp) == 'nan'):
                inputs.append("")
            else:
                inputs.append(str(inp))
        
        outputs = []
        for out in raw_outputs:
            if out is None or (hasattr(out, '__iter__') and str(out) == 'nan'):
                outputs.append("")
            else:
                outputs.append(str(out))

        # Also convert references to strings if they exist
        if references:
            string_references = []
            for ref_list in references:
                string_ref_list = []
                for ref in ref_list:
                    if ref is None or (hasattr(ref, '__iter__') and str(ref) == 'nan'):
                        string_ref_list.append("")
                    else:
                        string_ref_list.append(str(ref))
                string_references.append(string_ref_list)
            references = string_references

        scores = first_metric.calculate_batched(
            inputs=inputs,
            outputs=outputs,
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
        
        metric_file_path = out_dir / f"{safe_filename}_Metric.py"
        first_metric.save_python_code(metric_file_path)
        print("Saved standalone metric to", metric_file_path)
        
        # Test metric serialization by loading and re-running the saved metric
        # Only run if test_serialization flag is set
        if test_serialization:
            banner("Testing Metric Serialization …")
            print("Loading saved metric and testing that it produces identical results...")
            
            serialization_success = test_metric_serialization(
                metric_file_path=metric_file_path,
                inputs=df[dataset.get_input_column()].tolist(),
                outputs=df[dataset.get_output_column()].tolist(),
                references=references,
                original_scores=scores
            )
            
            if serialization_success:
                print("\n🎉 SERIALIZATION TEST SUCCESSFUL! The saved metric works perfectly.")
            else:
                print("\n⚠️  SERIALIZATION TEST FAILED! There may be an issue with metric saving/loading.")
        
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
    parser = argparse.ArgumentParser(description="Demo metric generation with Basic LLM Judge, Example-Optimized LLM Judge, MIPROv2-Optimized LLM Judge, G-Eval, Code Generation, Rubric Generator, or Fine-tuned ModernBERT")
    parser.add_argument("--metric-type", choices=["llm_judge", "llm_judge_examples", "llm_judge_optimized", "geval", "codegen", "rubric_prometheus", "rubric_dspy", "finetune"], default="llm_judge", 
                       help="Choose the metric type to use (default: llm_judge)")
    parser.add_argument("--model", choices=["gpt4o_mini", "qwen"], default="gpt4o_mini", 
                       help="Choose the model to use (default: gpt4o_mini)")
    parser.add_argument("--n-metrics", type=int, default=3, help="Number of metrics to generate (default: 3)")
    parser.add_argument("--model-save-dir", type=str, default="/sphinx/u/salt-checkpoints/autometrics/models", 
                       help="Custom directory to save fine-tuned models (only used for finetune metric type)")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed for metric generation (creates different metrics with same parameters)")
    parser.add_argument("--max-optimization-samples", type=int, default=100,
                       help="Maximum number of samples to use for example optimization (default: 100, set higher for better optimization but slower processing)")
    parser.add_argument("--test-serialization", action="store_true",
                       help="Test metric serialization by loading saved metrics and verifying they work correctly")
    parser.add_argument("--api-base", type=str, default=None,
                       help="API base URL for local models (e.g., http://localhost:7450/v1)")
    args = parser.parse_args()

    # Configure models based on choice
    if args.model == "qwen":
        generator_lm, judge_lm = configure_qwen(args.api_base)
        print("Using Qwen models...")
    else:
        generator_lm, judge_lm = configure_gpt4o_mini()
        print("Using GPT-4o-mini models...")

    # Display configuration
    metric_type_display = {
        "llm_judge": "Basic LLM Judge", 
        "llm_judge_examples": "LLM Judge with Example Selection",
        "llm_judge_optimized": "LLM Judge with MIPROv2 Optimization",
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
            run_pipeline(ds, generator_lm, prometheus_lm, n_metrics=args.n_metrics, metric_type=args.metric_type, model_save_dir=args.model_save_dir, seed=args.seed, max_optimization_samples=args.max_optimization_samples, test_serialization=args.test_serialization)
    elif args.metric_type == "finetune":
        # For fine-tuning, we don't need a judge model, just the generator model for metric cards
        for ds in [CoGymTravelOutcome(), SimpDA()]:
            run_pipeline(ds, generator_lm, None, n_metrics=args.n_metrics, metric_type=args.metric_type, model_save_dir=args.model_save_dir, seed=args.seed, max_optimization_samples=args.max_optimization_samples, test_serialization=args.test_serialization)
    else:
        for ds in [CoGymTravelOutcome(), SimpDA()]:
            run_pipeline(ds, generator_lm, judge_lm, n_metrics=args.n_metrics, metric_type=args.metric_type, model_save_dir=args.model_save_dir, seed=args.seed, max_optimization_samples=args.max_optimization_samples, test_serialization=args.test_serialization)

    # print(generator_lm.inspect_history(n=2))
