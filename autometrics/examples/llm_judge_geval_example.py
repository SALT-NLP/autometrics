"""
Example usage of LLMJudgeGEval metric.

This example demonstrates how to use the LLMJudgeGEval class which implements
the G-Eval methodology for LLM-as-a-judge evaluation with proper integration
into the autometrics framework.
"""

import dspy
import pandas as pd
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.llm_judge.LLMJudgeGEval import LLMJudgeGEval

def main():
    # Create a sample dataset
    data = {
        'input': [
            "Simplify this sentence: The bastion on the eastern approaches was built later.",
            "Translate this to simple English: The edifice exhibited remarkable architectural prowess.",
            "Make this clearer: The utilization of sophisticated methodologies was implemented."
        ],
        'output': [
            "A fort on the eastern side was built later.",
            "The building showed great design skills.", 
            "We used advanced methods."
        ],
        'reference': [
            "A fortress on the east was built afterward.",
            "The building had amazing architecture.",
            "Advanced techniques were used."
        ]
    }
    
    df = pd.DataFrame(data)
    dataset = Dataset(
        dataframe=df,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="simplification_dataset",
        input_column='input',
        output_column='output', 
        reference_columns=['reference']
    )
    
    # Initialize models - could use different models for different purposes
    evaluation_model = dspy.LM(
        "litellm_proxy/meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_base="http://future-hgx-1:7400/v1", 
        api_key="None"
    )
    
    # Example 1: Single model for both criteria generation and evaluation
    print("=== Example 1: Single Model Approach ===")
    geval_single = LLMJudgeGEval(
        name="clarity_geval_single",
        description="G-Eval metric using single model for both generation and evaluation",
        dataset=dataset,
        evaluation_criteria="clarity and simplicity of the text",
        model=evaluation_model,  # Used for both criteria generation and evaluation
        task_description="Simplify complex sentences to make them more understandable",
        auto_generate_steps=True
    )
    
    print(f"Single model approach - Evaluation steps generated:")
    print(f"{geval_single.evaluation_steps}\n")
    
    # Example 2: Dual model approach (in practice, you'd use different models)
    print("=== Example 2: Dual Model Approach ===")
    # For demonstration, we'll use the same model but in practice you could use:
    # - A more capable model (like GPT-4) for criteria generation
    # - A faster model (like Llama-3.1-8B) for bulk evaluation
    criteria_generation_model = evaluation_model  # In practice: different model
    
    geval_dual = LLMJudgeGEval(
        name="clarity_geval_dual",
        description="G-Eval metric using separate models for generation and evaluation",
        dataset=dataset,
        evaluation_criteria="clarity and simplicity of the text",
        model=evaluation_model,  # Fast model for bulk evaluation
        criteria_generation_model=criteria_generation_model,  # Capable model for criteria
        task_description="Simplify complex sentences to make them more understandable",
        auto_generate_steps=True
    )
    
    print(f"Dual model approach - Evaluation steps generated:")
    print(f"{geval_dual.evaluation_steps}\n")
    
    # Example 3: Pre-defined steps (no criteria generation)
    print("=== Example 3: Pre-defined Steps (No Criteria Generation) ===")
    geval_predefined = LLMJudgeGEval(
        name="clarity_geval_predefined",
        description="G-Eval metric with predefined evaluation steps",
        dataset=dataset,
        evaluation_criteria="clarity and simplicity of the text",
        model=evaluation_model,
        task_description="Simplify complex sentences to make them more understandable",
        evaluation_steps="""1. Read the original complex sentence and the simplified version.
2. Evaluate how much clearer and simpler the output is compared to the input.
3. Consider vocabulary simplification, sentence structure, and overall readability.
4. Assign a score from 1 (no improvement) to 5 (significantly clearer and simpler).""",
        auto_generate_steps=False  # Skip automatic generation
    )
    
    print(f"Pre-defined steps approach:")
    print(f"{geval_predefined.evaluation_steps}\n")
    
    # Calculate scores for the dataset using the single model approach
    print("Calculating G-Eval scores using single model approach...")
    scores = geval_single.predict(dataset, update_dataset=True, num_workers=4)
    
    print(f"\nG-Eval Scores: {scores}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    
    # Show updated dataset
    print("\nUpdated dataset:")
    print(dataset.get_dataframe())
    
    # Example of single calculation with dual model setup
    print("\nExample single calculation using dual model setup:")
    single_score = geval_dual.calculate(
        input="Make this simpler: The methodology was quite sophisticated.",
        output="The method was very advanced.",
        references=["The approach was complex."]
    )
    print(f"Single score: {single_score}")
    
    # Demonstrate the model separation
    print(f"\nModel Information:")
    print(f"Evaluation model: {geval_dual.model.model}")
    print(f"Criteria generation model: {geval_dual.criteria_generation_model.model}")
    print(f"Same model used for both: {geval_dual.model is geval_dual.criteria_generation_model}")

if __name__ == "__main__":
    main() 