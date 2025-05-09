from autometrics.experiments.experiment import Experiment
import dspy
from typing import Union, Literal, Callable
from autometrics.util.format import get_default_formatter
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.MultiMetric import MultiMetric
import pandas as pd
from autometrics.experiments.results import TabularResult
import litellm
from tqdm import tqdm

litellm.suppress_debug_info = True

class GeneratePerturbationStrategies(dspy.Signature):
    """You will be given:  
• A Task description  
• A Dimension to prioritize when perturbing outputs  
• The Example Input, optional Example Reference, and Example Output  

Instructions:  
Your primary focus should be on degrading performance along the specified Dimension.  
1. Begin with a rich reasoning paragraph (3–5 sentences) that explores a variety of ways to subtly degrade model outputs. Do **not** reference the specific example.  
2. Under the heading **Strategies:**, list **1–3** numbered, high-level perturbation strategies.  
   - Each strategy should be a short phrase (5–15 words) naming the category of change, followed by one concise sentence of abstract explanation.  
   - Do **not** include concrete rewrites, instance-specific examples, or example sentences.  

Task: Given a complicated original sentence, simplify it so a broader audience can easily understand it.  
Example Input: after the jerilderie raid, the gang laid low for 16 months evading capture.  
Example Reference: after the jerilderie raid, the gang laid low for 16 months avoiding capture.  
Example Output: after the jerilderie raid, the gang successfully hid for 16 months.  
Dimension: Meaning Preservation"""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    example_sets: list[str] = dspy.InputField(description="Example inputs, outputs, and (optionally) references showcasing the model's performance on the task")
    dimension: str = dspy.InputField(description="The dimension to prioritize for the perturbation (this should be the aspect of the model output that is most impacted by the perturbation)")
    perturbation_strategies: list[str] = dspy.OutputField(description="A list of perturbation strategies that can be used to test the robustness of the model")


class PerturbWorse(dspy.Signature):
    """You will be given:  
• A Task description  
• A Dimension to prioritize when perturbing outputs  
• The Example Input, optional Example Reference, and Model Output  
• A perturbation_strength value ("subtle" or "obvious")  
• A list of perturbation_strategies to apply  

Instructions:  
Your goal is to apply each strategy to the Model Output and produce a degraded version that specifically harms performance along the given Dimension, using the specified strength.  
Under the heading **Perturbed Outputs:**, return exactly one perturbed output per strategy.  
- For **subtle** strength, introduce minimal distortion.  
- For **obvious** strength, introduce more pronounced degradation.  
Do **not** include any reasoning, explanations, or examples -- only the perturbed text."""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    dimension: str = dspy.InputField(description="The dimension to prioritize for the perturbation (this should be the aspect of the model output that is most impacted by the perturbation)")
    input: str = dspy.InputField(description="The input provided to the model")
    references: Union[list[str], None] = dspy.InputField(description="The references of good outputs (may be None)")
    model_output: str = dspy.InputField(description="The output produced by the model")
    perturbation_strength: Literal["subtle", "obvious"] = dspy.InputField(description="The strength of the perturbation (subtle or obvious)")
    perturbation_strategies: list[str] = dspy.InputField(description="The perturbation strategies to use")
    perturbed_outputs: list[str] = dspy.OutputField(description="Perturbed text that is worse than the original model output.  Produce one perturbed output per strategy.")

class PerturbSame(dspy.Signature):
    """You will be given:  
• A Task description  
• A Dimension to preserve when perturbing outputs  
• The Example Input, optional Example Reference, and Model Output  
• A perturbation_strength value ("subtle" or "obvious")  
• A list of perturbation_strategies to apply  

Instructions:  
Your goal is to produce a perturbed version of the model output that specifically preserves performance along the given Dimension, using the specified strength.  
Under the heading **Perturbed Output:**, return exactly one perturbed output.
- For **subtle** strength, introduce minimal distortion.  
- For **obvious** strength, introduce more pronounced degradation.  
Do **not** include any reasoning, explanations, or examples -- only the perturbed text."""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    dimension: str = dspy.InputField(description="The dimension to preserve for the perturbation (this should be the aspect of the model output that is most impacted by the perturbation)")
    input: str = dspy.InputField(description="The input provided to the model")
    references: Union[list[str], None] = dspy.InputField(description="The references of good outputs (may be None)")
    model_output: str = dspy.InputField(description="The output produced by the model")
    perturbation_strength: Literal["subtle", "obvious"] = dspy.InputField(description="The strength of the perturbation (subtle or obvious)")
    perturbed_output: str = dspy.OutputField(description="Perturbed text that preserves performance along the given Dimension.")

class ProducePerturbations(dspy.Module):

    def __init__(self, num_examples: int = 3, formatter: Callable = None):
        self.generate_perturbation_strategies: GeneratePerturbationStrategies = dspy.ChainOfThought(GeneratePerturbationStrategies)
        self.perturb_worse: PerturbWorse = dspy.Predict(PerturbWorse)
        self.perturb_same: PerturbSame = dspy.Predict(PerturbSame)
        self.num_examples = num_examples
        self.formatter = formatter

    def forward(self, task: str, dimension: str, dataset: Dataset):

        if self.formatter is None:
            self.formatter = get_default_formatter(dataset)

        sampled_rows = dataset.get_dataframe().sample(self.num_examples) if self.num_examples < len(dataset.get_dataframe()) else dataset.get_dataframe()
        formatted_rows = [self.formatter(row) for row in sampled_rows.iterrows()]

        perturbation_strategies = self.generate_perturbation_strategies(task=task, dimension=dimension, example_sets=formatted_rows).perturbation_strategies

        overall_perturbed_worse_subtle = []
        overall_perturbed_worse_obvious = []
        overall_perturbed_same_subtle = []
        overall_perturbed_same_obvious = []

        for _, row in tqdm(dataset.get_dataframe().iterrows(), total=len(dataset.get_dataframe())):
            inp = row[dataset.get_input_column()]
            references = row[dataset.get_reference_columns()]
            model_output = row[dataset.get_output_column()]

            perturbed_worse_subtle = self.perturb_worse(task=task, dimension=dimension, input=inp, references=references, model_output=model_output, perturbation_strength="subtle", perturbation_strategies=perturbation_strategies).perturbed_outputs
            perturbed_worse_obvious = self.perturb_worse(task=task, dimension=dimension, input=inp, references=references, model_output=model_output, perturbation_strength="obvious", perturbation_strategies=perturbation_strategies).perturbed_outputs

            perturbed_same_subtle = self.perturb_same(task=task, dimension=dimension, input=inp, references=references, model_output=model_output, perturbation_strength="subtle", perturbation_strategies=perturbation_strategies).perturbed_output
            perturbed_same_obvious = self.perturb_same(task=task, dimension=dimension, input=inp, references=references, model_output=model_output, perturbation_strength="obvious", perturbation_strategies=perturbation_strategies).perturbed_output

            overall_perturbed_worse_subtle.append(perturbed_worse_subtle)
            overall_perturbed_worse_obvious.append(perturbed_worse_obvious)
            overall_perturbed_same_subtle.append(perturbed_same_subtle)
            overall_perturbed_same_obvious.append(perturbed_same_obvious)

        results = {
            "perturbed_worse_subtle": overall_perturbed_worse_subtle,
            "perturbed_worse_obvious": overall_perturbed_worse_obvious,
            "perturbed_same_subtle": overall_perturbed_same_subtle,
            "perturbed_same_obvious": overall_perturbed_same_obvious,
            "strategies": perturbation_strategies
        }

        return results

class RobustnessExperiment(Experiment):

    def _produce_perturbation_scores(self, dataset: Dataset, perturbations: dict[str, list[str]]) -> pd.DataFrame:
        worse_subtle, worse_obvious, same_subtle, same_obvious, strategies = perturbations["perturbed_worse_subtle"], \
                                                                                     perturbations["perturbed_worse_obvious"], \
                                                                                     perturbations["perturbed_same_subtle"], \
                                                                                     perturbations["perturbed_same_obvious"], \
                                                                                     perturbations["strategies"]
                
        amt_to_eval = (len(strategies) * 2) + 2
        
        inputs = dataset.get_dataframe()[dataset.get_input_column()].tolist()
        reference_columns = dataset.get_reference_columns()
        
        # Create a dictionary where each key is a reference column and value is the list of values

        inputs_structured = [[inputs[i]] * len(strategies) for i in range(len(inputs))]
        inputs_structured = [item for sublist in inputs_structured for item in sublist]

        data = {
            "input": inputs_structured + inputs_structured + inputs + inputs,
            "model_output": [],
            "strategy": (strategies * len(worse_subtle)) + (strategies * len(worse_obvious)) + (["same_subtle"] * len(same_subtle)) + (["same_obvious"] * len(same_obvious)),
            "group": ["worse_subtle"] * len(worse_subtle) * len(strategies) + ["worse_obvious"] * len(worse_obvious) * len(strategies) + ["same_subtle"] * len(same_subtle) + ["same_obvious"] * len(same_obvious)
        }
        
        # Add each reference column to the data dictionary, duplicated amt_to_eval times
        for ref_col in reference_columns:
            ref_values = dataset.get_dataframe()[ref_col].tolist()

            ref_values_structured = [[ref_values[i]] * len(strategies) for i in range(len(ref_values))]
            ref_values_structured = [item for sublist in ref_values_structured for item in sublist]

            data[ref_col] = ref_values_structured + ref_values_structured + ref_values + ref_values

        data["model_output"].extend([item for sublist in worse_subtle for item in sublist])
        data["model_output"].extend([item for sublist in worse_obvious for item in sublist])
        data["model_output"].extend(same_subtle)
        data["model_output"].extend(same_obvious)

        df = pd.DataFrame(data)

        df.to_csv("df.csv", index=False)

        for metric in self.metrics:
            original_values = dataset.get_metric_values(metric)
            true_outputs = dataset.get_dataframe()[dataset.get_output_column()]

            results = metric.calculate_batched(df["input"], df["model_output"], [[df[ref_col].iloc[i] for ref_col in reference_columns] for i in range(len(df))])

            # Check if the metric is a multi-metric
            if type(results[0]) == list and isinstance(metric, MultiMetric):
                for i, submetric_name in enumerate(metric.get_submetric_names()):
                    data[submetric_name] = results[i]
                    data[submetric_name].extend(original_values[i])
            else:
                data[metric.get_name()] = results
                data[metric.get_name()].extend(original_values)

        data["input"].extend(inputs)
        for ref_col in reference_columns:
            data[ref_col].extend(dataset.get_dataframe()[ref_col].tolist())
        data["model_output"].extend(true_outputs)
        data["strategy"].extend([["original"]] * len(inputs))
        data["group"].extend(["original"] * len(inputs))

        df = pd.DataFrame(data)
        return df

    def run(self, print_results: bool = False, num_demonstration_examples: int = 3, max_eval_examples: int = 30):

        test_dataset = self.test_dataset
        if max_eval_examples < len(test_dataset.get_dataframe()):
            test_dataset = test_dataset.get_subset(max_eval_examples, seed=self.seed)

        produce_perturbations = ProducePerturbations(num_examples=num_demonstration_examples)

        if self.kwargs.get("lm"):
            self.lm = self.kwargs.get("lm")
        else:
            self.lm = dspy.settings.lm

        with dspy.settings.context(lm=lm):
            for column in test_dataset.get_target_columns():
                # First, generate the perturbations
                perturbations = produce_perturbations(task=test_dataset.get_task_description(), dimension=column, dataset=test_dataset)
                df = self._produce_perturbation_scores(test_dataset, perturbations)
                self.results[f"full_table_{column}"] = TabularResult(df)

                if print_results:
                    print(df)

        print("TODO: The rest of the experiment is not implemented yet.")
        return

if __name__ == "__main__":

    from autometrics.dataset.datasets.simplification.simplification import SimpDA
    from autometrics.metrics.reference_based.BLEU import BLEU
    import os


    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)

    dataset = SimpDA()

    experiment = RobustnessExperiment(
        name="Robustness Experiment",
        description="An experiment to test the robustness of the model",
        metrics=[BLEU()],
        output_dir="outputs",
        dataset=dataset
    )

    experiment.run(print_results=True)

    experiment.save_results()