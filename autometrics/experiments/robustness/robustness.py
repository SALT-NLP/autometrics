# File: robustness/robustness.py

import hashlib
import pandas as pd
import dspy
import os

from autometrics.experiments.experiment import Experiment
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.experiments.results import TabularResult
from robustness.perturb import ProducePerturbations
from robustness.analysis import analyze_and_plot

class RobustnessExperiment(Experiment):

    def _produce_perturbation_scores(self, dataset, perturbations):
        worse_subtle, worse_obvious, same_subtle, same_obvious, strategies = (
            perturbations["perturbed_worse_subtle"],
            perturbations["perturbed_worse_obvious"],
            perturbations["perturbed_same_subtle"],
            perturbations["perturbed_same_obvious"],
            perturbations["strategies"],
        )

        inputs = dataset.get_dataframe()[dataset.get_input_column()].tolist()
        reference_columns = dataset.get_reference_columns()

        inputs_structured = [[inputs[i]] * len(strategies) for i in range(len(inputs))]
        inputs_structured = [item for sublist in inputs_structured for item in sublist]

        data = {
            "input": inputs_structured + inputs_structured + inputs + inputs,
            "model_output": [],
            "strategy": (strategies * len(worse_subtle))
                        + (strategies * len(worse_obvious))
                        + (["same_subtle"] * len(same_subtle))
                        + (["same_obvious"] * len(same_obvious)),
            "group": ["worse_subtle"] * len(worse_subtle) * len(strategies)
                     + ["worse_obvious"] * len(worse_obvious) * len(strategies)
                     + ["same_subtle"] * len(same_subtle)
                     + ["same_obvious"] * len(same_obvious),
        }

        for ref_col in reference_columns:
            ref_values = dataset.get_dataframe()[ref_col].tolist()
            ref_values_structured = [[ref_values[i]] * len(strategies) for i in range(len(ref_values))]
            ref_values_structured = [item for sublist in ref_values_structured for item in sublist]
            data[ref_col] = (
                ref_values_structured
                + ref_values_structured
                + ref_values
                + ref_values
            )

        data["model_output"].extend([item for sublist in worse_subtle for item in sublist])
        data["model_output"].extend([item for sublist in worse_obvious for item in sublist])
        data["model_output"].extend(same_subtle)
        data["model_output"].extend(same_obvious)

        df = pd.DataFrame(data)

        for metric in self.metrics:
            original_values = dataset.get_metric_values(metric)
            true_outputs = dataset.get_dataframe()[dataset.get_output_column()]

            results = metric.calculate_batched(
                df["input"],
                df["model_output"],
                [
                    [df[ref_col].iloc[i] for ref_col in reference_columns]
                    for i in range(len(df))
                ],
            )

            if isinstance(results, (list, tuple)) and isinstance(metric, MultiMetric):
                for i, submetric_name in enumerate(metric.get_submetric_names()):
                    data[submetric_name] = list(results[i])
                    data[submetric_name].extend(original_values[submetric_name])
            else:
                data[metric.get_name()] = results
                data[metric.get_name()].extend(original_values)

        data["input"].extend(inputs)
        for ref_col in reference_columns:
            data[ref_col].extend(dataset.get_dataframe()[ref_col].tolist())
        data["model_output"].extend(true_outputs)
        data["strategy"].extend([["original"]] * len(inputs))
        data["group"].extend(["original"] * len(inputs))

        return pd.DataFrame(data)

    def run(self, print_results=False, num_demonstration_examples=3, max_eval_examples=30, max_workers=8):
        test_dataset = self.test_dataset
        if max_eval_examples < len(test_dataset.get_dataframe()):
            test_dataset = test_dataset.get_subset(max_eval_examples, seed=self.seed)

        producer = ProducePerturbations(num_examples=num_demonstration_examples, max_workers=max_workers)

        if self.kwargs.get("lm"):
            self.lm = self.kwargs.get("lm")
        else:
            self.lm = dspy.settings.lm

        with dspy.settings.context(lm=self.lm):
            for column in test_dataset.get_target_columns():
                perturbations = producer.forward(
                    task=test_dataset.get_task_description(),
                    dimension=column,
                    dataset=test_dataset,
                )
                df = self._produce_perturbation_scores(test_dataset, perturbations)
                self.results[f"{column}/full_table"] = TabularResult(df)

                if print_results:
                    print(df)

                df["sample_id"] = (
                    df["input"]
                    .str.strip()
                    .str.lower()
                    .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
                )
                analyze_and_plot(df, self.metrics, column, self.results)

def main():
    import os
    from autometrics.dataset.datasets.simplification.simplification import SimpDA
    from autometrics.metrics.reference_based.BLEU import BLEU
    from autometrics.metrics.reference_based.SARI import SARI

    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)

    dataset = SimpDA()

    experiment = RobustnessExperiment(
        name="Robustness Experiment",
        description="An experiment to test the robustness of the model",
        metrics=[BLEU(), SARI()],
        output_dir="outputs/robustness",
        dataset=dataset,
    )

    experiment.run(print_results=True)
    experiment.save_results()

if __name__ == "__main__":
    main()