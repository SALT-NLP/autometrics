from autometrics.metrics.llm_judge.LLMJudgeRubric import LLMJudgeRubric
from autometrics.generator.LLMJudgeProposer import LLMJudgeProposer
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from prometheus_eval import PrometheusEval

class GenerateRubricSignature(dspy.Signature):
    """Given a dataset, task description, and an evaluation metric, generate a rubric for the metric scoring from 1 to 5."""
    task_description = dspy.InputField(desc="A description of the task that the model is trying to solve.")
    good_examples = dspy.InputField(desc="A list of good examples of outputs for a model.")
    bad_examples = dspy.InputField(desc="A list of bad examples of outputs for a model.")
    metric_title = dspy.InputField(desc="The title of the metric.")
    metric_description = dspy.InputField(desc="A description of the metric.")

    score_one_description = dspy.OutputField(desc="A description of what a score of 1 means.")
    score_two_description = dspy.OutputField(desc="A description of what a score of 2 means.")
    score_three_description = dspy.OutputField(desc="A description of what a score of 3 means.")
    score_four_description = dspy.OutputField(desc="A description of what a score of 4 means.")
    score_five_description = dspy.OutputField(desc="A description of what a score of 5 means.")

class GenerateRubric(dspy.Module):
    def __init__(self):
        super(GenerateRubric, self).__init__()
        self.generate_rubric = dspy.ChainOfThought(GenerateRubricSignature)

    def forward(self, task_description, good_examples, bad_examples, metric_title, metric_description):
        rubric = self.generate_rubric(task_description=task_description, good_examples=good_examples, bad_examples=bad_examples, metric_title=metric_title, metric_description=metric_description)

        score_descriptions = [
            rubric.score_one_description,
            rubric.score_two_description,
            rubric.score_three_description,
            rubric.score_four_description,
            rubric.score_five_description
        ]

        return dspy.Prediction(criteria=metric_description, score_descriptions=score_descriptions)

class LLMJudgeRubricProposer(LLMJudgeProposer):
    def __init__(self, name="LLMJudgeRubricProposer", description="Propose new llm as a judge metrics and rubrics based on the dataset and task description", train_dataset=None, task_description=None, formatter=None, proposer_model=None, judge_model=None, judge_api_base="http://jagupard37:8000/v1"):
        self.judge_api_base = judge_api_base

        super().__init__(name, description, train_dataset, task_description, formatter, proposer_model, judge_model)

        if self.judge_model_name == 'None':
            self.judge_model_name = 'prometheus-7b-v2.0'

    def generate(self, train_dataset=None, target_column=None, **kwargs):
        """
        Generate new metrics based on the dataset and task description.
        """
        good_examples_formatted, bad_examples_formatted = self._preprocess_dataset(train_dataset, target_column)
        axis_of_variation = self._get_axes_of_variation(good_examples_formatted, bad_examples_formatted)

        new_metrics = []

        with ThreadPoolExecutor() as executor:
            futures = []
            rubric_generator = GenerateRubric()  # Instantiate once to ensure consistent state

            for axis in axis_of_variation:
                metric_title = axis.split(":")[0].replace("*", "")
                metric_description = axis
                metric_name = f"{metric_title}_{self.judge_model_name}_rubric"

                # Submit the forward method with its arguments
                futures.append(
                    executor.submit(
                        rubric_generator.forward,
                        self.task_description,
                        good_examples_formatted,
                        bad_examples_formatted,
                        metric_title,
                        metric_description
                    )
                )

            for future in as_completed(futures):
                try:
                    llm_rubric = future.result()

                    rubric = {
                        "criteria": llm_rubric.criteria,
                        "score1_description": llm_rubric.score_descriptions[0],
                        "score2_description": llm_rubric.score_descriptions[1],
                        "score3_description": llm_rubric.score_descriptions[2],
                        "score4_description": llm_rubric.score_descriptions[3],
                        "score5_description": llm_rubric.score_descriptions[4],
                    }

                    metric_name = f"{llm_rubric.criteria.split(':')[0].replace('*', '')}_{self.judge_model_name}_rubric"

                    if self.judge_model and isinstance(self.judge_model, PrometheusEval):
                        new_metrics.append(
                            LLMJudgeRubric(
                                metric_name,
                                llm_rubric.criteria,
                                train_dataset,
                                rubric,
                                judge=self.judge_model,
                                judge_api_base=self.judge_api_base
                            )
                        )
                    else:
                        new_metrics.append(
                            LLMJudgeRubric(
                                metric_name,
                                llm_rubric.criteria,
                                train_dataset,
                                rubric,
                                model=self.judge_model,
                                judge_api_base=self.judge_api_base
                            )
                        )

                except Exception as e:
                    print(f"Error generating rubric: {e}")

        return new_metrics