from autometrics.generator.Generator import Generator
from autometrics.metrics.llm_judge.LLMJudge import LLMJudge
import dspy
import re
from autometrics.util.format import get_default_formatter
from prometheus_eval.litellm import LiteLLM

def get_good_bad_examples(df, target_column, num_examples=5, flip=False):
    '''
        Get the good and bad examples (if flip is True, then the good examples are the ones with the lowest values)
    '''
    good_examples = df.sort_values(by=target_column, ascending=False).head(num_examples)
    bad_examples = df.sort_values(by=target_column, ascending=True).head(num_examples)

    if flip:
        return bad_examples, good_examples

    return good_examples, bad_examples

class GenerateAxisOfVariationSignature(dspy.Signature):
    """Given some good examples of outputs for a model and some bad examples, generate axes of variation that can explain some of the important differences related to the quality of the outputs.  Return a list of axes of variation from most important to least important alongside few word descriptions (part of the same list).  An additional description of the task is provided for context."""
    task_description = dspy.InputField(desc="A description of the task that the model is trying to solve.")
    target_name = dspy.InputField(desc="If provided, a brief suggestion of the overall target value we are trying to predict.  Can sometimes be useful for generating axes of variation, and other times be ignored (when 'None' or not useful).")
    good_examples = dspy.InputField(desc="A list of good examples of outputs for a model.")
    bad_examples = dspy.InputField(desc="A list of bad examples of outputs for a model.")
    axes_of_variation = dspy.OutputField(desc="A numbered list of five axes of variation from most important to least important including a few word description of each axis.")

class GenerateAxisOfVariation(dspy.Module):
    def __init__(self):
        super(GenerateAxisOfVariation, self).__init__()
        self.generate_axes = dspy.ChainOfThought(GenerateAxisOfVariationSignature)

    def forward(self, task_description, good_examples, bad_examples, target_name=None):
        if not target_name:
            target_name = "None"
        axes_of_variation = self.generate_axes(task_description=task_description, target_name=target_name, good_examples=good_examples, bad_examples=bad_examples).axes_of_variation

        # Split the axes of variation based on the newline, number, (optional period) pattern
        axes = re.split(r"\n\d+\.", axes_of_variation)

        # Remove any empty strings from the list and strip any leading or trailing whitespace
        axes = [axis.strip() for axis in axes if axis.strip()]

        # If axes[0] starts with 1. then strip it
        if axes[0].startswith("1."):
            axes[0] = axes[0][2:].strip()

        return dspy.Prediction(task_description=task_description, target_name=target_name, good_examples=good_examples, bad_examples=bad_examples, axes_of_variation=axes)

class LLMJudgeProposer(Generator):
    def __init__(self, name="LLMJudgeProposer", description= "Propose new llm as a judge metrics based on the dataset and task description", train_dataset=None, task_description=None, formatter=None, proposer_model=None, judge_model=None):
        self.task_description = task_description
        self.dataset = train_dataset
        self.proposer_model = proposer_model
        self.judge_model = judge_model
        self.formatter = formatter
        if judge_model and hasattr(judge_model, 'name'):
            self.judge_model_name = judge_model.name
        elif judge_model and isinstance(judge_model.model, LiteLLM):
            self.judge_model_name = judge_model.model.name
        else:
            self.judge_model_name = judge_model.model.split('/')[-1] if judge_model else "None"


        if formatter is None:
            self.formatter = self._get_formatter(train_dataset)

        super().__init__(name, description)

    def _get_formatter(self, dataset):
        if self.formatter:
            return self.formatter
        if not dataset:
            return lambda x: str(x)
        return get_default_formatter(dataset)
    
    # Get Good and Bad Examples formatted properly
    def _preprocess_dataset(self, dataset, target_column):
        dataset = dataset
        if dataset:
            self.dataset = dataset
            self.formatter = self._get_formatter(dataset)
        elif not self.dataset:
            raise ValueError("No dataset provided")

        if self.formatter is None:
            self.formatter = self._get_formatter(dataset)

        df = dataset.get_dataframe()
        if not target_column:
            target_column = dataset.get_target_columns()[0]

        good_examples, bad_examples = get_good_bad_examples(df, target_column)

        good_examples_formatted = [self.formatter(row) for row in good_examples.iterrows()]
        bad_examples_formatted = [self.formatter(row) for row in bad_examples.iterrows()]

        return good_examples_formatted, bad_examples_formatted
    
    def _get_axes_of_variation(self, good_examples, bad_examples):
        response = None
        with dspy.settings.context(lm=self.proposer_model):
            response = GenerateAxisOfVariation()(task_description=self.task_description, good_examples=good_examples, bad_examples=bad_examples)

        return response.axes_of_variation


    def generate(self, train_dataset=None, target_column=None, **kwargs):
        """
        Generate new metrics based on the dataset and task description
        """
        
        good_examples_formatted, bad_examples_formatted = self._preprocess_dataset(train_dataset, target_column)

        axis_of_variation = self._get_axes_of_variation(good_examples_formatted, bad_examples_formatted)

        new_metrics = []
        for i, axis in enumerate(axis_of_variation):
            metric_name = axis.split(":")[0].replace("*", "") + "_" + self.judge_model_name

            new_metrics.append(LLMJudge(metric_name, f"{axis}", self.judge_model, self.dataset, axis, self.formatter, self.task_description))

        return new_metrics


    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return