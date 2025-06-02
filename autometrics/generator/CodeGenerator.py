import dspy
from autometrics.generator.Generator import Generator
from autometrics.util.format import get_default_formatter
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# This is a code generator that uses a LLM to generate code.
# It is used to generate code for the metrics.

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

class CodeGenReferenceBasedSignature(dspy.Signature):
    """Given a task description, a measurement name, and a list of good and bad examples, generate code that will compute a useful score for the metric.

Surround the code with ```python and ``` to make it easier to read.

The code will plug into a method with the following signature:
def compute_score(input: str, output: str, references: list[str] = None) -> float:
    '''
    Compute a score for the metric.
    input: The input text that the model was given as an instruction or source text.
    output: The output text that the model generated.
    references: A list of reference outputs that showcase optimal outputs (often human generated).
    '''
    pass
    
You do not need to output the method header, just the code.

For example if you think that character level length would correlate highly with the measure this would be a good output:

metric_name: "character_length"
code: ```python
return len(output)
```

As another example -- if you think that the model output should contain all words in the input, then this would be a good metric:

metric_name: "contains_all_words"
code: ```python
return all(word in output.lower() for word in input.lower().split())
```

or even better (because scaling is more useful than binary):
metric_name: "coverage"
code: ```python
return len(set(output.lower().split()) & set(input.lower().split())) / len(set(input.lower().split()))
```

Your metric can be a simple function like above or it can be a more complex function spanning multiple lines and using the following libraries:

- numpy (for numerical operations)
- Standard Python libraries (math, re, collections, etc.)

Do not use any other external libraries like pandas, scipy, sklearn, or nltk as they may not be available in all environments.

You will be provided some examples of good and bad model outputs. Pay attention to the formatting of these inputs and outputs because they may inform your code. For instance if the model output is code this changes the possibilities of what your evaluation can assess."""

    task_description = dspy.InputField(desc="A description of the task that the model is trying to solve.")
    measurement_name = dspy.InputField(desc="The name of the measurement.")
    good_examples = dspy.InputField(desc="A list of good examples of outputs for a model.")
    bad_examples = dspy.InputField(desc="A list of bad examples of outputs for a model.")
    metric_name = dspy.OutputField(desc="The name of the metric that the LLM is inventing in order to correlate with the measurement.")
    code = dspy.OutputField(desc="The code that will compute a score for the metric.")

class CodeGenReferenceFreeSignature(dspy.Signature):
    """Given a task description, a measurement name, and a list of good and bad examples, generate code that will compute a useful score for the metric.

Surround the code with ```python and ``` to make it easier to read.

The code will plug into a method with the following signature:
def compute_score(input: str, output: str) -> float:
    '''
    Compute a score for the metric.
    input: The input text that the model was given as an instruction or source text.
    output: The output text that the model generated.
    '''
    pass
    
You do not need to output the method header, just the code.

For example if you think that character level length would correlate highly with the measure this would be a good output:

metric_name: "character_length"
code: ```python
return len(output)
```

As another example -- if you think that the model output should contain all words in the input, then this would be a good metric:

metric_name: "contains_all_words"
code: ```python
return all(word in output.lower() for word in input.lower().split())
```

or even better (because scaling is more useful than binary):
metric_name: "coverage"
code: ```python
return len(set(output.lower().split()) & set(input.lower().split())) / len(set(input.lower().split()))
```

Your metric can be a simple function like above or it can be a more complex function spanning multiple lines and using the following libraries:

- numpy (for numerical operations)
- Standard Python libraries (math, re, collections, etc.)

Do not use any other external libraries like pandas, scipy, sklearn, or nltk as they may not be available in all environments.

You will be provided some examples of good and bad model outputs. Pay attention to the formatting of these inputs and outputs because they may inform your code. For instance if the model output is code this changes the possibilities of what your evaluation can assess."""

    task_description = dspy.InputField(desc="A description of the task that the model is trying to solve.")
    measurement_name = dspy.InputField(desc="The name of the measurement.")
    good_examples = dspy.InputField(desc="A list of good examples of outputs for a model.")
    bad_examples = dspy.InputField(desc="A list of bad examples of outputs for a model.")
    metric_name = dspy.OutputField(desc="The name of the metric that the LLM is inventing in order to correlate with the measurement.")
    code = dspy.OutputField(desc="The code that will compute a score for the metric.")

class CodeGenReferenceBased(dspy.Module):
    def __init__(self):
        self.code_gen = dspy.ChainOfThought(CodeGenReferenceBasedSignature)

    def forward(self, task_description: str, measurement_name: str, good_examples: list[str], bad_examples: list[str]):
        output = self.code_gen.forward(task_description=task_description, measurement_name=measurement_name, good_examples=good_examples, bad_examples=bad_examples)
        return output.metric_name, output.code

class CodeGenReferenceFree(dspy.Module):
    def __init__(self):
        self.code_gen = dspy.ChainOfThought(CodeGenReferenceFreeSignature)

    def forward(self, task_description: str, measurement_name: str, good_examples: list[str], bad_examples: list[str]):
        output = self.code_gen.forward(task_description=task_description, measurement_name=measurement_name, good_examples=good_examples, bad_examples=bad_examples)
        return output.metric_name, output.code

class CodeGenerator(Generator):
    def __init__(self, name="CodeGenerator", description="Generate new code-based metrics using LLM", train_dataset=None, task_description=None, formatter=None, proposer_model=None, generate_axes=True):
        self.task_description = task_description
        self.dataset = train_dataset
        self.proposer_model = proposer_model
        self.formatter = formatter
        self.generate_axes = generate_axes
        
        if formatter is None:
            self.formatter = self._get_formatter(train_dataset)

        super().__init__(name, description)

    def _get_formatter(self, dataset):
        if self.formatter:
            return self.formatter
        if not dataset:
            return lambda x: str(x)
        return get_default_formatter(dataset)
    
    def _preprocess_dataset(self, dataset, target_column):
        """Get good and bad examples formatted properly"""
        dataset = dataset or self.dataset
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
        """Generate axes of variation to create multiple metrics"""
        response = None
        with dspy.settings.context(lm=self.proposer_model):
            response = GenerateAxisOfVariation()(task_description=self.task_description, good_examples=good_examples, bad_examples=bad_examples)
        return response.axes_of_variation

    def generate(self, train_dataset=None, target_column=None, metric_type="both", max_workers=4, **kwargs):
        """
        Generate new code-based metrics based on the dataset and task description.
        
        Args:
            train_dataset: Dataset to analyze for generating metrics
            target_column: Column to use for identifying good vs bad examples
            metric_type: "reference_based", "reference_free", or "both"
            max_workers: Number of parallel workers for code generation
            **kwargs: Additional arguments
        
        Returns:
            List of GeneratedCodeMetric instances
        """
        # Import here to avoid circular imports
        from autometrics.metrics.generated.GeneratedCodeMetric import GeneratedCodeReferenceBasedMetric, GeneratedCodeReferenceFreeMetric
        
        good_examples_formatted, bad_examples_formatted = self._preprocess_dataset(train_dataset, target_column)

        # Generate measurement axes if enabled
        if self.generate_axes:
            axes_of_variation = self._get_axes_of_variation(good_examples_formatted, bad_examples_formatted)
        elif self.generate_axes is False and target_column:
            axes_of_variation = [target_column]
        else:
            # Use a simple default axis
            axes_of_variation = ["Quality: Overall quality of the output"]

        new_metrics = []

        # Determine which metric types to generate
        generate_reference_based = metric_type in ["reference_based", "both"]
        generate_reference_free = metric_type in ["reference_free", "both"]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for axis in axes_of_variation:
                measurement_name = axis.split(":")[0].replace("*", "").strip()
                
                if generate_reference_based:
                    code_gen_ref = CodeGenReferenceBased()
                    futures.append((
                        executor.submit(
                            code_gen_ref.forward,
                            self.task_description,
                            measurement_name,
                            good_examples_formatted,
                            bad_examples_formatted
                        ),
                        "reference_based",
                        measurement_name,
                        axis
                    ))
                
                if generate_reference_free:
                    code_gen_free = CodeGenReferenceFree()
                    futures.append((
                        executor.submit(
                            code_gen_free.forward,
                            self.task_description,
                            measurement_name,
                            good_examples_formatted,
                            bad_examples_formatted
                        ),
                        "reference_free",
                        measurement_name,
                        axis
                    ))

            # Collect results
            for future, metric_type_generated, measurement_name, axis in futures:
                try:
                    with dspy.settings.context(lm=self.proposer_model):
                        metric_name, code = future.result()
                    
                    # Clean the generated code
                    code = self._clean_generated_code(code)
                    
                    # Create appropriate metric instance
                    if metric_type_generated == "reference_based":
                        metric = GeneratedCodeReferenceBasedMetric(
                            name=f"{metric_name}_generated_ref",
                            description=f"Generated reference-based metric for {axis}",
                            generated_code=code,
                            task_description=self.task_description
                        )
                    else:  # reference_free
                        metric = GeneratedCodeReferenceFreeMetric(
                            name=f"{metric_name}_generated_free",
                            description=f"Generated reference-free metric for {axis}",
                            generated_code=code,
                            task_description=self.task_description
                        )
                    
                    new_metrics.append(metric)
                    
                except Exception as e:
                    print(f"Error generating metric for {measurement_name} ({metric_type_generated}): {e}")
                    continue

        return new_metrics

    def _clean_generated_code(self, code):
        """Clean and extract Python code from LLM output"""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Strip whitespace and ensure proper indentation
        lines = code.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove any common leading whitespace but preserve relative indentation
            cleaned_lines.append(line.rstrip())
        
        return '\n'.join(cleaned_lines)
    