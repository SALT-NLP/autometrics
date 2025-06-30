from autometrics.generator.Generator import Generator
from autometrics.metrics.generated.GeneratedCodeMetric import (
    GeneratedRefFreeCodeMetric,
    GeneratedRefBasedCodeMetric
)
import dspy
from typing import Optional, Callable

# Utilities to avoid duplication and enable reuse across generators
from autometrics.generator.utils import (
    get_good_bad_examples,
    generate_axes_of_variation,
)

from autometrics.util.format import get_default_formatter

# DSPy signatures for code generation
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

metric_name: "Character_Length"
code: ```python
return len(output)
```

As another example -- if you think that the model output should contain all words in the input, then this would be a good metric:

metric_name: "Contains_All_Words"
code: ```python
return all(word in output.lower() for word in input.lower().split())
```

or even better (because scaling is more useful than binary):
metric_name: "Coverage"
code: ```python
return len(set(output.lower().split()) & set(input.lower().split())) / len(set(input.lower().split()))
```

Your metric can be a simple function like above or it can be a more complex function spanning multiple lines and using the following pre-imported libraries:

    - numpy/np (for numerical operations)
    - math (for mathematical functions)
    - statistics (for statistical functions)
    - re (for regular expressions)
    - Counter, defaultdict (from collections)
    - itertools (for iteration tools)
    - scipy (for scientific operations, if available)
    - nltk (for NLP operations, if available)

These packages are already imported, so you can use them directly (e.g., np.mean(), math.sqrt(), Counter(), etc.). You can also import additional packages if needed, but the above are readily available.

IMPORTANT!!! DO NOT DEFINE A FUNCTION TO BE RUN WITH ANY OTHER NAME THAN compute_score.  Ideally you should not define any functions at all (just output the contents of the compute_score function), but if you do, make sure to call it compute_score.  Otherwise the code will not be runnable."""

    task_description: str = dspy.InputField(desc="A description of the task that the model is trying to solve.")
    measurement_name: str = dspy.InputField(desc="The name of the measurement.")
    good_examples: list = dspy.InputField(desc="A list of good examples of outputs for a model.")
    bad_examples: list = dspy.InputField(desc="A list of bad examples of outputs for a model.")
    metric_name: str = dspy.OutputField(desc="The name of the metric that the LLM is inventing in order to correlate with the measurement.")
    code: str = dspy.OutputField(desc="The code that will compute a score for the metric.")


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

metric_name: "Character_Length"
code: ```python
return len(output)
```

As another example -- if you think that the model output should contain all words in the input, then this would be a good metric:

metric_name: "Contains_All_Words"
code: ```python
return all(word in output.lower() for word in input.lower().split())
```

or even better (because scaling is more useful than binary):
metric_name: "Coverage"
code: ```python
return len(set(output.lower().split()) & set(input.lower().split())) / len(set(input.lower().split()))
```

Your metric can be a simple function like above or it can be a more complex function spanning multiple lines and using the following pre-imported libraries:

    - numpy/np (for numerical operations)
    - math (for mathematical functions)
    - statistics (for statistical functions)
    - re (for regular expressions)
    - Counter, defaultdict (from collections)
    - itertools (for iteration tools)
    - scipy (for scientific operations, if available)
    - nltk (for NLP operations, if available)

These packages are already imported, so you can use them directly (e.g., np.mean(), math.sqrt(), Counter(), etc.). You can also import additional packages if needed, but the above are readily available.

IMPORTANT!!! DO NOT DEFINE A FUNCTION TO BE RUN WITH ANY OTHER NAME THAN compute_score.  Ideally you should not define any functions at all (just output the contents of the compute_score function), but if you do, make sure to call it compute_score.  Otherwise the code will not be runnable."""

    task_description: str = dspy.InputField(desc="A description of the task that the model is trying to solve.")
    measurement_name: str = dspy.InputField(desc="The name of the measurement.")
    good_examples: list = dspy.InputField(desc="A list of good examples of outputs for a model.")
    bad_examples: list = dspy.InputField(desc="A list of bad examples of outputs for a model.")
    metric_name: str = dspy.OutputField(desc="The name of the metric that the LLM is inventing in order to correlate with the measurement.")
    code: str = dspy.OutputField(desc="The code that will compute a score for the metric.")


class CodeGenerator(Generator):
    """Generate *Code-based* metrics by proposing axes of variation and generating executable Python code.

    The class conforms to the new *Generator* interface which includes an
    optional *generator_llm* (used here to generate axes and code) as well as the
    ability to specify a custom executor class. The executor class is automatically
    determined based on whether the dataset has reference columns.
    """

    def __init__(
        self,
        name: str = "CodeGenerator",
        description: str = "Propose code-based metrics based on the dataset and task description",
        generator_llm: Optional[dspy.LM] = None,
        executor_class: type | None = None,
        executor_kwargs: dict | None = None,
        seed: Optional[int] = None,
    ):

        super().__init__(
            name=name,
            description=description,
            generator_llm=generator_llm,
            executor_class=executor_class,
            executor_kwargs=executor_kwargs or {},
        )

        # Store seed for temperature-based cache busting
        self.seed = seed

        # Guarantee attribute is a dictionary for ** expansion later
        if self.executor_kwargs is None:
            self.executor_kwargs = {}

    def _get_formatter(self, dataset):
        if not dataset:
            return lambda x: str(x)
        return get_default_formatter(dataset)

    def _determine_executor_class(self, dataset):
        """Determine whether to use reference-based or reference-free metrics based on dataset."""
        reference_columns = dataset.get_reference_columns()
        has_references = reference_columns is not None and len(reference_columns) > 0
        
        if has_references:
            return GeneratedRefBasedCodeMetric
        else:
            return GeneratedRefFreeCodeMetric
    
    # Get Good and Bad Examples formatted properly
    def _preprocess_dataset(self, dataset, target_measure, formatter: Optional[Callable] = None):  # type: ignore[override]
        if not formatter:
            formatter = self._get_formatter(dataset)

        df = dataset.get_dataframe()
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]

        good_examples, bad_examples = get_good_bad_examples(df, target_measure)

        good_examples_formatted = [formatter(row) for row in good_examples.iterrows()]
        bad_examples_formatted = [formatter(row) for row in bad_examples.iterrows()]

        return good_examples_formatted, bad_examples_formatted
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and extract Python code from LLM output using the proven approach"""
        # Import the header stripping function from GeneratedCodeMetric
        from autometrics.metrics.generated.GeneratedCodeMetric import _strip_header_and_dedent
        
        # Use the proven approach to clean the code
        return _strip_header_and_dedent(code)
    
    def generate(self, dataset, target_measure: Optional[str] = None, n_metrics: int = 5, formatter: Optional[Callable] = None, **kwargs):
        """
        Generate new code-based metrics based on the dataset and task description.
        Automatically detects if the dataset has references and uses the appropriate metric class.
        """

        task_description = dataset.get_task_description()

        if not formatter:
            formatter = self._get_formatter(dataset)
        
        # Step-1: Determine the appropriate executor class based on dataset
        if self.executor_class is None:
            dynamic_executor_class = self._determine_executor_class(dataset)
        else:
            dynamic_executor_class = self.executor_class
        
        # Step-2: Prepare / cache dataset & formatter ---------------------------------
        good_examples_formatted, bad_examples_formatted = self._preprocess_dataset(dataset, target_measure, formatter)

        # Step-3: Ask the language model to propose axes -----------------------------
        axes = generate_axes_of_variation(
            task_description=task_description,
            good_examples=good_examples_formatted,
            bad_examples=bad_examples_formatted,
            generator_llm=self.generator_llm,
            target_name=target_measure,
            num_axes_to_generate=n_metrics,
            seed=self.seed,
        )

        axes = axes[:n_metrics] if n_metrics else axes

        # Step-4: Generate code for each axis and wrap in appropriate metric --------
        new_metrics = []
        
        # Determine if this is reference-based or reference-free
        is_reference_based = dynamic_executor_class == GeneratedRefBasedCodeMetric
        
        for axis in axes:
            # Extract metric name from axis - handle both *Name* and **Name** formats
            name_part = axis.split(":")[0].strip()
            if "**" in name_part:
                metric_name = name_part.split("**")[1].replace("*", "") + "_code"
            elif "*" in name_part:
                metric_name = name_part.split("*")[1].replace("*", "") + "_code"
            else:
                metric_name = name_part + "_code"

            # Generate code using appropriate signature
            try:
                # Set temperature based on seed for cache busting
                temperature = 0.0001 * self.seed if self.seed is not None else None
                
                if temperature is not None:
                    with dspy.settings.context(lm=self.generator_llm, temperature=temperature):
                        if is_reference_based:
                            code_gen = dspy.ChainOfThought(CodeGenReferenceBasedSignature, max_tokens=10000)
                        else:
                            code_gen = dspy.ChainOfThought(CodeGenReferenceFreeSignature, max_tokens=10000)
                        
                        result = code_gen(
                            task_description=task_description,
                            measurement_name=axis,
                            good_examples=good_examples_formatted,
                            bad_examples=bad_examples_formatted
                        )
                else:
                    with dspy.settings.context(lm=self.generator_llm):
                        if is_reference_based:
                            code_gen = dspy.ChainOfThought(CodeGenReferenceBasedSignature, max_tokens=10000)
                        else:
                            code_gen = dspy.ChainOfThought(CodeGenReferenceFreeSignature, max_tokens=10000)
                        
                        result = code_gen(
                            task_description=task_description,
                            measurement_name=axis,
                            good_examples=good_examples_formatted,
                            bad_examples=bad_examples_formatted
                        )
                    
                generated_metric_name = result.metric_name.replace(" ", "_").replace("-", "_").replace("\"", "").replace("*", "").replace("\'", "")
                generated_code = self._clean_generated_code(result.code)

                # Skip metrics with too many imports (likely malformed)
                if generated_code.count("import") > 10:
                    continue

                # Generate metric name and description (removed generator_llm_name reference)
                # metric_name is already set above as generated_metric_name
                
                # Validate and reconcile seed values
                executor_kwargs = self.executor_kwargs.copy()
                if self.seed is not None:
                    if 'seed' in executor_kwargs and executor_kwargs['seed'] != self.seed:
                        print(f"Warning: Seed mismatch detected. Proposer seed ({self.seed}) differs from executor_kwargs seed ({executor_kwargs['seed']}). Using proposer seed.")
                    executor_kwargs['seed'] = self.seed
                elif 'seed' not in executor_kwargs:
                    # No seed provided anywhere, that's fine
                    pass

                # Create the metric instance
                metric = dynamic_executor_class(
                    name=f"{generated_metric_name}_Generated",
                    description=f"Generated code-based metric for {axis}",
                    generated_code=generated_code,
                    task_description=task_description,
                    measurement_axis=axis,
                    metric_card_author_model=self.generator_llm,
                    **executor_kwargs,
                )

                # Test the metric on a sample example
                test_example = dataset.get_dataframe().iloc[0]
                test_input = test_example[dataset.get_input_column()]
                test_output = test_example[dataset.get_output_column()]
                
                if is_reference_based:
                    ref_cols = dataset.get_reference_columns()
                    test_references = test_example[ref_cols].tolist() if ref_cols else None
                else:
                    test_references = None

                # Try to run the metric
                test_score = metric.calculate(test_input, test_output, test_references)
                new_metrics.append(metric)

            except Exception as e:
                print(f"Error generating metric for {axis}: {e}")
                continue

        return new_metrics

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__() 