from autometrics.generator.Generator import Generator
import dspy
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Utilities to avoid duplication and enable reuse across generators
from autometrics.generator.utils import (
    get_good_bad_examples,
    generate_axes_of_variation,
)

from autometrics.util.format import get_default_formatter

# DSPy signature for rubric generation
class GenerateRubricSignature(dspy.Signature):
    """Given a dataset, task description, and an evaluation metric, generate a rubric for the metric scoring from 1 to 5."""
    task_description = dspy.InputField(desc="A description of the task that the model is trying to solve.")
    good_examples = dspy.InputField(desc="A list of good examples of outputs for a model.")
    bad_examples = dspy.InputField(desc="A list of bad examples of outputs for a model.")
    metric_title = dspy.InputField(desc="The title of the metric.")
    metric_description = dspy.InputField(desc="A description of the metric.")

    score_one_description = dspy.OutputField(desc="A description of what a score of 1 means.  This can be a bullet point list of what criteria to look for in assigning a score of 1.")
    score_two_description = dspy.OutputField(desc="A description of what a score of 2 means.  This can be a bullet point list of what criteria to look for in assigning a score of 2.")
    score_three_description = dspy.OutputField(desc="A description of what a score of 3 means.  This can be a bullet point list of what criteria to look for in assigning a score of 3.")
    score_four_description = dspy.OutputField(desc="A description of what a score of 4 means.  This can be a bullet point list of what criteria to look for in assigning a score of 4.")
    score_five_description = dspy.OutputField(desc="A description of what a score of 5 means.  This can be a bullet point list of what criteria to look for in assigning a score of 5.")

class GenerateRubric(dspy.Module):
    def __init__(self):
        super(GenerateRubric, self).__init__()
        self.generate_rubric = dspy.ChainOfThought(GenerateRubricSignature)

    def forward(self, task_description, good_examples, bad_examples, metric_title, metric_description):
        rubric = self.generate_rubric(
            task_description=task_description, 
            good_examples=good_examples, 
            bad_examples=bad_examples, 
            metric_title=metric_title, 
            metric_description=metric_description
        )

        score_descriptions = [
            rubric.score_one_description,
            rubric.score_two_description,
            rubric.score_three_description,
            rubric.score_four_description,
            rubric.score_five_description
        ]

        return dspy.Prediction(criteria=metric_description, score_descriptions=score_descriptions)

class RubricGenerator(Generator):
    """Generate rubric-based metrics by proposing axes of variation and creating detailed rubrics.

    The class conforms to the Generator interface and can work with multiple executor classes:
    - GeneratedPrometheusMetric (for Prometheus-based evaluation)
    - GeneratedLLMJudgeMetric (for DSPy-based evaluation)
    
    The executor class is determined by the `use_prometheus` parameter or can be explicitly set.
    """

    def __init__(
        self,
        name: str = "RubricGenerator",
        description: str = "Propose rubric-based metrics based on the dataset and task description",
        generator_llm: Optional[dspy.LM] = None,
        executor_class: type | None = None,
        executor_kwargs: dict | None = None,
        use_prometheus: bool = True,
    ):

        super().__init__(
            name=name,
            description=description,
            generator_llm=generator_llm,
            executor_class=executor_class,
            executor_kwargs=executor_kwargs or {},
        )

        self.use_prometheus = use_prometheus

        # Guarantee attribute is a dictionary for ** expansion later
        if self.executor_kwargs is None:
            self.executor_kwargs = {}

        # Extract judge model information for naming
        if executor_kwargs and 'model' in executor_kwargs:
            judge_model = executor_kwargs['model']
            if judge_model and hasattr(judge_model, 'name'):
                self.judge_model_name = judge_model.name
            elif judge_model and hasattr(judge_model, 'model'):
                if hasattr(judge_model.model, 'name'):
                    self.judge_model_name = judge_model.model.name
                else:
                    self.judge_model_name = judge_model.model.split('/')[-1]
        else:
            self.judge_model_name = "M-Prometheus-14B" if use_prometheus else "UnknownLLM"

        # Keep a reference to judge_model for executor_kwargs convenience
        self.judge_model = executor_kwargs.get('model') if executor_kwargs else None

    def _get_formatter(self, dataset):
        if not dataset:
            return lambda x: str(x)
        return get_default_formatter(dataset)

    def _determine_executor_class(self, dataset):
        """Determine whether to use reference-based or reference-free metrics based on dataset and use_prometheus setting."""
        reference_columns = dataset.get_reference_columns()
        has_references = reference_columns is not None and len(reference_columns) > 0
        
        if self.use_prometheus:
            # Import here to avoid circular imports
            from autometrics.metrics.generated.GeneratedPrometheus import (
                GeneratedRefBasedPrometheusMetric,
                GeneratedRefFreePrometheusMetric
            )
            if has_references:
                return GeneratedRefBasedPrometheusMetric
            else:
                return GeneratedRefFreePrometheusMetric
        else:
            # Use DSPy-based LLM Judge metrics with rubrics
            from autometrics.metrics.generated.GeneratedLLMJudgeMetric import (
                GeneratedRefBasedLLMJudgeMetric,
                GeneratedRefFreeLLMJudgeMetric
            )
            if has_references:
                return GeneratedRefBasedLLMJudgeMetric
            else:
                return GeneratedRefFreeLLMJudgeMetric
    
    def _preprocess_dataset(self, dataset, target_measure, formatter: Optional[Callable] = None):
        if not formatter:
            formatter = self._get_formatter(dataset)

        df = dataset.get_dataframe()
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]

        good_examples, bad_examples = get_good_bad_examples(df, target_measure)

        good_examples_formatted = [formatter(row) for row in good_examples.iterrows()]
        bad_examples_formatted = [formatter(row) for row in bad_examples.iterrows()]

        return good_examples_formatted, bad_examples_formatted
    
    def generate(self, dataset, target_measure: Optional[str] = None, n_metrics: int = 5, formatter: Optional[Callable] = None, **kwargs):
        """
        Generate new rubric-based metrics based on the dataset and task description.
        Automatically detects if the dataset has references and uses the appropriate metric class.
        """
        print(f"DEBUG: RubricGenerator.generate called with use_prometheus={self.use_prometheus}")
        print(f"DEBUG: generator_llm type: {type(self.generator_llm)}")
        print(f"DEBUG: judge_model type: {type(self.judge_model)}")
        print(f"DEBUG: executor_kwargs: {self.executor_kwargs}")

        task_description = dataset.get_task_description()

        if not formatter:
            formatter = self._get_formatter(dataset)
        
        # Step-1: Determine the appropriate executor class based on dataset and use_prometheus setting
        if self.executor_class is None:
            dynamic_executor_class = self._determine_executor_class(dataset)
        else:
            dynamic_executor_class = self.executor_class
        
        print(f"DEBUG: Using executor class: {dynamic_executor_class}")
        
        # Step-2: Prepare / cache dataset & formatter ---------------------------------
        good_examples_formatted, bad_examples_formatted = self._preprocess_dataset(dataset, target_measure, formatter)

        # Step-3: Ask the language model to propose axes -----------------------------
        print("DEBUG: Generating axes of variation...")
        try:
            axes = generate_axes_of_variation(
                task_description=task_description,
                good_examples=good_examples_formatted,
                bad_examples=bad_examples_formatted,
                generator_llm=self.generator_llm,
                target_name=target_measure,
                num_axes_to_generate=n_metrics,
            )
            print(f"DEBUG: Generated {len(axes)} axes")
        except Exception as e:
            print(f"ERROR: Failed to generate axes: {e}")
            raise

        axes = axes[:n_metrics] if n_metrics else axes

        # Step-4: Generate rubrics for each axis and wrap in appropriate metric ------
        new_metrics = []

        # Helper function to generate rubric with proper DSPy context
        def generate_rubric_with_context(task_description, good_examples_formatted, bad_examples_formatted, metric_title, metric_description):
            print(f"DEBUG: Generating rubric for '{metric_title}' in thread")
            try:
                with dspy.settings.context(lm=self.generator_llm):
                    rubric_generator = GenerateRubric()
                    result = rubric_generator.forward(
                        task_description,
                        good_examples_formatted,
                        bad_examples_formatted,
                        metric_title,
                        metric_description
                    )
                    print(f"DEBUG: Successfully generated rubric for '{metric_title}'")
                    return result
            except Exception as e:
                print(f"ERROR: Failed to generate rubric for '{metric_title}': {e}")
                raise

        print("DEBUG: Starting rubric generation with ThreadPoolExecutor...")
        with ThreadPoolExecutor() as executor:
            futures = []

            for i, axis in enumerate(axes):
                metric_title = axis.split(":")[0].replace("*", "").strip()
                metric_description = axis
                print(f"DEBUG: Submitting rubric generation task {i+1}/{len(axes)} for '{metric_title}'")

                # Submit the helper function that ensures proper DSPy context
                futures.append(
                    executor.submit(
                        generate_rubric_with_context,
                        task_description,
                        good_examples_formatted,
                        bad_examples_formatted,
                        metric_title,
                        metric_description
                    )
                )

            print(f"DEBUG: Waiting for {len(futures)} rubric generation tasks to complete...")
            for i, future in enumerate(as_completed(futures)):
                try:
                    print(f"DEBUG: Processing completed rubric generation task {i+1}/{len(futures)}")
                    llm_rubric = future.result()

                    rubric = {
                        "criteria": llm_rubric.criteria,
                        "score1_description": llm_rubric.score_descriptions[0],
                        "score2_description": llm_rubric.score_descriptions[1],
                        "score3_description": llm_rubric.score_descriptions[2],
                        "score4_description": llm_rubric.score_descriptions[3],
                        "score5_description": llm_rubric.score_descriptions[4],
                    }

                    # Extract metric name from the criteria - handle edge cases properly
                    criteria_text = llm_rubric.criteria.strip()
                    
                    # Try to extract title from criteria - look for common patterns
                    if ':' in criteria_text:
                        # Standard format: "Title: Description"
                        metric_title_clean = criteria_text.split(':')[0].replace('*', '').strip()
                    elif criteria_text.startswith('**') and '**' in criteria_text[2:]:
                        # Bold format: "**Title** Description"
                        metric_title_clean = criteria_text.split('**')[1].strip()
                    else:
                        # Fallback: take first few words and clean them up
                        words = criteria_text.split()[:4]  # Take first 4 words max
                        metric_title_clean = ' '.join(words).replace('*', '').strip()
                    
                    # Clean up the metric name: remove problematic characters and limit length
                    metric_title_clean = (metric_title_clean
                                         .replace("'", "")      # Remove apostrophes
                                         .replace('"', "")      # Remove quotes
                                         .replace(".", "")      # Remove periods
                                         .replace(",", "")      # Remove commas
                                         .replace("(", "")      # Remove parentheses
                                         .replace(")", "")      # Remove parentheses
                                         .replace("/", " ")     # Replace slashes with spaces
                                         .strip())
                    
                    # If it's still too long, take only the first few meaningful words
                    words = metric_title_clean.split()
                    if len(words) > 4:
                        metric_title_clean = ' '.join(words[:4])
                    
                    # Generate clean metric name with proper suffix
                    if self.use_prometheus:
                        metric_name = f"{metric_title_clean}_Prometheus"
                    else:
                        metric_name = f"{metric_title_clean}_Rubric"
                    
                    print(f"DEBUG: Creating metric '{metric_name}' with executor class {dynamic_executor_class}")

                    # Create the appropriate metric with rubric
                    if self.use_prometheus:
                        # For Prometheus metrics, pass the rubric
                        new_metrics.append(
                            dynamic_executor_class(
                                name=metric_name,
                                description=llm_rubric.criteria,
                                rubric=rubric,
                                task_description=task_description,
                                metric_card_author_model=self.generator_llm,
                                **self.executor_kwargs,
                            )
                        )
                    else:
                        # For DSPy metrics, convert rubric to axis format
                        rubric_text = f"{llm_rubric.criteria}\n\nScoring Guidelines:\n"
                        for j, desc in enumerate(llm_rubric.score_descriptions, 1):
                            rubric_text += f"Score {j}: {desc}\n"
                        
                        new_metrics.append(
                            dynamic_executor_class(
                                name=metric_name,
                                description=llm_rubric.criteria,
                                axis=rubric_text,
                                rubric=rubric,  # Store the rubric for markdown display
                                task_description=task_description,
                                metric_card_author_model=self.generator_llm,
                                **self.executor_kwargs,
                            )
                        )

                except Exception as e:
                    print(f"Error generating rubric: {e}")
                    import traceback
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")

        print(f"DEBUG: Generated {len(new_metrics)} metrics total")
        return new_metrics

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__() 