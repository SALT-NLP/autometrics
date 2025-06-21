from typing import List, Optional
import re
import dspy

# Reusable helper -----------------------------------------------------------

def get_good_bad_examples(df, target_column: str, num_examples: int = 5, flip: bool = False):
    """Return two dataframes: examples with *highest* target values and *lowest* target values.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the examples.
    target_column : str
        Column in *df* that contains the numeric quality signal we want to sort by.
    num_examples : int, optional
        How many good / bad examples to return, by default 5.
    flip : bool, optional
        If *True* the meaning of good/bad is flipped (good becomes lowest values), by default False.
    """
    good_examples = df.sort_values(by=target_column, ascending=False).head(num_examples)
    bad_examples = df.sort_values(by=target_column, ascending=True).head(num_examples)

    if flip:
        return bad_examples, good_examples

    return good_examples, bad_examples

# ---------------------------------------------------------------------------
# DSPy module for generating axes of variation
# ---------------------------------------------------------------------------

class GenerateAxisOfVariationSignature(dspy.Signature):
    """Given a task description, a target metric, and good/bad examples, generate a list of axes of variation which could be used to explain the differences between the good and bad examples.  These axes of variation will be used as measures to evaluate the model's performance, so they should be informative and useful for the model to improve on."""

    task_description: str = dspy.InputField(desc="A description of the overall task the model is trying to solve.")
    target_name: Optional[str] = dspy.InputField(desc="Optional hint of the target metric/column we care about. Could be 'None' or something generic like 'quality' or 'score'.")
    good_examples: List[str] = dspy.InputField(desc="A list of examples with *high* quality according to the target metric.")
    bad_examples: List[str] = dspy.InputField(desc="A list of examples with *low* quality according to the target metric.")
    num_axes_to_generate: int = dspy.InputField(desc="The number of axes of variation to generate.")
    axes_of_variation: List[str] = dspy.OutputField(desc="An ordered list (most-important first) describing possible axes of variation. Please bold the name of the axis of variation (e.g. **Axes Name**), and ALSO include a brief sentence-long explanation of the axis of variation. (e.g. **Axes Name** Brief Explanation).  Please include exactly `num_axes_to_generate` axes of variation in the output.  Avoid special characters since they sometimes mess up the parsing.")


class GenerateAxisOfVariation(dspy.Module):
    """DSPy module wrapping a Chain-of-Thought call to generate axes of variation."""

    def __init__(self):
        super().__init__()
        self.generate_axes = dspy.ChainOfThought(GenerateAxisOfVariationSignature)

    def forward(self, task_description: str, good_examples: List[str], bad_examples: List[str], target_name: Optional[str] = None, num_axes_to_generate: int = 5):
        if not target_name:
            target_name = "None"
        response = self.generate_axes(
            task_description=task_description,
            target_name=target_name,
            good_examples=good_examples,
            bad_examples=bad_examples,
            num_axes_to_generate=num_axes_to_generate,
        ).axes_of_variation

        # Clean up each axis string in the list
        axes = [axis.strip() for axis in response]
        # Remove any empty strings
        axes = [axis for axis in axes if axis]
        # If first item starts with a number, remove it
        if axes and axes[0].startswith("1."):
            axes[0] = axes[0][2:].strip()

        return dspy.Prediction(axes_of_variation=axes)


# ---------------------------------------------------------------------------
# Convenience wrapper that downstream code can call directly
# ---------------------------------------------------------------------------

def generate_axes_of_variation(
    task_description: str,
    good_examples: List[str],
    bad_examples: List[str],
    generator_llm: Optional[dspy.LM] = None,
    target_name: Optional[str] = None,
    num_axes_to_generate: int = 5,
) -> List[str]:
    """Generate a ranked list of textual axes of variation.

    This thin wrapper takes care of temporarily swapping in a custom LLM if
    provided so that call-sites do not need to handle *dspy.settings.context*
    themselves.
    """
    if generator_llm is not None:
        with dspy.settings.context(lm=generator_llm):
            axes_pred = GenerateAxisOfVariation()(task_description, good_examples, bad_examples, target_name, num_axes_to_generate)
    else:
        axes_pred = GenerateAxisOfVariation()(task_description, good_examples, bad_examples, target_name, num_axes_to_generate)

    return axes_pred.axes_of_variation