import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import re

import dspy

from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code
from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric

__all__ = ["GeneratedRefFreeLLMJudgeMetric"]


class _LLMJudgeSignatureRefFree(dspy.Signature):
    """Given the task description, and an evaluation axis, rate the output text along the axis. It may be helpful to use the input text as context."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
    input_text: str = dspy.InputField(desc="The text that was input to the model to produce the output text.")
    output_text: str = dspy.InputField(desc="The text that was produced by the model (this is the text that we want to rate).")
    score: int = dspy.OutputField(desc="A numerical score 1-5.")


class GeneratedRefFreeLLMJudgeMetric(GeneratedRefFreeMetric):
    """Reference-free metric that leverages an LLM to judge outputs along a textual axis.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    axis            The textual axis/rubric used for judgement (e.g. "*Clarity*: How clear is …")
    model           A *dspy.LM* instance (or wrapper exposing .model attribute) used for judging
    task_description Optional task context passed to the judge
    metric_card_author_model  LLM used to generate the metric-card (defaults to *model*)
    """

    DEFAULT_MAX_WORKERS = 32

    def __init__(
        self,
        name: str,
        description: str,
        axis: str,
        model: dspy.LM,
        task_description: Optional[str] = None,
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        **kwargs,
    ):
        self.axis = axis
        self.task_description = task_description or "None"
        self.model = model
        self.max_workers = max_workers

        if metric_card_author_model is None:
            metric_card_author_model = model if isinstance(model, dspy.LM) else None

        if metric_card == "provided":
            self.metric_card = self.__doc__
            metric_card = self.metric_card

        super().__init__(
            name,
            description,
            metric_card=metric_card,
            metric_card_author_model=metric_card_author_model,
            axis=axis,
            model_str=str(getattr(model, "model", model)),
            task_description=self.task_description,
            **kwargs,
        )

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model")

        # Prepare the DSPy module once
        self._judge_module = dspy.ChainOfThought(_LLMJudgeSignatureRefFree)

    # ------------------------------------------------------------------
    # Metric implementation
    # ------------------------------------------------------------------

    def _call_llm(self, input_text: str, output_text: str) -> float:
        with dspy.settings.context(lm=self.model):
            score = self._judge_module(
                task_description=self.task_description,
                axis=self.axis,
                input_text=input_text,
                output_text=output_text,
            ).score

        return float(score)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._call_llm(input, output)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        results: List[float] = [0.0] * len(outputs)

        # Fail-fast if workers=1
        if self.max_workers == 1:
            return [self._call_llm(i, o) for i, o in zip(inputs, outputs)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._call_llm, i, o): idx for idx, (i, o) in enumerate(zip(inputs, outputs))}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        code = f"""# Auto-generated metric file for {self.name}
import dspy
import os
from autometrics.metrics.generated.GeneratedRefFreeLLMJudgeMetric import GeneratedRefFreeLLMJudgeMetric

DEFAULT_MODEL = {generate_llm_constructor_code(self.model)}

class {self.name.replace(" ", "_").replace("-", "_")}_LLMJudge(GeneratedRefFreeLLMJudgeMetric):
    \"\"\"{self.metric_card if include_metric_card else ""}\"\"\"
    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            axis={json.dumps(self.axis)},
            model=model,
            task_description={json.dumps(self.task_description)},
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_workers={self.max_workers},
        )

    def __repr__(self):
        return "{self.name.replace(" ", "_").replace("-", "_")}_LLMJudge(model={generate_llm_constructor_code(self.model)})"

"""
        return code
    
    def save(self, path: str):
        dump_kwargs = {
            "name": self.name,
            "description": self.description,
            "axis": self.axis,
            "model": generate_llm_constructor_code(self.model),
            "task_description": self.task_description,
            "metric_card": self.metric_card,
            "max_workers": self.max_workers,
        }
        with open(path, "w") as f:
            json.dump(dump_kwargs, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            dump_kwargs = json.load(f)
        return cls(**dump_kwargs)
    
    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based judges.

        Parameters
        ----------
        reference_based : bool
            If True, emit the reference-based variant; otherwise emit the
            reference-free variant.
        """
        kind = "reference-based" if reference_based else "reference-free"
        ref_flag = "Yes" if reference_based else "No"
        input_req = "Yes (plus reference)" if reference_based else "Yes"

        # --- Header & description ----------------------------------------
        lines = [
            f"**{self.name}** is a **{kind}** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.",
            f"In this case the axis is `{self.axis}`.",
            "",
            "The prompt supplies:",
            "",
            "1. **Task description** *d*",
            f"2. **Axis rubric** `{self.axis}`",
            "3. **Input text** *x*",
        ]
        if reference_based:
            lines.append("4. **Reference text** *r*")
            lines.append("5. **Output text** *y*")
        else:
            lines.append("4. **Output text** *y*")

        # --- Scoring sentence --------------------------------------------
        lines.extend(
            [
                "",
                r"Greedy decoding (temperature = 0) yields an integer score "
                r"$\hat{s}\!\in\!\{1,2,3,4,5\}$; higher = better adherence "
                "to the axis.",
                "",
                "- **Metric Type:** LLM as a Judge",
                "- **Range:** 1-5 (1 = worst, 5 = best)",
                "- **Higher is Better?:** Yes",
                f"- **Reference-Based?:** {ref_flag}",
                f"- **Input-Required?:** {input_req}",
                "",
                "### Formal Definition",
                "",
                r"Let $f _{\\theta}$ be the LLM and",
            ]
        )

        if reference_based:
            lines.append(
                r"$\pi _{\text{RB}}(d,\{axis\},x,r,y)$ construct the textual "
                "prompt."
            )
        else:
            lines.append(
                r"$\pi _{\text{RF}}(d,\{axis\},x,y)$ construct the textual "
                "prompt."
            )

        lines.extend(
            [
                "",
                "$$",
                r"\hat{s} \;=\; \operatorname*{arg\,max}\limits_{s \in "
                r"\{1,\dots,5\}} "
                r"f _{\theta}\!\bigl("
                r"s \,\bigl|\, "
                + (r"\pi _{\text{RB}}(d,\{axis\},x,r,y)"
                   if reference_based
                   else r"\pi _{\text{RF}}(d,\{axis\},x,y)")
                + r"\bigr)",
                "$$",
                "",
                r"The metric value is "
                + (
                    r"$\operatorname{LJ}^{\text{RB}}_{\{axis\}}"
                    r"(d,x,r,y)=\hat{s}$."
                    if reference_based
                    else r"$\operatorname{LJ}^{\text{RF}}_{\{axis\}}"
                    r"(d,x,y)=\hat{s}$."
                ),
                "",
                "### Inputs and Outputs",
                "- **Inputs:**",
                "  - **Task description** *d*",
                f"  - **Axis rubric** `{self.axis}`",
                "  - **Input text** *x*",
            ]
        )
        if reference_based:
            lines.append("  - **Reference text** *r*")
        lines.append("  - **Output text** *y*")
        lines.extend(
            [
                "- **Outputs:**",
                "  - Scalar score "
                r"$\hat{s} \in \{1,2,3,4,5\}$",
            ]
        )

        return "\n".join(lines)
    
    def generate_metric_details_ref_free(self) -> str:
        """Metric-details section for the **reference-free** variant."""
        return self._metric_details_template(reference_based=False)

    def generate_metric_details_ref_based(self) -> str:
        """Metric-details section for the **reference-based** variant."""
        return self._metric_details_template(reference_based=True)

    def generate_intended_use(self):
        class IntendedUseSignature(dspy.Signature):
            """Given the task description, and an evaluation axis, consider an LLM Judge that is evaluating the text along this axis.  Your task is to generate the domain, a list of tasks, and a set of circumstances where the LLM Judge is best suited to be used as well as where it should not be used.  Note that you are generating the intended use for the LLM Judge, not the intended use for the task!!"""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used as the LLM Judge.")
            domain: str = dspy.OutputField(desc="The domain of the task.  Some examples are: Text Generation, Code Generation, Discourse, etc.")
            tasks: List[str] = dspy.OutputField(desc="A list of tasks that the LLM Judge is best suited to be used for.  Some examples are: Travel Planning, Code Review, Machine Translation, Dialogue Response Generation, etc.")
            best_suited_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where the LLM Judge is best suited to be used.  This can describe properties of the task, data, environment, etc. that would lead to successful evaluation when using the LLM Judge on this axis. (approximately one sentence each)")
            not_recommended_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where the LLM Judge is not recommended to be used.  This can describe properties of the task, data, environment, etc. that would lead to unsuccessful evaluation when using the LLM Judge on this axis. (approximately one sentence each)")

        with dspy.settings.context(lm=self.model):
            outputs = dspy.ChainOfThought(IntendedUseSignature)(
                task_description=self.task_description,
                axis=self.axis,
                model_name=str(getattr(self.model, "model", self.model)),
            )
        
        return f"""- **Domain:** {outputs.domain}
- **Tasks:** {"\n  - " + "\n  - ".join(outputs.tasks)}
- **Best Suited For:** {"\n  - " + "\n  - ".join(outputs.best_suited_for_circumstances)}
- **Not Recommended For:** {"\n  - " + "\n  - ".join(outputs.not_recommended_for_circumstances)}"""

    def generate_metric_implementation(self):
        return """### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics LLM as a Judge](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedRefFreeLLMJudgeMetric.py)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair.
  - AutoMetrics does parallel calls on batched inputs.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.  Additional consideration would include whether or not the LLM is a reasoning model."""

    def generate_known_limitations(self):
        class KnownLimitationsSignature(dspy.Signature):
            """Given the task description, and an evaluation axis, consider an LLM Judge that is evaluating the text along this axis.  Your task is to generate a list of biases, task misalignment risks, and failure cases that could be present in this evaluation.  Especially consider the axis and how it is aligned or misaligned with BOTH this task and other tasks that the LLM Judge may be used for.  Note that you are generating the known limitations for the LLM Judge, not the known limitations for the task!!"""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used as the LLM Judge.")
            biases: List[str] = dspy.OutputField(desc="A list of biases the could be present in this evaluation (approximately one sentence each).")
            task_misalignment_risks: List[str] = dspy.OutputField(desc="A list of ways in which this evaluation could be misaligned with the task (approximately one sentence each).")
            failure_cases: List[str] = dspy.OutputField(desc="A list of failure cases that could occur in this evaluation (approximately one sentence each).")

        with dspy.settings.context(lm=self.model):
            outputs = dspy.ChainOfThought(KnownLimitationsSignature)(
                task_description=self.task_description,
                axis=self.axis,
                model_name=str(getattr(self.model, "model", self.model)),
            )
        
        return f"""- **Biases:** {"\n  - " + "\n  - ".join(outputs.biases)}
- **Task Misalignment Risks:** {"\n  - " + "\n  - ".join(outputs.task_misalignment_risks)}
- **Failure Cases:** {"\n  - " + "\n  - ".join(outputs.failure_cases)}"""

    def generature_further_reading(self):
        return generate_further_reading(self) + "\n  - [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://openreview.net/pdf?id=uccHPGDlao)"

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class LLMJudgeMetricCardBuilder(MetricCardBuilder):
            def metric_details(self) -> str:
                return self.metric.generate_metric_details_ref_free()
            
            def intended_use(self) -> str:
                return self.metric.generate_intended_use()
            
            def metric_implementation(self) -> str:
                return self.metric.generate_metric_implementation()
            
            def known_limitations(self) -> str:
                return self.metric.generate_known_limitations()
            
            def further_reading(self) -> str:
                return self.metric.generature_further_reading()

        with dspy.settings.context(lm=author_model or self.model):
            builder = LLMJudgeMetricCardBuilder(self)
            return builder.build()