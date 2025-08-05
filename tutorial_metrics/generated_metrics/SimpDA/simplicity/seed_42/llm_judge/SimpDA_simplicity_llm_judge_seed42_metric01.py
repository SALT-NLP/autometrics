# Auto-generated metric file for Clarity of Language_gpt-4o-mini
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedRefBasedLLMJudgeMetric
from typing import ClassVar

DEFAULT_MODEL = dspy.LM(model='openai/gpt-4o-mini', temperature=0.0, max_tokens=1000, api_key=os.getenv("OPENAI_API_KEY"))

class Clarity_of_Language_gpt_4o_mini_LLMJudge(GeneratedRefBasedLLMJudgeMetric):
    """---
# Metric Card for Clarity of Language_gpt-4o-mini

**Clarity of Language** The use of straightforward and easily understandable words and phrases enhances the simplicity of the sentence.

## Metric Details

**Clarity of Language_gpt-4o-mini** is a **reference-based** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.
In this case the axis is `**Clarity of Language** The use of straightforward and easily understandable words and phrases enhances the simplicity of the sentence.`.

The prompt supplies:

1. **Task description** *d*
2. **Axis rubric** `**Clarity of Language** The use of straightforward and easily understandable words and phrases enhances the simplicity of the sentence.`
3. **Input text** *x*
4. **Reference text** *r*
5. **Output text** *y*

Greedy decoding (temperature = 0) yields an integer score $\hat{s}\!\in\!\{1,2,3,4,5\}$; higher = better adherence to the axis.

- **Metric Type:** LLM as a Judge
- **Range:** 1-5 (1 = worst, 5 = best)
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes (plus reference)

### Formal Definition

Let $f _{\\theta}$ be the LLM and
$\pi _{\text{RB}}(d,\{axis\},x,r,y)$ construct the textual prompt.

$$
\hat{s} \;=\; \operatorname*{arg\,max}\limits_{s \in \{1,\dots,5\}} f _{\theta}\!\bigl(s \,\bigl|\, \pi _{\text{RB}}(d,\{axis\},x,r,y)\bigr)
$$

The metric value is $\operatorname{LJ}^{\text{RB}}_{\{axis\}}(d,x,r,y)=\hat{s}$.

### Inputs and Outputs
- **Inputs:**
  - **Task description** *d*
  - **Axis rubric** `**Clarity of Language** The use of straightforward and easily understandable words and phrases enhances the simplicity of the sentence.`
  - **Input text** *x*
  - **Reference text** *r*
  - **Output text** *y*
- **Outputs:**
  - Scalar score $\hat{s} \in \{1,2,3,4,5\}$

## Intended Use

- **Domain:** Text Simplification
- **Tasks:** 
  - Sentence Simplification
  - Text Summarization
  - Educational Material Creation
  - Content Editing
  - Accessibility Improvement
- **Best Suited For:** 
  - The original sentences are complex and contain jargon that may confuse readers.
  - The target audience includes individuals with varying levels of language proficiency.
  - The task requires maintaining the original meaning while enhancing readability.
  - Feedback is needed on multiple iterations of simplifications to refine clarity.
- **Not Recommended For:** 
  - The original text is already simple and clear, requiring no further simplification.
  - The audience is highly specialized and requires technical language for understanding.
  - The task involves creative writing where stylistic elements are prioritized over clarity.
  - There is a strict word limit that may hinder the simplification process.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics LLM as a Judge (reference-based)](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedLLMJudgeMetric.py)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair.
  - AutoMetrics does parallel calls on batched inputs.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.  Additional consideration would include whether or not the LLM is a reasoning model.

## Known Limitations

- **Biases:** 
  - The model may favor certain dialects or forms of language that it has been trained on, leading to biased evaluations of clarity.
  - There may be an inherent bias towards simpler vocabulary that does not account for the audience's familiarity with specific terms.
  - The model might exhibit bias towards more common phrases, potentially overlooking valid but less conventional simplifications.
- **Task Misalignment Risks:** 
  - The evaluation may prioritize brevity over clarity, leading to overly simplified sentences that lose essential meaning.
  - The model might misinterpret the target audience's comprehension level, resulting in evaluations that do not align with the intended simplification goals.
  - There is a risk that the model's understanding of clarity does not align with the nuances of the original sentence, leading to misjudged simplifications.
- **Failure Cases:** 
  - The model could incorrectly assess a sentence as clear when it still contains complex ideas that are not adequately simplified.
  - The evaluation might result in a sentence that is too simplistic, stripping away necessary context or detail.
  - The model may fail to recognize idiomatic expressions that require specific cultural knowledge, leading to inappropriate simplifications.

## Related Metrics

- **Related Metrics:**
  - **LevenshteinDistance:** Levenshtein Distance measures the minimum number of single-character edits—insertions, deletions, or substitutions—required to transform one sequence into another.
  - **BLEU:** BLEU (Bilingual Evaluation Understudy) is a widely used metric for evaluating the quality of text generated in tasks like machine translation and summarization.
  - **CIDEr:** CIDEr (Consensus-based Image Description Evaluation) measures the similarity between a candidate image caption and a set of human-generated reference captions.

## Further Reading

- **Papers:**
  - [Autometrics](https://github.com/XenonMolecule/autometrics)
  - [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://openreview.net/pdf?id=uccHPGDlao)

## Citation

```
@software{Ryan_Autometrics_2025,
    author = {Ryan, Michael J. and Zhang, Yanzhe and Salunkhe, Amol and Chu, Yi and Rahman, Emily and Xu, Di and Yang, Diyi},
    license = {MIT},
    title = {{Autometrics}},
    url = {https://github.com/XenonMolecule/autometrics},
    version = {1.0.0},
    year = {2025}
}
```

## Metric Card Authors

- **Authors:** This metric card was automatically generated by gpt-4o-mini.
- **Acknowledgement of AI Assistance:** This metric card was entirely automatically generated by gpt-4o-mini using the Autometrics library. No human intervention was involved. User discretion is advised.
- **Contact:** For questions about the autometrics library, please contact [Michael J Ryan](mailto:mryan0@stanford.edu)."""

    description: ClassVar[str] = "**Clarity of Language** The use of straightforward and easily understandable words and phrases enhances the simplicity of the sentence."

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name="Clarity of Language_gpt-4o-mini",
            description="**Clarity of Language** The use of straightforward and easily understandable words and phrases enhances the simplicity of the sentence.",
            axis="**Clarity of Language** The use of straightforward and easily understandable words and phrases enhances the simplicity of the sentence.",
            model=model,
            task_description="Given a complicated original sentence, simplify it in a way such that a broader audience could easily understand it.",
            metric_card="provided",
            max_workers=32,
        )

    def __repr__(self):
        return f"Clarity_of_Language_gpt_4o_mini_LLMJudge(model=dspy.LM(model='openai/gpt-4o-mini', temperature=0.0, max_tokens=1000, api_key=os.getenv(\"OPENAI_API_KEY\")))"

