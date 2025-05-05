import torch
from typing import List, Union, Tuple
from lens import download_model, LENS_SALSA as _LENS_SALSA_Model
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric

class LENS_SALSA(ReferenceFreeMetric):
    """---
# Metric Card for LENS_SALSA

LENS_SALSA is a reference-free metric designed to evaluate the overall quality of text simplification outputs. It leverages the SALSA (Simplification Analysis via Lexical and Structural Alignment) framework introduced by Heineman et al. (2023), which analyzes edits at the word level to assess whether a simplification succeeds or fails. The LENS_SALSA model uses these insights to produce a scalar simplification quality score based on input-output pairs, with no need for reference texts. This makes it particularly useful in settings where reference simplifications are unavailable or unreliable.

## Metric Details

### Metric Description

LENS_SALSA is a neural, reference-free metric for evaluating sentence-level simplification quality. It builds on the SALSA framework, which identifies and categorizes the types of edits performed when transforming a complex sentence into a simplified one. SALSA aligns input and output tokens using an alignment algorithm and labels each word-level edit with one of several tags—e.g., deletion, substitution, or addition—and further classifies the edit as a *success* or *failure* based on its impact on fluency, adequacy, and simplicity. These labels are derived from a manually annotated corpus.

The LENS_SALSA model is trained using these edit-level annotations. It learns to aggregate the local edit patterns into a global simplification quality score using a supervised regression objective. Crucially, this scoring process does not require reference simplifications at inference time, making LENS_SALSA a practical tool in real-world simplification pipelines.

- **Metric Type:** Reference-Free
- **Range:** Unbounded (empirically observed in [0, 100] scale)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

The LENS_SALSA score is generated via a neural model trained on edit-labeled simplification data. The core of the system is the SALSA framework, which performs alignment and tagging of edits between a complex input sentence $x$ and a simplified candidate $\hat{x}$.

Let:
- $A = \text{Align}(x, \hat{x})$ be the alignment between input and output tokens,
- $E(x, \hat{x}, A)$ be the set of word-level edits extracted from the alignment,
- $T(e)$ be the success/failure tag for an edit $e$ (as determined by the SALSA labeling scheme),
- $f(E)$ be the feature vector summarizing the counts and types of edit tags in $E$.

Then, LENS_SALSA computes the final score using a regression model (MLP):

$$
\text{LENS\_SALSA}(x, \hat{x}) = \text{MLP}(f(E(x, \hat{x}, A)))
$$

The model is trained using human-annotated quality scores from simplification corpora.

### Inputs and Outputs

- **Inputs:**  
  - Input text (original complex sentence)  
  - Output text (candidate simplified sentence)  
  - (Optional) Reference text(s), used only during training or secondary analysis  
  
- **Outputs:**  
  - Scalar simplification score (float, typically between 0 and 100)  
  - (Optional) Word-level edit tags indicating success/failure for interpretability

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Text Simplification

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating text simplification outputs where reference simplifications are unavailable, unreliable, or highly variable.  
  Particularly effective for sentence-level simplification tasks focused on fluency, adequacy, and simplicity.

- **Not Recommended For:**  
  - Tasks outside of simplification, such as summarization or paraphrasing  
  - Long-form or document-level generation  
  - Settings where simplification quality depends heavily on context beyond a single sentence

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Hugging Face: davidheineman/lens-salsa](https://huggingface.co/davidheineman/lens-salsa)  
  - `autometrics` (custom wrapper around LENS_SALSA model for reference-free evaluation)

### Computational Complexity

- **Efficiency:**  
  Relatively efficient inference via PyTorch-based model. Overhead comes from computing alignment-based features and scoring.

- **Scalability:**  
  Scales to batched input using the `calculate_batched` method. Memory usage depends on model size and batch configuration.

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  Designed specifically for simplification; using it for other tasks may result in misleading evaluations.

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- **SARI:** Reference-based simplification metric often used alongside LENS_SALSA.  
- **BERTScore (adapted to simplification):** Captures semantic similarity between input and output.  
- **LENS Framework:** Edit-level analysis from which LENS_SALSA is derived.

## Further Reading

- **Papers:**  
  - [Dancing Between Success and Failure: Edit-level Simplification Evaluation using SALSA](https://aclanthology.org/2023.emnlp-main.211/)  

- **Blogs/Tutorials:**  
  Needs more information.

## Citation

```
@inproceedings{heineman-etal-2023-dancing,
    title = "Dancing Between Success and Failure: Edit-level Simplification Evaluation using {SALSA}",
    author = "Heineman, David  and
      Dou, Yao  and
      Maddela, Mounica  and
      Xu, Wei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.211/",
    doi = "10.18653/v1/2023.emnlp-main.211",
    pages = "3466--3495"
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    def __init__(
        self,
        name: str = "LENS_SALSA",
        description: str = "LENS-SALSA reference-free simplification metric (overall score).",
        model_id: str = "davidheineman/lens-salsa",
        batch_size: int = 16,
        devices: List[int] = None,
        persistent: bool = True
    ):
        super().__init__(name, description)
        self.model_id = model_id
        self.batch_size = batch_size
        self.devices = devices
        self.persistent = persistent
        self.model = None
        if self.persistent:
            self._load_model()

    def _load_model(self):
        """Download SALSA checkpoint and load the LENS_SALSA model."""
        if self.model is None:
            ckpt_path = download_model(self.model_id)
            self.model = _LENS_SALSA_Model(ckpt_path)

    def _unload_model(self):
        """Unload SALSA model to free resources."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def calculate(self,
                  input: str,
                  output: str,
                  references: Union[List[str], None] = None,
                  **kwargs) -> float:
        """
        Compute overall SALSA score for a single example.
        Returns a float score.
        """
        if self.model is None:
            self._load_model()
        # prepare references argument
        refs_arg = references if references else None
        scores, _ = self.model.score(
            [input], [output], [refs_arg] if refs_arg is not None else None,
            batch_size=self.batch_size,
            devices=self.devices
        )
        result = float(scores[0])
        if not self.persistent:
            self._unload_model()
        return result

    def calculate_batched(self,
                          inputs: List[str],
                          outputs: List[str],
                          references: Union[List[List[str]], None] = None,
                          **kwargs) -> List[float]:
        """
        Compute overall SALSA scores for a batch of examples.
        Returns list of float scores.
        """
        if self.model is None:
            self._load_model()
        refs_arg = references if references is not None else None
        scores, _ = self.model.score(
            inputs,
            outputs,
            refs_arg,
            batch_size=self.batch_size,
            devices=self.devices
        )
        results = [float(s) for s in scores]
        if not self.persistent:
            self._unload_model()
        return results 