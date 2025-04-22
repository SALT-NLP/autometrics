# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class BARTScorer:
    def __init__(
        self,
        device: str = None,
        max_length: int = None,
        checkpoint: str = "facebook/bart-large-cnn"
    ):
        # Set up tokenizer and model
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        # pick device
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # never exceed the model’s own max_position_embeddings
        self.max_length = (
            max_length
            if max_length is not None
            else self.tokenizer.model_max_length
        )

        # loss & log‐softmax
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.tokenizer.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=-1)

    def load(self, path: str = None):
        """Load model weights (e.g. after paraphrase fine‑tuning)."""
        if path is None:
            path = "models/bart.pth"
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

    def score(self, srcs: List[str], tgts: List[str], batch_size: int = 4):
        """Score a batch of (source, target) pairs, returning log‑probability scores."""
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_batch = srcs[i : i + batch_size]
            tgt_batch = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    # tokenize
                    enc_src = self.tokenizer(
                        src_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    enc_tgt = self.tokenizer(
                        tgt_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )

                    src_ids = enc_src.input_ids.to(self.device)
                    src_mask = enc_src.attention_mask.to(self.device)

                    tgt_ids = enc_tgt.input_ids.to(self.device)
                    tgt_mask = enc_tgt.attention_mask.to(self.device)
                    tgt_lens = tgt_mask.sum(dim=1).to(self.device)

                    # now call BART with explicit decoder mask
                    output = self.model(
                        input_ids=src_ids,
                        attention_mask=src_mask,
                        decoder_attention_mask=tgt_mask,
                        labels=tgt_ids,
                    )

                    # flatten logits and compute per‑token NLL
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    logp = self.lsm(logits)
                    losses = self.loss_fct(logp, tgt_ids.view(-1))
                    losses = losses.view(tgt_ids.size(0), -1)
                    # average over true length
                    curr_scores = [-(l.sum().item() / lgt) for l, lgt in zip(losses, tgt_lens)]
                    score_list.extend(curr_scores)

            except RuntimeError as e:
                traceback.print_exc()
                raise RuntimeError(
                    f"Error scoring batch starting at index {i}: {e}"
                ) from e

        return score_list

    def multi_ref_score(
        self,
        srcs: List[str],
        tgts: List[List[str]],
        agg: str = "mean",
        batch_size: int = 4,
    ):
        # ensure uniform number of references
        ref_counts = [len(r) for r in tgts]
        if len(set(ref_counts)) > 1:
            raise ValueError("All examples must have the same number of references.")
        num_refs = ref_counts[0]

        # score each “column” of references
        matrix = []
        for idx in range(num_refs):
            col = [refs[idx] for refs in tgts]
            matrix.append(self.score(srcs, col, batch_size))
        arr = np.array(matrix)  # shape (num_refs, batch_size)
        if agg == "mean":
            return list(arr.mean(axis=0))
        elif agg == "max":
            return list(arr.max(axis=0))
        else:
            raise ValueError(f"Unknown agg mode: {agg}")

    def test(self, batch_size: int = 3):
        srcs = [
            "This is a very good idea. Although simple, but very insightful.",
            "Can I take a look?",
            "Do not trust him, he is a liar.",
        ]
        tgts = [
            "That's stupid.",
            "What's the problem?",
            "He is trustworthy.",
        ]
        print(self.score(srcs, tgts, batch_size))


class BARTScore(ReferenceBasedMetric):
    """---
# Metric Card for BARTScore

BARTScore is a reference-based evaluation metric for text generation that formulates evaluation as a text generation task. It leverages the pre-trained BART model to compute the conditional likelihood of one text given another, enabling flexible evaluation of different aspects such as informativeness, fluency, factuality, and coherence. BARTScore outperforms existing metrics across multiple tasks and evaluation settings.

## Metric Details

### Metric Description

BARTScore conceptualizes evaluation as a text generation problem, assessing how likely a hypothesis (generated text) is given a reference text, source text, or both. This probability is computed using the log-likelihood of the hypothesis under a pre-trained BART model. Different evaluation perspectives can be achieved by modifying the generation direction:

- **Faithfulness ($s \to h$)**: Measures how well the generated text aligns with the source text.
- **Precision ($r \to h$)**: Evaluates the likelihood of generating the hypothesis given the reference text.
- **Recall ($h \to r$)**: Assesses how easily the reference could be generated from the hypothesis.
- **F-score ($r \leftrightarrow h$)**: Computes an average of Precision and Recall.

Fine-tuning on downstream tasks (e.g., summarization, paraphrasing) and prompt engineering further enhance BARTScore’s adaptability to different domains.

- **Metric Type:** Semantic Similarity  
- **Range:** $(-\infty, 0]$ (log-probabilities, higher is better)  
- **Higher is Better?:** Yes  
- **Reference-Based?:** Yes  
- **Input-Required?:** Yes  

### Formal Definition

BARTScore is computed as:

$$
BARTScore = \sum _{t=1}^{m} \omega _{t} \log p ( y _{t} \mid y _{\text{<}t}, x, \theta )
$$

where:

- $p(y _{t} \mid y _{<t}, x, \theta)$ is the probability of the $t$-th token in the hypothesis $y$ given the preceding tokens and the source/reference text $x$ under the BART model parameters $\theta$.
- $\omega _{t}$ is an optional weighting factor (default: uniform).

The choice of $x$ and $y$ varies depending on the evaluation perspective (e.g., source-to-hypothesis for faithfulness, reference-to-hypothesis for precision).

### Inputs and Outputs

- **Inputs:**  
  - Source text (optional, for faithfulness evaluation)
  - Generated text (hypothesis)
  - Reference text(s) (for precision, recall, and F-score)

- **Outputs:**  
  - Scalar log-likelihood score (higher indicates better alignment)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation  
- **Tasks:**  
  - Machine Translation  
  - Summarization  
  - Paraphrasing  
  - Data-to-Text Generation  
  - Dialogue Generation  

### Applicability and Limitations

- **Best Suited For:**  
  - Tasks where reference-based evaluation is appropriate (e.g., machine translation, summarization).
  - Evaluating generated text from multiple perspectives (e.g., factuality, coherence, fluency).
  - Cases where fine-tuning and prompt-based customization are beneficial.

- **Not Recommended For:**  
  - Fully reference-free evaluation tasks.
  - Open-ended generation tasks where diversity matters more than similarity to references.
  - Evaluating highly extractive summaries, where performance may degrade.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [BARTScore GitHub Repository](https://github.com/neulab/BARTScore)  
  - Available in Hugging Face `evaluate` module  

### Computational Complexity

- **Efficiency:**  
  - Requires forward passes through BART, making it more computationally expensive than n-gram-based metrics.  
  - Can be optimized using batch processing.  

- **Scalability:**  
  - Suitable for large-scale evaluations but requires GPU acceleration for efficiency.  
  - Performance depends on the pre-trained model size and dataset length.

## Known Limitations

- **Biases:**  
  - BARTScore tends to favor abstractive over extractive summaries.  
  - May be sensitive to the domain of the pre-trained BART model used.  

- **Task Misalignment Risks:**  
  - May not fully capture factual correctness despite faithfulness scoring.  
  - Sensitive to tokenization and domain shift effects.  

- **Failure Cases:**  
  - Performance degrades when evaluating extractive summarization models.  
  - Prompt engineering impacts results significantly, requiring careful selection.  

## Related Metrics

- **ROUGE:** Measures lexical overlap, whereas BARTScore captures semantic similarity.  
- **BERTScore:** Also embeds text using pre-trained models but computes cosine similarity instead of generation probabilities.  
- **BLEU:** Focuses on n-gram precision, lacking semantic alignment capabilities.  

## Further Reading

- **Papers:**  
  - [BARTScore: Evaluating Generated Text as Text Generation (Yuan et al., 2021)](https://arxiv.org/abs/2106.11520)  

- **Blogs/Tutorials:**  
  - [BARTScore GitHub Documentation](https://github.com/neulab/BARTScore)  

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  """
    def __init__(
        self, batch_size: int = 4, model: str = "facebook/bart-large-cnn"
    ):
        super().__init__(
            name=f"BARTScore_{model.split('/')[-1]}",
            description=(
                "BARTScore is a reference-based metric for evaluating text quality "
                "using a pre-trained BART model to compute likelihoods."
            ),
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bart_scorer = BARTScorer(
            device=str(self.device), max_length=None, checkpoint=model
        )
        self.batch_size = batch_size

    def calculate(self, input: str, output: str, references=None, **kwargs):
        refs = references or []
        if len(refs) > 1:
            scores = self.bart_scorer.multi_ref_score(
                [output], [refs], agg="max", batch_size=self.batch_size
            )
        else:
            # single-ref case
            scores = self.bart_scorer.score(
                [output], [refs[0] if refs else ""], batch_size=self.batch_size
            )
        return scores[0]

    def calculate_batched(
        self, inputs: List[str], outputs: List[str], references=None, **kwargs
    ):
        refs = references or [[] for _ in inputs]
        # group by num refs to handle variable ref counts
        groups = {}
        for i, r in enumerate(refs):
            groups.setdefault(len(r), []).append(i)

        all_scores = [0] * len(outputs)
        for ref_count, idxs in groups.items():
            outs = [outputs[i] for i in idxs]
            rfs = [refs[i] for i in idxs]
            if ref_count > 1:
                sc = self.bart_scorer.multi_ref_score(
                    outs, rfs, agg="max", batch_size=self.batch_size
                )
            else:
                # single‐ref or no-ref
                single_refs = [r[0] if r else "" for r in rfs]
                sc = self.bart_scorer.score(
                    outs, single_refs, batch_size=self.batch_size
                )
            for idx, score in zip(idxs, sc):
                all_scores[idx] = score

        return all_scores


if __name__ == "__main__":
    # Example usage
    metric = BARTScore()

    # single example
    inp = (
        "Peter and Elizabeth took a taxi to attend the night party in the city. "
        "While in the party, Elizabeth collapsed and was rushed to the hospital."
    )
    out = "Peter and Elizabeth attend party city. Elizabeth rushed hospital."
    refs = ["Elizabeth was hospitalized after attending a party with Peter."]
    score = metric.calculate(inp, out, references=refs)
    print("BARTScore:", score)

    # batched examples
    inputs = [
        inp,
        "The cat sat on the mat.",
    ]
    outputs = [
        out,
        "The cat is on the mat.",
    ]
    references = [
        ["Elizabeth was hospitalized after attending a party with Peter."],
        ["The cat sat on the mat.", "The cat is on the mat.", "The cat is on the rug."],
    ]
    batch_scores = metric.calculate_batched(inputs, outputs, references=references)
    print("BARTScore batch scores:", batch_scores)