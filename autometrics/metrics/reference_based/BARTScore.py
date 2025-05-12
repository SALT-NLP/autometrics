# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizerFast, BartForConditionalGeneration
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
        # Use the fast tokenizer to get correct model_max_length
        self.tokenizer = BartTokenizerFast.from_pretrained(checkpoint, use_fast=True)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        # Pick device
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Cap max_length to the model's own limit
        limit = self.tokenizer.model_max_length
        if limit is None or limit > 1024:
            # most BART checkpoints report 1024
            limit = 1024
        self.max_length = limit if max_length is None else min(max_length, limit)

        # Loss & log-softmax
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.tokenizer.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=-1)

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def _truncate_text(self, text: str) -> str:
        """Return `text` shortened so that it tokenizes to <= max_length."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_length:
            return text
        ids = ids[: self.max_length]
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # -------------------------------------------------
    # main scoring routine
    # -------------------------------------------------
    def score(self, srcs: List[str], tgts: List[str], batch_size: int = 4):
        """Score (src, tgt) pairs, guarding against overly‑long inputs."""
        assert len(srcs) == len(tgts), "srcs and tgts must be same length"

        # pre‑truncate texts to avoid the tokenizer overflow bug
        srcs_trunc = [self._truncate_text(s) for s in srcs]
        tgts_trunc = [self._truncate_text(t) for t in tgts]

        scores: List[float] = []
        for i in range(0, len(srcs_trunc), batch_size):
            sb = srcs_trunc[i : i + batch_size]
            tb = tgts_trunc[i : i + batch_size]
            try:
                with torch.no_grad():
                    enc_src = self.tokenizer(
                        sb,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    enc_tgt = self.tokenizer(
                        tb,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )

                    src_ids = enc_src.input_ids.to(self.device)
                    src_mask = enc_src.attention_mask.to(self.device)
                    tgt_ids = enc_tgt.input_ids.to(self.device)
                    tgt_mask = enc_tgt.attention_mask.to(self.device)
                    tgt_lens = tgt_mask.sum(dim=1)

                    out = self.model(
                        input_ids=src_ids,
                        attention_mask=src_mask,
                        decoder_attention_mask=tgt_mask,
                        labels=tgt_ids,
                    )

                    logp = self.lsm(out.logits.view(-1, self.model.config.vocab_size))
                    losses = self.loss_fct(logp, tgt_ids.view(-1)).view(tgt_ids.size(0), -1)
                    scores.extend([-(l.sum().item() / ln.item()) for l, ln in zip(losses, tgt_lens)])

            except (RuntimeError, OverflowError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Batch starting at {i} failed: {e}") from e

        return scores

    def multi_ref_score(
        self,
        srcs: List[str],
        tgts: List[List[str]],
        agg: str = "mean",
        batch_size: int = 4,
    ):
        # Check uniform references per sample
        ref_counts = [len(r) for r in tgts]
        if len(set(ref_counts)) > 1:
            raise ValueError("All examples must have the same number of references.")
        num_refs = ref_counts[0]

        # Score each reference column
        matrix = []
        for idx in range(num_refs):
            column = [refs[idx] for refs in tgts]
            matrix.append(self.score(srcs, column, batch_size))
        arr = np.array(matrix)
        if agg == "mean":
            return list(arr.mean(axis=0))
        elif agg == "max":
            return list(arr.max(axis=0))
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

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

Fine-tuning on downstream tasks (e.g., summarization, paraphrasing) and prompt engineering further enhance BARTScore's adaptability to different domains.

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

## Citation

```
@inproceedings{10.5555/3540261.3542349,
   author = {Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
   title = {BARTSCORE: evaluating generated text as text generation},
   year = {2021},
   isbn = {9781713845393},
   publisher = {Curran Associates Inc.},
   address = {Red Hook, NY, USA},
   booktitle = {Proceedings of the 35th International Conference on Neural Information Processing Systems},
   articleno = {2088},
   numpages = {15},
   series = {NIPS '21}
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""

    def __init__(
        self, batch_size: int = 4, model: str = "facebook/bart-large-cnn"
    ):
        # Get device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Pass all parameters explicitly to parent constructor for caching
        super().__init__(
            name=f"BARTScore_{model.split('/')[-1]}",
            description=(
                "BARTScore is a reference-based metric for evaluating text quality "
                "using a pre-trained BART model to compute likelihoods."
            ),
            batch_size=batch_size,
            model=model,
            device=str(self.device)  # Include device as string
        )
        
        # Initialize scorer
        self.bart_scorer = BARTScorer(
            device=str(self.device), max_length=None, checkpoint=model
        )
        self.batch_size = batch_size
        
        # Exclude device from cache key as it doesn't affect results, just computation method
        # Likewise, exclude batch_size as it doesn't affect results in theory
        self.exclude_from_cache_key('device', 'batch_size')

    def calculate(self, input: str, output: str, references=None, **kwargs):
        refs = references or []
        if len(refs) > 1:
            scores = self.bart_scorer.multi_ref_score(
                [input], [refs], agg="max", batch_size=self.batch_size
            )
        else:
            scores = self.bart_scorer.score(
                [input], [refs[0] if refs else ""], batch_size=self.batch_size
            )
        return scores[0]

    def calculate_batched(
        self, inputs: List[str], outputs: List[str], references=None, **kwargs
    ):
        refs = references or [[] for _ in inputs]
        groups = {}
        for i, r in enumerate(refs):
            groups.setdefault(len(r), []).append(i)

        all_scores = [0] * len(outputs)
        for ref_count, idxs in groups.items():
            cur_inputs = [inputs[i] for i in idxs]
            cur_refs   = [refs[i] for i in idxs]
            if ref_count > 1:
                sc = self.bart_scorer.multi_ref_score(
                    cur_inputs, cur_refs, agg="max", batch_size=self.batch_size
                )
            else:
                single_refs = [r[0] if r else "" for r in cur_refs]
                sc = self.bart_scorer.score(
                    cur_inputs, single_refs, batch_size=self.batch_size
                )
            for idx, score in zip(idxs, sc):
                all_scores[idx] = score

        return all_scores


if __name__ == "__main__":
    metric = BARTScore()

    # single example
    inp = (
        "Peter and Elizabeth took a taxi to attend the night party in the city. "
        "While in the party, Elizabeth collapsed and was rushed to the hospital."
    )
    out = "Peter and Elizabeth attend party city. Elizabeth rushed hospital."
    refs = ["Elizabeth was hospitalized after attending a party with Peter."]
    print("BARTScore:", metric.calculate(inp, out, references=refs))

    # batched examples
    inputs = [inp, "The cat sat on the mat."]
    outputs = [out, "The cat is on the mat."]
    references = [
        ["Elizabeth was hospitalized after attending a party with Peter."],
        ["The cat sat on the mat.", "The cat is on the mat.", "The cat is on the rug."],
    ]
    print("BARTScore batch scores:", metric.calculate_batched(inputs, outputs, references=references))
