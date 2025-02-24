# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


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
BARTScore = \sum _{t=1}^{m} \omega _{t} \log p( y _{t} \mid y _{<t}, x, \theta )
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

    def __init__(self, batch_size=4, model='facebook/bart-large-cnn'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bart_scorer = BARTScorer(device=self.device, checkpoint=model)
        self.batch_size = batch_size

        name = "BARTScore" + "_" + model.split("/")[-1]
        description = "BARTScore is a reference-based metric for evaluating the quality of generated text. It uses a pre-trained BART model to compute the likelihood of the generated text given the reference text, providing a score that reflects the fluency and relevance of the generated content."

        super().__init__(name, description)

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate BARTScore for the given input and output.
        """
        if references is None:
            references = []

        out = None
        if len(references) > 1:
            out = self.bart_scorer.multi_ref_score([output], [references], agg="max", batch_size=self.batch_size)
        else:
            out = self.bart_scorer.score([output], [references[0]], batch_size=self.batch_size)

        return out[0]
    
    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate BARTScore for the given inputs and outputs in batches.
        """
        if references is None:
            references = [[] for _ in range(len(inputs))]

        # BARTScore expects the multireferences to be passed with the same number of references.  This is not a requirement in our setting.
        # Thus we must group the references by the number of references and compute the scores for each group.
        ref_nums = [len(x) for x in references]

        groups = {}
        for i, ref_num in enumerate(ref_nums):
            if ref_num not in groups:
                groups[ref_num] = []
            groups[ref_num].append(i)

        out = []
        out_indices = []
        for ref_num, indices in groups.items():
            curr_outputs = [outputs[i] for i in indices]
            curr_references = [references[i] for i in indices]

            if ref_num > 1:
                curr_out = self.bart_scorer.multi_ref_score(curr_outputs, curr_references, agg="max", batch_size=self.batch_size)
            else:
                curr_out = self.bart_scorer.score(curr_outputs, curr_references[0], batch_size=self.batch_size)

            out.extend(curr_out)
            out_indices.extend(indices)

        # Reorder the scores to match the original order of the outputs
        new_out = [0] * len(outputs)
        for i, index in enumerate(out_indices):
            new_out[index] = out[i]

        return new_out
    
if __name__ == "__main__":
    # Example usage
    unieval = BARTScore()
    input = "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital."
    output = "Peter and Elizabeth attend party city. Elizabeth rushed hospital."
    references = ["Elizabeth was hospitalized after attending a party with Peter."]
    scores = unieval.calculate(input, output, references)
    print("BARTScore scores:", scores)

    # Test batch processing
    inputs = [
        "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.",
        "The cat sat on the mat."
    ]
    outputs = [
        "Peter and Elizabeth attend party city. Elizabeth rushed hospital.",
        "The cat is on the mat."
    ]
    references = [
        ["Elizabeth was hospitalized after attending a party with Peter."],
        ["The cat sat on the mat.", "The cat is on the mat.", "The cat is on the rug."]
    ]
    scores = unieval.calculate_batched(inputs, outputs, references)
    print("UniEvalSum batch scores:", scores)
