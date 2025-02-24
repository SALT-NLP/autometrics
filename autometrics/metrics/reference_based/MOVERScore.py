from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
import os
from pyemd import emd, emd_with_flow
from torch import nn
from math import log
from itertools import chain

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

from typing import List, Union, Iterable
from itertools import zip_longest


from transformers import AutoTokenizer, AutoModel

from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric


class MOVERScore(ReferenceBasedMetric):
    """---
# Metric Card for MoverScore

MoverScore is a semantic similarity metric for evaluating generated text, leveraging contextualized embeddings (such as BERT) and Earth Mover’s Distance (EMD) to measure the alignment between system outputs and reference texts. It is designed to capture semantic similarity beyond lexical overlap and has been shown to achieve a high correlation with human judgments across tasks like machine translation, summarization, image captioning, and data-to-text generation.

## Metric Details

### Metric Description

MoverScore measures text similarity by computing the minimum cost required to move the distributed representations of words from the generated text to the reference text. It uses contextualized word embeddings (from models like BERT) and optimizes a transport cost matrix to determine the most efficient mapping between words. Unlike surface-level metrics such as BLEU and ROUGE, MoverScore accounts for semantic equivalence even when lexical forms differ.

- **Metric Type:** Semantic Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No (evaluates only system output and reference)
  
### Formal Definition

MoverScore extends **Word Mover’s Distance (WMD)** by incorporating contextualized embeddings. Given a generated sentence $x$ and a reference sentence $y$, let $x_n$ and $y_n$ represent their n-grams. The distance between these sentences is computed as:

$$
\text{WMD}(x _{n}, y _{n}) = \min _{F} \sum _{i,j} C _{ij} F _{ij}
$$

subject to:

$$
F 1 = f _{x _{n}}, \quad F^T 1 = f _{y _{n}}
$$

where:
- $F$ is the transportation flow matrix,
- $C _{ij}$ is the Euclidean distance between the embeddings of n-grams $x _{i}^{n}$ and $y _{j}^{n}$,
- $f _{x _{n}}$ and $f _{y _{n}}$ represent n-gram weight distributions, computed using inverse document frequency (IDF).

MoverScore supports multiple variations, including **Word Mover Distance (WMD) on unigrams/bigrams** and **Sentence Mover Distance (SMD)**.

### Inputs and Outputs

- **Inputs:**  
  - Generated text (system output)  
  - Reference text(s) (gold-standard text)

- **Outputs:**  
  - A scalar similarity score in the range [0,1], where higher values indicate better semantic alignment.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Machine Translation, Summarization, Image Captioning, Data-to-Text Generation

### Applicability and Limitations

- **Best Suited For:**  
  - Tasks where capturing semantic similarity is critical, such as summarization, paraphrasing, and machine translation.  
  - Cases where lexical overlap is insufficient to judge text quality (e.g., abstractive summarization).  

- **Not Recommended For:**  
  - Evaluating grammatical correctness or fluency in isolation.  
  - Tasks where exact lexical matching is the primary evaluation criterion (e.g., extractive summarization).  

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - [MoverScore GitHub Repository](https://github.com/AIPHES/emnlp19-moverscore)  
  - Available in `moverscore.py` (original) and `moverscore_v2.py` (faster but less accurate version).

### Computational Complexity

- **Efficiency:**  
  - MoverScore is computationally more expensive than n-gram-based metrics due to the need for contextualized embeddings and solving an optimal transport problem.
  
- **Scalability:**  
  - Can be slow for large datasets but optimized versions exist (`moverscore_v2.py` uses DistilBERT for speed).

## Known Limitations

- **Biases:**  
  - Performance depends on the pre-trained language model used. Models trained on large-scale English corpora may not generalize well to low-resource languages.
  
- **Task Misalignment Risks:**  
  - May not accurately reflect human preferences when lexical precision is crucial.
  
- **Failure Cases:**  
  - Contextual embeddings may not effectively capture domain-specific terminology or named entity variations.
  
## Related Metrics

- **BERTScore:** Similar to MoverScore but relies on cosine similarity rather than Earth Mover’s Distance.
- **BLEU, ROUGE:** Traditional surface-level n-gram overlap metrics, which MoverScore seeks to improve upon.
- **CIDEr, METEOR:** Alternative semantic similarity-based evaluation metrics.

## Further Reading

- **Papers:**  
  - Zhao et al., 2019. *MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance.* [EMNLP 2019](https://arxiv.org/abs/1909.02622)  
  - Peyrard et al., 2019. *Supervised and Unsupervised Metrics for Machine Translation and Summarization.*

- **Blogs/Tutorials:**  
  - [MoverScore GitHub README](https://github.com/AIPHES/emnlp19-moverscore)  
  - [Evaluating Text Generation with MoverScore](https://arxiv.org/pdf/1909.02622.pdf)

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu  """

    def __init__(self, model_name='distilbert-base-uncased', device='cuda'):
        """
        Construct the MoverScore model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.model_name = model_name

        name = "MOVERScore_" + model_name
        description = "MOVERScore is a metric that computes the similarity between two sentences using a pre-trained BERT model. It is based on the cosine similarity between the embeddings of the two sentences, and it uses the Earth Mover's Distance (EMD) to compute the distance between the two sets of embeddings."
        
        super().__init__(name, description)


    def truncate(self, tokens):
        if len(tokens) > self.tokenizer.model_max_length - 2:
            tokens = tokens[0:(self.tokenizer.model_max_length - 2)]
        return tokens

    def process(self, a):
        a = ["[CLS]"]+self.truncate(self.tokenizer.tokenize(a))+["[SEP]"]
        a = self.tokenizer.convert_tokens_to_ids(a)
        return set(a)


    def get_idf_dict(self, arr, nthreads=4):
        idf_count = Counter()
        num_docs = len(arr)

        process_partial = partial(self.process)

        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

        idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
        idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def padding(self, arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask

    def bert_encode(self, model, x, attention_mask):
        model.eval()
        with torch.no_grad():
            result = model(x, attention_mask = attention_mask)
        if self.model_name == 'distilbert-base-uncased':
            return result[1] 
        else:
            return result[2] 

    #with open('stopwords.txt', 'r', encoding='utf-8') as f:
    #    stop_words = set(f.read().strip().split(' '))

    def collate_idf(self, arr, tokenize, numericalize, idf_dict,
                    pad="[PAD]",device='cuda:0'):
        
        tokens = [["[CLS]"]+self.truncate(tokenize(a))+["[SEP]"] for a in arr]  
        arr = [numericalize(a) for a in tokens]

        idf_weights = [[idf_dict[i] for i in a] for a in arr]
        
        pad_token = numericalize([pad])[0]

        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = self.padding(idf_weights, pad_token, dtype=torch.float)
        padded = padded.to(device=device)
        mask = mask.to(device=device)
        lens = lens.to(device=device)

        return padded, padded_idf, lens, mask, tokens

    def get_bert_embedding(self, all_sens, model, tokenizer, idf_dict,
                        batch_size=-1,device='cuda:0'):

        padded_sens, padded_idf, lens, mask, tokens = self.collate_idf(all_sens,
                                                        tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                        idf_dict,device=device)

        if batch_size == -1: batch_size = len(all_sens)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = self.bert_encode(model, padded_sens[i:i+batch_size],
                                            attention_mask=mask[i:i+batch_size])
                batch_embedding = torch.stack(batch_embedding)
                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens

    def _safe_divide(self, numerator, denominator):
        return numerator / (denominator + 1e-30)

    def batched_cdist_l2(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.baddbmm(
            x2_norm.transpose(-2, -1),
            x1,
            x2.transpose(-2, -1),
            alpha=-2
        ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
        return res

    def word_mover_score(self, refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords = True, batch_size=256,device='cuda:0'):
        preds = []
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start:batch_start+batch_size]
            batch_hyps = hyps[batch_start:batch_start+batch_size]
            
            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self.get_bert_embedding(batch_refs, self.model, self.tokenizer, idf_dict_ref,device=device)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self.get_bert_embedding(batch_hyps, self.model, self.tokenizer, idf_dict_hyp,device=device)

            ref_embedding = ref_embedding[-1]
            hyp_embedding = hyp_embedding[-1]
            
            batch_size = len(ref_tokens)
            for i in range(batch_size):  
                ref_ids = [k for k, w in enumerate(ref_tokens[i]) 
                                    if w in stop_words or '##' in w 
                                    or w in set(string.punctuation)]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) 
                                    if w in stop_words or '##' in w
                                    or w in set(string.punctuation)]
            
                ref_embedding[i, ref_ids,:] = 0                        
                hyp_embedding[i, hyp_ids,:] = 0
                
                ref_idf[i, ref_ids] = 0
                hyp_idf[i, hyp_ids] = 0
                
            raw = torch.cat([ref_embedding, hyp_embedding], 1)
                                
            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
            
            distance_matrix = self.batched_cdist_l2(raw, raw).double().cpu().numpy()
                    
            for i in range(batch_size):  
                c1 = np.zeros(raw.shape[1], dtype=float)
                c2 = np.zeros(raw.shape[1], dtype=float)
                c1[:len(ref_idf[i])] = ref_idf[i]
                c2[len(ref_idf[i]):] = hyp_idf[i]
                
                c1 = self._safe_divide(c1, np.sum(c1))
                c2 = self._safe_divide(c2, np.sum(c2))
                
                dst = distance_matrix[i]
                _, flow = emd_with_flow(c1, c2, dst)
                flow = np.array(flow, dtype=float)
                score = 1./(1. + np.sum(flow * dst))#1 - np.sum(flow * dst)
                preds.append(score)

        return preds
    
    def sentence_score(self, hypothesis: str, references: List[str], trace=0):
        
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
        
        hypothesis = [hypothesis] * len(references)
        
        sentence_score = 0 

        scores = self.word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
        
        sentence_score = np.mean(scores)
        
        if trace > 0:
            print(hypothesis, references, sentence_score)
                
        return sentence_score

    def corpus_score(self, sys_stream: List[str],
                        ref_streams:Union[str, List[Iterable[str]]], trace=0):

        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        fhs = [sys_stream] + ref_streams

        corpus_score = 0
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")
                
            hypo, *refs = lines
            corpus_score += self.sentence_score(hypo, refs, trace=0)
            
        corpus_score /= len(sys_stream)

        return corpus_score
    
    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        if references is None:
            references = []

        output = [output]
        references = [[r] for r in references]

        return self.corpus_score(output, references)
    
if __name__ == "__main__":
    # Example usage
    moverscore = MOVERScore()
    input = "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital."
    output = "Peter and Elizabeth attend party city. Elizabeth rushed hospital."
    references = ["Elizabeth was hospitalized after attending a party with Peter."]
    scores = moverscore.calculate(input, output, references)
    print("MOVERScore scores:", scores)

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
    scores = moverscore.calculate_batched(inputs, outputs, references)
    print("MOVERScore batch scores:", scores)

