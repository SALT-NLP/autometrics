# ------------------------------------------------------------------------------
# This implementation of the CIDEr metric is based on:
#   Vedantam, Zitnick, and Parikh (2015), "CIDEr: Consensus-based Image Description Evaluation"
# and incorporates elements from the reference implementation available in the coco-caption repository:
#   https://github.com/tylin/coco-caption/blob/master/pycocoevalcap
#
# Credit to Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu> for most of the original code.
# ------------------------------------------------------------------------------

import math
import numpy as np
import copy
from collections import defaultdict
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

# ----------------------------
# Internal Functions and Classes
# ----------------------------

def precook(s, n=4, out=False):
    """
    Convert a sentence string into a dictionary of n-gram counts.
    
    Args:
        s (str): The input sentence.
        n (int): Maximum n-gram length.
        out (bool): Unused flag for compatibility.
        
    Returns:
        dict: Mapping from n-gram tuple to count.
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    """
    Process a list of reference sentences.
    
    Args:
        refs (list of str): Reference sentences.
        n (int): Maximum n-gram length.
        
    Returns:
        list of dict: List of n-gram count dictionaries for each reference.
    """
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    """
    Process a single candidate sentence.
    
    Args:
        test (str): Candidate sentence.
        n (int): Maximum n-gram length.
        
    Returns:
        dict: n-gram count dictionary for the candidate.
    """
    return precook(test, n, True)

class CiderScorer(object):
    """
    Class for computing the CIDEr score.
    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        """
        Append a candidate and its references to the internal lists.
        """
        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
            if test is not None:
                self.ctest.append(cook_test(test, self.n))
            else:
                self.ctest.append(None)

    def __iadd__(self, other):
        """
        Overload the += operator to add a candidate and references tuple.
        """
        if isinstance(other, tuple):
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """
        Compute document frequencies for all n-grams in the references.
        """
        for refs in self.crefs:
            for ref in refs:
                for (ngram, count) in ref.items():
                    self.document_frequency[ngram] += 1

    def compute_cider(self):
        """
        Compute the CIDEr score for each candidate.
        
        Returns:
            list: CIDEr score for each candidate.
        """
        def counts2vec(cnts):
            """
            Map n-gram counts to a TF-IDF vector.
            
            Returns:
                vec: List of dicts (one per n-gram length).
                norm: List of vector norms.
                length: Total count for bigrams (used for length penalty).
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            norm = [0.0 for _ in range(self.n)]
            length = 0
            for (ngram, term_freq) in cnts.items():
                # Calculate inverse document frequency (IDF)
                df = np.log(max(1.0, self.document_frequency[ngram]))
                index = len(ngram) - 1  # index 0 for unigrams, etc.
                vec[index][ngram] = float(term_freq) * (self.ref_len - df)
                norm[index] += vec[index][ngram] ** 2
                # For length penalty, using bigrams
                if len(ngram) == 2:
                    length += term_freq
            norm = [np.sqrt(x) for x in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute cosine similarity with a Gaussian length penalty.
            """
            delta = float(length_hyp - length_ref)
            sim_scores = np.array([0.0 for _ in range(self.n)])
            for i in range(self.n):
                for (ngram, value) in vec_hyp[i].items():
                    ref_val = vec_ref[i].get(ngram, 0.0)
                    sim_scores[i] += min(value, ref_val) * ref_val
                if norm_hyp[i] != 0 and norm_ref[i] != 0:
                    sim_scores[i] /= (norm_hyp[i] * norm_ref[i])
                sim_scores[i] *= np.exp(- (delta ** 2) / (2 * self.sigma ** 2))
            return sim_scores

        self.ref_len = np.log(float(len(self.crefs)))
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score) / len(refs)
            score_avg *= 10.0  # Scaling factor for numerical consistency
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        """
        Compute the overall CIDEr score.
        
        Returns:
            (float, np.array): Mean CIDEr score and array of individual scores.
        """
        self.compute_doc_freq()
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)

class CiderImpl:
    """
    Internal implementation of CIDEr that mimics the original 'Cider' class.
    """
    def __init__(self, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Compute the CIDEr score given ground truth and candidate dictionaries.
        
        Args:
            gts (dict): Keys are IDs, values are lists of reference sentences.
            res (dict): Keys are IDs, values are lists containing a single candidate sentence.
        
        Returns:
            (float, np.array): Mean CIDEr score and array of individual scores.
        """
        assert(gts.keys() == res.keys())
        imgIds = list(gts.keys())
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for imgId in imgIds:
            hypo = res[imgId]
            ref = gts[imgId]
            # Sanity checks
            assert(isinstance(hypo, list))
            assert(len(hypo) == 1)
            assert(isinstance(ref, list))
            assert(len(ref) > 0)
            cider_scorer += (hypo[0], ref)
        score, scores = cider_scorer.compute_score()
        return score, scores

    def method(self):
        return "CIDEr"

# ----------------------------
# Public CIDEr Metric Class
# ----------------------------

class CIDEr(ReferenceBasedMetric):
    """---
# Metric Card for CIDEr

CIDEr (Consensus-based Image Description Evaluation) evaluates the quality of generated textual descriptions by measuring their similarity to a consensus of human-written reference captions. Primarily developed for image captioning, CIDEr incorporates lexical overlap and content consensus through TF-IDF weighted n-gram matching.

## Metric Details

### Metric Description

CIDEr measures how closely a candidate caption matches the consensus of multiple human reference captions. It uses Term Frequency-Inverse Document Frequency (TF-IDF) weighting to emphasize informative n-grams (1 to 4 words). Cosine similarity between candidate and reference captions captures both precision and recall, rewarding descriptions aligning closely with multiple human references.

- **Metric Type:** Surface-Level Similarity
- **Range:** Typically 0 to 10 (scaled)
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** No

### Formal Definition

CIDEr first calculates TF-IDF weights for each n-gram ($\omega_k$):

$$
g_k(s_{ij}) = \frac{h_k(s_{ij})}{\sum_{\omega_l \in \Omega} h_l(s_{ij})} \log\left(\frac{|I|}{\sum_{I_p \in I} \min(1, \sum_q h_k(s_{pq}))}\right)
$$

Then, CIDEr computes cosine similarity across reference sentences ($S_i$) for each n-gram level ($n$):

$$
\text{CIDEr}_n(c_i, S_i) = \frac{1}{m}\sum_j \frac{g^n(c_i) \cdot g^n(s_{ij})}{\|g^n(c_i)\|\,\|g^n(s_{ij})\|}
$$

The final CIDEr score averages across all n-gram levels (N=4):

$$
\text{CIDEr}(c_i, S_i) = \frac{1}{N}\sum_{n=1}^{N}\text{CIDEr}_n(c_i, S_i)
$$

### Inputs and Outputs

- **Inputs:**  
  - Candidate sentence (generated caption)  
  - Multiple reference sentences (human-generated captions)

- **Outputs:**  
  - Scalar CIDEr score (typically 0 to 10)

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Image Captioning, Multimodal Generation
- **Tasks:** Image-to-Text Generation, Data-to-Text Generation

### Applicability and Limitations

- **Best Suited For:**  
  Evaluating image captions or short descriptive texts where multiple reference captions are available, and capturing consensus is essential.

- **Not Recommended For:**  
  Tasks involving open-ended, creative, or semantically diverse descriptions (e.g., storytelling, creative writing), where consensus and lexical overlap are not reliable indicators of quality.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [coco-caption](https://github.com/tylin/coco-caption) (official MS COCO evaluation implementation)  
  - [autometrics](https://github.com/autometrics-dev/autometrics)

### Computational Complexity

- **Efficiency:**  
  Moderate efficiency; complexity scales linearly with candidate length and number of references due to TF-IDF and cosine similarity computations.

- **Scalability:**  
  Scalable for typical captioning datasets; computationally heavier on large datasets due to global IDF calculation.

## Known Limitations

- **Biases:** Needs more information.
- **Task Misalignment Risks:**  
  CIDEr can underestimate semantically correct but lexically distinct captions due to strict reliance on lexical overlap.
- **Failure Cases:**  
  CIDEr fails to adequately evaluate semantically rich paraphrases or descriptions using uncommon wording.

## Related Metrics

- **BLEU:** Precision-based n-gram overlap metric.
- **ROUGE:** Recall-based summarization metric.
- **METEOR:** Incorporates semantic similarity and alignment, less strict on lexical matching.

## Further Reading

- **Papers:**  
  - [CIDEr: Consensus-based Image Description Evaluation (Vedantam et al., 2015)](https://arxiv.org/abs/1411.5726)  
  - [Microsoft COCO Captions: Data Collection and Evaluation Server](https://arxiv.org/abs/1504.00325)

- **Blogs/Tutorials:**  
  - [OECD.AI CIDEr Metric Overview](https://oecd.ai/en/catalogue/metrics/consensus-based-image-description-evaluation-cider)

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu"""
    def __init__(self, n=4, sigma=6.0, name="CIDEr", description="CIDEr measures consensus between a candidate sentence and its references."):
        super().__init__(name + "_n" + str(n) + "_sig" + str(sigma), description)
        self._n = n
        self._sigma = sigma

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the CIDEr score for a single candidate sentence.
        
        Args:
            input: (Unused) Placeholder for interface consistency.
            output (str): The candidate sentence.
            references (list of str): List of reference sentences.
        
        Returns:
            float: The computed CIDEr score.
        """
        if references is None:
            references = []
        # Wrap the candidate and references in dictionaries using a dummy key (0)
        gts = {0: references}
        res = {0: [output]}
        cider_impl = CiderImpl(n=self._n, sigma=self._sigma)
        score, _ = cider_impl.compute_score(gts, res)
        return score