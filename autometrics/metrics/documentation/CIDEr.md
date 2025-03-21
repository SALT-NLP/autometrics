---
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
- **Contact:** mryan0@stanford.edu