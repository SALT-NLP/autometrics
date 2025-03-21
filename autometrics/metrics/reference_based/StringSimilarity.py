from Levenshtein import distance, ratio, hamming, jaro, jaro_winkler
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

def _aggregate(values, method):
    """
    Aggregates a list of numeric values using the specified method.
    
    Args:
        values (list of float/int): The values to aggregate.
        method (str): Aggregation method: "min", "max", or "avg".
    
    Returns:
        The aggregated value.
    """
    if not values:
        return None
    if method == "min":
        return min(values)
    elif method == "max":
        return max(values)
    elif method == "avg":
        return sum(values) / len(values)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

class LevenshteinDistance(ReferenceBasedMetric):
    """---
# Metric Card for Levenshtein Distance

Levenshtein Distance measures the minimum number of single-character edits—insertions, deletions, or substitutions—required to transform one sequence into another. It is a fundamental metric in text processing, error correction, and approximate string matching.

## Metric Details

### Metric Description

Levenshtein Distance calculates the minimal cost of edit operations needed to convert one string into another. The computation is typically performed using a dynamic programming approach that considers insertions, deletions, and substitutions. Users can optionally assign custom weights to each operation (defaulting to 1 for all), allowing the metric to adapt to different application needs.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to $\infty$ (practically 0 to $\max(|s_1|, |s_2|)$ with unit costs)
- **Higher is Better?:** No (lower values indicate greater similarity)
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Given two sequences $s_1$ and $s_2$, and weights $w_{ins}$, $w_{del}$, $w_{sub}$, the Levenshtein Distance $D(i, j)$ is defined as:

$$
D(i, 0) = i \cdot w_{del}, \quad D(0, j) = j \cdot w_{ins}
$$

$$
D(i, j) = \min \begin{cases}
D(i-1, j) + w_{del}, \\
D(i, j-1) + w_{ins}, \\
D(i-1, j-1) + \begin{cases}
0, & \text{if } s_1[i] = s_2[j] \\
w_{sub}, & \text{otherwise}
\end{cases}
\end{cases}
$$

where $1 \leq i \leq |s_1|$, $1 \leq j \leq |s_2|$, and typically $w_{ins} = w_{del} = w_{sub} = 1$.

### Inputs and Outputs

- **Inputs:**  
  - Two sequences (e.g., strings or lists of hashable items) to compare.
  - Optional parameters:
    - `weights`: A tuple $(w_{ins}, w_{del}, w_{sub})$ specifying custom costs.
    - `processor`: A callable to preprocess the inputs.
    - `score_cutoff`: A threshold to limit computation for large distances.
  
- **Outputs:**  
  - An integer representing the computed Levenshtein Distance between the two inputs.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Machine Translation, Summarization, Paraphrasing, Spell Checking, Error Correction

### Applicability and Limitations

- **Best Suited For:**  
  - Evaluating character-level similarity between two sequences.
  - Applications in spell checking, optical character recognition, and error correction where precise, literal differences matter.
  
- **Not Recommended For:**  
  - Tasks requiring semantic or context-aware similarity measures (e.g., creative text generation, open-ended dialogue).
  - Scenarios where reordering or paraphrasing plays a significant role in perceived similarity.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Levenshtein Python module](https://rapidfuzz.github.io/Levenshtein/index.html) – A highly optimized C-based implementation for computing Levenshtein Distance.

### Computational Complexity

- **Efficiency:**  
  - The standard dynamic programming solution operates in $O(m \times n)$ time, where $m$ and $n$ are the lengths of the two input sequences.
  
- **Scalability:**  
  - Memory usage can be optimized to $O(\min(m, n))$ using a two-row technique, making it feasible for moderately sized inputs. However, performance may become an issue for extremely long sequences.

## Known Limitations

- **Biases:**  
  - Focuses solely on literal character differences and does not account for semantic or contextual similarity.
  
- **Task Misalignment Risks:**  
  - May not correlate with human judgments in cases where meaning is preserved despite significant character-level differences.
  
- **Failure Cases:**  
  - In tasks with high variability in acceptable outputs (e.g., creative generation), the metric may yield misleadingly high distances despite semantically similar content.

## Related Metrics

- **Damerau-Levenshtein Distance:** Considers transpositions in addition to insertions, deletions, and substitutions.
- **Hamming Distance:** Measures the number of differing characters for sequences of equal length.
- **ROUGE and BLEU:** Surface-level similarity metrics commonly used in text generation evaluation.
- **BERTScore:** Evaluates semantic similarity using contextual embeddings.

## Further Reading

- **Papers:**  
  - V. I. Levenshtein, "Binary Codes with Correction of Deletions, Insertions, and Substitutions of Symbols", Doklady of the USSR Academy of Sciences, 1965. [Available here](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=31411)
  
- **Blogs/Tutorials:**  
  - [Levenshtein Distance Documentation](https://rapidfuzz.github.io/Levenshtein/index.html)

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu"""
    def __init__(self, weights=(1, 1, 1), processor=None, score_cutoff=None, aggregation="min"):
        metric_name = f"LevenshteinDistance_{aggregation}"
        description = "Computes the Levenshtein distance between the candidate and reference strings."
        super().__init__(metric_name, description)
        self.weights = weights
        self.processor = processor
        self.score_cutoff = score_cutoff
        self.aggregation = aggregation

    def calculate(self, input, output, references=None, **kwargs):
        if references is None or len(references) == 0:
            return 0  # No references; assume zero distance.
        w = kwargs.get("weights", self.weights)
        proc = kwargs.get("processor", self.processor)
        cutoff = kwargs.get("score_cutoff", self.score_cutoff)
        # Compute distance for each reference.
        distances = [
            distance(output, ref, weights=w, processor=proc, score_cutoff=cutoff)
            for ref in references
        ]
        return _aggregate(distances, self.aggregation)

class LevenshteinRatio(ReferenceBasedMetric):
    """---
# Metric Card for Levenshtein Ratio

Levenshtein Ratio is a normalized similarity metric that computes the relative similarity between two sequences by evaluating the minimum number of insertions and deletions required to transform one sequence into the other. The result is expressed as a value between 0 and 1, where 1 indicates identical sequences.

## Metric Details

### Metric Description

Levenshtein Ratio calculates a normalized indel similarity score. It uses the indel distance (i.e., the minimum number of insertions and deletions required to change one sequence into the other) and normalizes this value by the total length of both sequences. This provides a score in the range [0, 1], where higher scores indicate greater similarity.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let $d(s_1, s_2)$ be the indel distance between sequences $s_1$ and $s_2$. The Levenshtein Ratio is defined as:

$$
\text{Levenshtein Ratio} = 1 - \frac{d(s_1, s_2)}{|s_1| + |s_2|}
$$

where $|s_1|$ and $|s_2|$ denote the lengths of the sequences $s_1$ and $s_2$, respectively.

### Inputs and Outputs

- **Inputs:**  
  - Two sequences (e.g., strings or lists of hashable elements) to compare.
  - Optional parameters:
    - `processor`: A callable to preprocess the inputs (default is None).
    - `score_cutoff`: A threshold for early termination, specified as a float between 0 and 1 (default is 0, which deactivates this behavior).
  
- **Outputs:**  
  - A float representing the normalized similarity between the two sequences, ranging from 0 (completely different) to 1 (identical).

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Spell Checking, Error Correction, Approximate String Matching, Quality Evaluation of Generated Text

### Applicability and Limitations

- **Best Suited For:**  
  - Situations requiring a normalized measure of string similarity.
  - Applications where the relative similarity (rather than the absolute number of edit operations) is more informative.
  
- **Not Recommended For:**  
  - Scenarios where semantic similarity is crucial, as the metric only considers literal character differences.
  - Tasks with high variability in acceptable outputs, such as creative text generation.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - [Levenshtein Python module](https://rapidfuzz.github.io/Levenshtein/index.html)

### Computational Complexity

- **Efficiency:**  
  - The computation typically requires $O(m \times n)$ time, where $m$ and $n$ are the lengths of the input sequences.
  
- **Scalability:**  
  - Memory optimizations (e.g., using a two-row dynamic programming approach) allow the metric to scale for moderately sized sequences.

## Known Limitations

- **Biases:**  
  - The metric is sensitive only to literal character differences and does not account for semantic or contextual similarity.
  
- **Task Misalignment Risks:**  
  - May yield low similarity scores for strings that are semantically similar but differ significantly in character order or structure.
  
- **Failure Cases:**  
  - Not effective for comparing sequences where insertions and deletions are less indicative of overall similarity (e.g., when substitutions are more prevalent).

## Related Metrics

- **Levenshtein Distance:** The non-normalized version measuring the absolute number of edit operations.
- **Damerau-Levenshtein Ratio:** A variant that also considers transpositions.
- **BERTScore:** Evaluates semantic similarity using contextual embeddings.

## Further Reading

- **Papers:**  
  - V. I. Levenshtein, "Binary Codes with Correction of Deletions, Insertions, and Substitutions of Symbols", Doklady of the USSR Academy of Sciences, 1965. [Available here](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=31411&option_lang=rus)
  
- **Blogs/Tutorials:**  
  - [Levenshtein Python Module Documentation](https://rapidfuzz.github.io/Levenshtein/index.html)

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu"""
    def __init__(self, processor=None, score_cutoff=None, aggregation="max"):
        metric_name = f"LevenshteinRatio_{aggregation}"
        description = "Computes the normalized Levenshtein ratio between the candidate and reference strings."
        super().__init__(metric_name, description)
        self.processor = processor
        self.score_cutoff = score_cutoff
        self.aggregation = aggregation

    def calculate(self, input, output, references=None, **kwargs):
        if references is None or len(references) == 0:
            return 1.0  # Identical string yields ratio 1.
        proc = kwargs.get("processor", self.processor)
        cutoff = kwargs.get("score_cutoff", self.score_cutoff)
        ratios = [
            ratio(output, ref, processor=proc, score_cutoff=cutoff)
            for ref in references
        ]
        return _aggregate(ratios, self.aggregation)

class HammingDistance(ReferenceBasedMetric):
    """---
# Metric Card for Hamming Distance

Hamming Distance measures the number of positions at which two equal-length sequences differ. It is a fundamental metric in coding theory and information theory, commonly used for error detection and correction in digital communications.

## Metric Details

### Metric Description

Hamming Distance calculates the number of substitutions required to change one sequence into the other, or equivalently, the number of positions where the corresponding symbols differ. This metric is strictly defined for sequences of equal length.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to $n$ (where $n$ is the length of the sequences)
- **Higher is Better?:** No (lower values indicate greater similarity)
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

For two sequences $s$ and $t$ of equal length $n$, the Hamming Distance is defined as:

$$
H(s, t) = \sum_{i=1}^{n} \mathbf{1}\{ s_i \neq t_i \}
$$

where $\mathbf{1}\{ s_i \neq t_i \}$ is an indicator function that equals 1 if $s_i \neq t_i$ and 0 otherwise.

### Inputs and Outputs

- **Inputs:**  
  - Two equal-length sequences (e.g., binary strings, character arrays)
  
- **Outputs:**  
  - An integer representing the number of differing positions between the two sequences.

## Intended Use

### Domains and Tasks

- **Domain:**  
  - Coding Theory  
  - Information Theory
  
- **Tasks:**  
  - Error Detection  
  - Error Correction  
  - Code Evaluation

### Applicability and Limitations

- **Best Suited For:**  
  - Fixed-length sequences in digital communication systems and error correcting codes.
  
- **Not Recommended For:**  
  - Sequences of differing lengths or applications requiring semantic similarity evaluation (e.g., natural language text similarity).

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Custom implementations in Python (using simple loops or vectorized operations in NumPy)
  - Functions available in libraries such as SciPy.
  - For a conceptual overview, see the [Hamming Distance - Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance).

### Computational Complexity

- **Efficiency:**  
  - Operates in $O(n)$ time, where $n$ is the length of the sequences.
  
- **Scalability:**  
  - Highly efficient for moderate-length sequences; performance scales linearly with sequence length.

## Known Limitations

- **Biases:**  
  - Considers only literal character differences without accounting for semantic or contextual similarities.
  
- **Task Misalignment Risks:**  
  - Not applicable to sequences of unequal length.
  
- **Failure Cases:**  
  - Use with unequal-length inputs will result in errors or undefined behavior.

## Related Metrics

- **Levenshtein Distance:** Measures the minimum edit operations (insertions, deletions, substitutions) required to transform one sequence into another.
- **Damerau-Levenshtein Distance:** Extends Levenshtein Distance by considering transpositions as well.
- **Jaccard Index:** Evaluates similarity based on set overlap, rather than positional differences.

## Further Reading

- **Papers:**  
  - Richard W. Hamming, "Error detecting and error correcting codes", *The Bell System Technical Journal*, 1950. [Available here](https://hdl.handle.net/10945/64206)
  
- **Blogs/Tutorials:**  
  - [Hamming Distance - Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance)

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu"""
    def __init__(self, pad=True, processor=None, score_cutoff=None, aggregation="min"):
        metric_name = f"HammingDistance_{aggregation}"
        description = "Computes the Hamming distance between the candidate and reference strings."
        super().__init__(metric_name, description)
        self.pad = pad
        self.processor = processor
        self.score_cutoff = score_cutoff
        self.aggregation = aggregation

    def calculate(self, input, output, references=None, **kwargs):
        if references is None or len(references) == 0:
            return 0
        pad = kwargs.get("pad", self.pad)
        proc = kwargs.get("processor", self.processor)
        cutoff = kwargs.get("score_cutoff", self.score_cutoff)
        distances = [
            hamming(output, ref, pad=pad, processor=proc, score_cutoff=cutoff)
            for ref in references
        ]
        return _aggregate(distances, self.aggregation)

class JaroSimilarity(ReferenceBasedMetric):
    """---
# Metric Card for Jaro Similarity

Jaro Similarity is a string metric used to measure the similarity between two strings based on the number of matching characters and the number of transpositions. It produces a score between 0 and 1, where 1 indicates an exact match and 0 indicates no similarity.

## Metric Details

### Metric Description

Jaro Similarity computes a score by comparing two strings as follows:
- **Matching Characters:** Two characters from the two strings are considered matching if they are the same and not farther than $\lfloor\max(|s_1|, |s_2|)/2\rfloor - 1$ positions apart.
- **Transpositions:** Transpositions are counted as half the number of matching characters that appear in a different order.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let $s_1$ and $s_2$ be two strings with lengths $|s_1|$ and $|s_2|$, respectively. Let $m$ denote the number of matching characters, and $t$ denote half the number of transpositions. The Jaro Similarity $J$ is defined as:

$$
J = \frac{1}{3} \left( \frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m - t}{m} \right)
$$

### Inputs and Outputs

- **Inputs:**  
  - Two strings (or sequences) to compare.
  
- **Outputs:**  
  - A float representing the similarity score in the range [0, 1].

## Intended Use

### Domains and Tasks

- **Domain:**  
  - Record Linkage
  
- **Tasks:**  
  - Record Linkage, Data Deduplication, Approximate String Matching

### Applicability and Limitations

- **Best Suited For:**  
  - Comparing strings in applications such as record linkage or data deduplication where a normalized similarity score is useful.
  
- **Not Recommended For:**  
  - Applications requiring semantic or context-aware similarity, as this metric considers only character-level differences.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Implementations are available in multiple programming languages (Python, C++, Java, etc.). For example, a detailed explanation and implementation can be found on [GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)

### Computational Complexity

- **Efficiency:**  
  - The algorithm operates in $O(|s_1| \times |s_2|)$ time.
  
- **Scalability:**  
  - Suitable for comparing moderate-length strings; performance may degrade with very long strings.

## Known Limitations

- **Biases:**  
  - The metric is sensitive to the chosen matching window, which may affect the similarity score for strings of varying lengths.
  
- **Task Misalignment Risks:**  
  - May not perform well in scenarios where higher-level semantic similarity is required.
  
- **Failure Cases:**  
  - Less effective when strings contain numerous transpositions or when small differences are critical.

## Related Metrics

- **Jaro-Winkler Similarity:** A variant that incorporates a prefix scale to give extra weight to common prefixes.
- **Levenshtein Distance:** Measures the number of edit operations (insertions, deletions, substitutions) needed to transform one string into another.
- **Damerau-Levenshtein Distance:** Extends Levenshtein by also considering transpositions as a valid edit operation.

## Further Reading

- **Papers:**  
  - Matthew A. Jaro, "Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida", *Journal of the American Statistical Association*, 1989. [Available here](https://www.jstor.org/stable/2289924) Jaro-AdvancesRecordLinkageMethodology-1989.pdf](https://www.jstor.org/stable/2289924)
  
- **Blogs/Tutorials:**  
  - [Jaro and Jaro-Winkler Similarity - GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu"""
    def __init__(self, processor=None, score_cutoff=None, aggregation="max"):
        metric_name = f"JaroSimilarity_{aggregation}"
        description = "Computes the Jaro similarity between the candidate and reference strings."
        super().__init__(metric_name, description)
        self.processor = processor
        self.score_cutoff = score_cutoff
        self.aggregation = aggregation

    def calculate(self, input, output, references=None, **kwargs):
        if references is None or len(references) == 0:
            return 0.0
        proc = kwargs.get("processor", self.processor)
        cutoff = kwargs.get("score_cutoff", self.score_cutoff)
        similarities = [
            jaro(output, ref, processor=proc, score_cutoff=cutoff)
            for ref in references
        ]
        return _aggregate(similarities, self.aggregation)

class JaroWinklerSimilarity(ReferenceBasedMetric):
    """---
# Metric Card for Jaro-Winkler Similarity

Jaro-Winkler Similarity is a string metric that builds upon the Jaro Similarity by incorporating a prefix scale to give extra weight to common prefixes. It is widely used in record linkage, data deduplication, and other applications where matching similar strings (such as names) is critical.

## Metric Details

### Metric Description

Jaro-Winkler Similarity adjusts the base Jaro Similarity score by factoring in the length of the common prefix (up to 4 characters) between two strings. This enhancement increases the similarity score for strings that match from the beginning. The similarity score is normalized between 0 and 1, where 1 indicates an exact match.

- **Metric Type:** Surface-Level Similarity
- **Range:** 0 to 1
- **Higher is Better?:** Yes
- **Reference-Based?:** Yes
- **Input-Required?:** Yes

### Formal Definition

Let $J$ be the Jaro Similarity between two strings $s_1$ and $s_2$, defined as:

$$
J = \frac{1}{3} \left( \frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m - t}{m} \right)
$$

where:
- $m$ is the number of matching characters,
- $t$ is half the number of transpositions.

The Jaro-Winkler Similarity, $JW$, is then given by:

$$
JW = J + l \cdot p \cdot (1 - J)
$$

where:
- $l$ is the length of the common prefix (maximum 4),
- $p$ is a constant scaling factor (typically 0.1).

### Inputs and Outputs

- **Inputs:**  
  - Two strings to compare.
  - Optional parameters:
    - `prefix_weight` ($p$), default is 0.1.
    - Maximum prefix length, default is 4.
  
- **Outputs:**  
  - A float representing the similarity score in the range [0, 1].

## Intended Use

### Domains and Tasks

- **Domain:**  
  - Record Linkage  
  - Data Deduplication  
  - Information Retrieval
  
- **Tasks:**  
  - Matching names and addresses in databases.
  - Detecting duplicate records.
  - Evaluating string similarity in record linkage.

### Applicability and Limitations

- **Best Suited For:**  
  - Applications where the initial part of the string is particularly significant (e.g., surnames in record linkage).
  - Scenarios where slight typographical errors need to be tolerated.
  
- **Not Recommended For:**  
  - Cases where the common prefix is not indicative of overall similarity.
  - Applications that require a metric adhering to the triangle inequality (Jaro-Winkler does not satisfy this).

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**  
  - Implementations are available in multiple programming languages (Python, C++, Java, etc.).
  - Detailed implementations and explanations can be found on [GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)
  - The [Levenshtein module](https://rapidfuzz.github.io/Levenshtein/levenshtein.html#jaro-winkler) documentation provides context on related edit-distance metrics.
  - The original paper by Winkler (1990), "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage", outlines the theoretical foundation. [here](https:files.eric.ed.gov:fulltext:ED325505.pdf)

### Computational Complexity

- **Efficiency:**  
  - The algorithm typically operates in $O(|s_1| \times |s_2|)$ time.
  
- **Scalability:**  
  - Suitable for moderate-length strings, with performance diminishing for very long strings.

## Known Limitations

- **Biases:**  
  - The metric heavily weights the common prefix, which might not be appropriate for all types of data.
  
- **Task Misalignment Risks:**  
  - It may overestimate similarity for strings with similar beginnings but divergent endings.
  
- **Failure Cases:**  
  - Less effective when the prefix does not carry significant meaning or when errors occur predominantly beyond the prefix.

## Related Metrics

- **Jaro Similarity:** The base metric without the prefix adjustment.
- **Levenshtein Distance:** Measures the number of edit operations required to transform one string into another.
- **Damerau-Levenshtein Distance:** Extends Levenshtein by also considering transpositions.
- **Hamming Distance:** Counts differences in fixed-length strings.

## Further Reading

- **Papers:**  
  - Winkler, W. E. (1990). "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage." [Available here](https://files.eric.ed.gov/fulltext/ED325505.pdf)
  - Jaro, M. A. (1989). "Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida." [Available here](https://www.jstor.org/stable/2289924)
  
- **Blogs/Tutorials:**  
  - [Jaro and Jaro-Winkler Similarity - GeeksforGeeks](https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/)

## Metric Card Authors

- **Authors:** Michael J. Ryan
- **Acknowledgment of AI Assistance:**
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT (o3-mini-high). All content has been reviewed and curated by the author to ensure accuracy.
- **Contact:** mryan0@stanford.edu"""
    def __init__(self, prefix_weight=0.1, processor=None, score_cutoff=None, aggregation="max"):
        metric_name = f"JaroWinklerSimilarity_{aggregation}"
        description = "Computes the Jaro-Winkler similarity between the candidate and reference strings."
        super().__init__(metric_name, description)
        self.prefix_weight = prefix_weight
        self.processor = processor
        self.score_cutoff = score_cutoff
        self.aggregation = aggregation

    def calculate(self, input, output, references=None, **kwargs):
        if references is None or len(references) == 0:
            return 0.0
        prefix_weight = kwargs.get("prefix_weight", self.prefix_weight)
        proc = kwargs.get("processor", self.processor)
        cutoff = kwargs.get("score_cutoff", self.score_cutoff)
        similarities = [
            jaro_winkler(output, ref, prefix_weight=prefix_weight, processor=proc, score_cutoff=cutoff)
            for ref in references
        ]
        return _aggregate(similarities, self.aggregation)