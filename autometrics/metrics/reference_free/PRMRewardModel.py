import nltk
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from autometrics.metrics.reference_free.ReferenceFreeMultiMetric import ReferenceFreeMultiMetric
from typing import Tuple, List

class MathProcessRewardModel(ReferenceFreeMultiMetric):
    """---
# Metric Card for MathProcessRewardModel (Qwen2.5-Math-PRM-7B)

MathProcessRewardModel is a process-level reward model that evaluates each intermediate step in a multi-step mathematical reasoning problem. Rather than scoring the final answer alone, it provides token-level feedback across a reasoning chain, identifying helpful versus unhelpful steps using a learned binary classifier. This allows for granular supervision of multi-hop reasoning in LLMs and is particularly effective in domains where correctness must be verified incrementally.

## Metric Details

### Metric Description

MathProcessRewardModel evaluates step-by-step mathematical reasoning by assigning a reward score to each reasoning step in a sequence. The model inserts a special token (`<extra_0>`) after each reasoning step and computes the probability that the token is classified as “positive” using a softmax over logits. This yields a scalar between 0 and 1 indicating how helpful or correct the step is deemed to be. The model is trained on labels derived from whether a step leads to a correct solution trajectory, allowing it to generalize to unseen reasoning processes.

- **Metric Type:** Semantic Similarity, Reference-Free, Faithfulness
- **Range:** [0, 1]
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Let $x$ be a problem prompt, and $z_1, z_2, \dots, z_T$ be a sequence of reasoning steps. Let $<\!extra_0\!>$ be a separator token inserted after each step. Let $s_i$ denote the model’s score for step $z_i$.

For each step:
$$
s_i = P(\text{label} = \text{positive} \mid z_i)
$$

This is computed via softmax over the model's logits:
$$
s_i = \text{softmax}(l_i)[\text{positive\_class}]
$$

where $l_i$ are the logits at the token position corresponding to $<\!extra_0\!>$ following step $z_i$.

### Inputs and Outputs

- **Inputs:**  
  - Problem prompt (e.g., math word problem)  
  - Step-by-step reasoning sequence (as multiple text spans, each ending with `<extra_0>`)  

- **Outputs:**  
  - List of step-level scores (floats between 0 and 1), one for each reasoning step

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation
- **Tasks:** Mathematical Reasoning, Step-by-Step Problem Solving, Chain-of-Thought Evaluation

### Applicability and Limitations

- **Best Suited For:**  
  Evaluation of mathematical reasoning chains, especially in contexts where intermediate steps need supervision (e.g., tutoring systems, math QA).

- **Not Recommended For:**  
  Open-ended creative generation tasks, final-answer-only assessments, or settings where reasoning steps are implicit or uninterpretable.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - Hugging Face Transformers ([Model Page](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B))
  - Source example and logic: see [Qwen2.5 PRM README](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)

### Computational Complexity

- **Efficiency:**  
  Inference is linear in the number of steps and token length; a forward pass over the model is required per input with all reasoning steps and inserted tokens.

- **Scalability:**  
  Scales well for moderate-length reasoning chains (~4–20 steps), but cost grows with step count due to softmax computation per `<extra_0>` token.

## Known Limitations

- **Biases:**  
  Needs more information.

- **Task Misalignment Risks:**  
  May be less applicable to domains where step-wise correctness is hard to define or subjective.

- **Failure Cases:**  
  Needs more information.

## Related Metrics

- **Outcome Reward Models (ORM):** Only evaluate final answer correctness.
- **BERTScore:** Evaluates semantic similarity but not process reasoning.
- **Verifier Models:** Sometimes used to verify entire chains rather than local steps.

## Further Reading

- **Papers:**  
  - [Zhang et al. (2025) - The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)

- **Blogs/Tutorials:**  
  - [Stephen Diehl – Process Reward Models](https://www.stephendiehl.com/posts/process-reward-models.html)

## Citation

```
@misc{zhang2025lessonsdevelopingprocessreward,
      title={The Lessons of Developing Process Reward Models in Mathematical Reasoning}, 
      author={Zhenru Zhang and Chujie Zheng and Yangzhen Wu and Beichen Zhang and Runji Lin and Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin},
      year={2025},
      eprint={2501.07301},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.07301}, 
}
```

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from generative AI. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    def __init__(
        self,
        name: str = "PRMRewardModel",
        description: str = "Process Reward Model Qwen2.5-Math-PRM-7B sentence-based min/max/mean",
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device_map=None,
        persistent: bool = True,
        **kwargs
    ):
        super().__init__(name, description, submetric_names=["min", "max", "mean"], model_id=model_name, device_map=device_map, persistent=persistent, **kwargs)
        self.model_name = model_name
        self.device_map = device_map
        self.persistent = persistent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        if self.persistent:
            self._load_model()

        self.exclude_from_cache_key('device_map', 'persistent')

    def _load_model(self):
        if self.model is None:
            # Download sentence tokenizer if needed
            nltk.download('punkt', quiet=True)
            # Load tokenizer and model with remote code
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).eval()

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None

    def _make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """Compute positive-class probability for each <extra_0> step in a single forward pass."""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
        all_scores_res: List[List[float]] = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores_res.append(positive_probs.cpu().tolist())
        return all_scores_res

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs) -> Tuple[float, float, float]:
        # Lazy load model if needed
        if self.model is None:
            self._load_model()
        # Ensure sentence tokenizer is available
        nltk.download('punkt', quiet=True)
        # Sentence-tokenize and append <extra_0> separators
        sentences = nltk.sent_tokenize(output)
        assistant_content = "<extra_0>".join(sentences) + "<extra_0>"
        # Build full conversation
        system_prompt = "Please reason step by step."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": assistant_content}
        ]
        conv_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Single forward pass
        input_ids = self.tokenizer.encode(conv_str, return_tensors="pt").to(self.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        # Identify step separators and compute masks
        sep_id = self.tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == sep_id)
        # Compute per-step rewards
        step_rewards_list = self._make_step_rewards(logits, token_masks)[0]
        if not step_rewards_list:
            step_rewards_list = [0.0]
        # Aggregate rewards
        min_score = min(step_rewards_list)
        max_score = max(step_rewards_list)
        mean_score = sum(step_rewards_list) / len(step_rewards_list)
        # Optionally unload model
        if not self.persistent:
            self._unload_model()
        return (min_score, max_score, mean_score) 