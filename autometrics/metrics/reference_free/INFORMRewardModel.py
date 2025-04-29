# autometrics/metrics/reference_free/INFORMRewardModel.py

import torch
from typing import List, Optional, Union
from transformers import LlamaPreTrainedModel, LlamaModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric


class INFORMForSequenceClassification(LlamaPreTrainedModel):
    """
    Sequence classification head on Llama for INF-ORM reward.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, self.num_labels)
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        # find last non-padding position per sequence
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is not None:
            seq_lens = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            seq_lens = seq_lens % input_ids.shape[-1]
            seq_lens = seq_lens.to(logits.device)
        else:
            if batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no pad_token_id is set.")
            seq_lens = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), seq_lens]

        return SequenceClassifierOutputWithPast(
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class INFORMRewardModel(ReferenceFreeMetric):
    """---
# Metric Card for INFORM Reward Model 70B

The INFORM Reward Model 70B (INF-ORM-Llama3.1-70B) is a large-scale outcome reward model designed to evaluate the quality of generated conversational responses. It predicts scalar reward scores for response texts, supporting preference-based fine-grained evaluations without requiring a reference response. The model is finetuned from the Llama-3.1-70B-Instruct backbone using preference-labeled datasets, employing scaled Bradley-Terry loss to incorporate preference magnitudes.

## Metric Details

### Metric Description

INFORM Reward Model 70B measures the quality of generated responses by assigning scalar reward scores. It uses a fine-tuned Llama 3.1-70B-Instruct model trained on paired comparisons, with annotated preference magnitudes indicating how much better one response is than another. A modified score head projects the hidden states to reward scores, and the model employs a scaled Bradley-Terry loss to better reflect differences in human preference strengths.

During training, human preference annotations originally assigned discrete scores of 1, 2, or 3 (for slight, better, or much better). These were **rescaled** during dataset preparation to magnitudes of 1, 3, and 10 respectively, amplifying stronger preferences to better guide the model's optimization.

- **Metric Type:** Reference-Free
- **Range:** Unbounded (observed values typically between approximately -33 and +3)
- **Higher is Better?:** Yes
- **Reference-Based?:** No
- **Input-Required?:** Yes

### Formal Definition

Given a prompt and a candidate response $y$, the INFORM Reward Model predicts a scalar reward $r(x, y)$.

For training, it optimizes the **Scaled Bradley-Terry loss**:

$$
L_{\text{Scaled-BT}} = -d \log(\sigma(r(x, y_{\text{chosen}}) - r(x, y_{\text{rejected}})))
$$

where:
- $d$ is the magnitude of preference between the chosen and rejected responses (scaled to values like 1, 3, or 10),
- $\sigma$ is the sigmoid function,
- $r(x, y)$ is the predicted reward for response $y$ given prompt $x$.

### Inputs and Outputs

- **Inputs:**  
  - Conversation history including user input and model response (as tokenized chat sequences).
  
- **Outputs:**  
  - A scalar reward score (floating point) indicating the response quality. Higher values indicate better responses.

## Intended Use

### Domains and Tasks

- **Domain:** Text Generation, Dialogue Systems
- **Tasks:** Dialogue Generation, Response Generation

### Applicability and Limitations

- **Best Suited For:**  
  - Comparing the quality of candidate responses in dialogue or conversation settings, particularly for tasks where reference outputs are unavailable.
  - Reward modeling for RLHF (Reinforcement Learning from Human Feedback) setups.

- **Not Recommended For:**  
  - Tasks requiring direct evaluation against reference answers (e.g., machine translation).
  - Evaluation scenarios where absolute calibration of scores is necessary.

## Metric Implementation

### Reference Implementations

- **Libraries/Packages:**
  - Hugging Face Transformers (custom Llama3.1-70B model checkpoint: [https://huggingface.co/infly/INF-ORM-Llama3.1-70B](https://huggingface.co/infly/INF-ORM-Llama3.1-70B))
  
### Computational Complexity

- **Efficiency:**  
  Efficient at inference time; a single forward pass per response is needed. Complexity is dominated by a transformer pass ($O(n \cdot d)$ where $n$ is sequence length and $d$ is hidden dimension).

- **Scalability:**  
  Scales linearly with batch size and input length; requires significant memory (70B parameters). Intended for GPU-based inference.

## Known Limitations

- **Biases:**  
  - Potential biases inherited from training datasets, including topic or stylistic biases present in the preference judgments.
  - Training rescaling of preference magnitudes (1 → 1, 2 → 3, 3 → 10) may amplify annotator subjectivity and increase sensitivity to preference errors.

- **Task Misalignment Risks:**  
  - May not align well with human preferences for tasks outside of the dialogue domain.
  - Risk of misinterpreting slight vs. large differences in quality due to score scaling.

- **Failure Cases:**  
  - Struggles in evaluating extremely diverse or creative responses where strict preference orders are unclear.
  - Calibration across very different prompt domains is not guaranteed.

## Related Metrics

- **Bradley-Terry Loss Models:** Standard Bradley-Terry models trained without magnitude scaling.
- **Scaled BT Models:** Models using magnitude information outside the log-sigmoid, as explored in [HelpSteer2-Preference](https://arxiv.org/pdf/2410.01257).
- **RewardBench Metrics:** INFORM Reward Model was benchmarked on RewardBench and compared against other reward models.

## Further Reading

- **Papers:**  
  - [INF-ORM-Llama3.1-70B Model Card on Hugging Face](https://huggingface.co/infly/INF-ORM-Llama3.1-70B)  
  - [HELPSTEER2-PREFERENCE: COMPLEMENTING RATINGS WITH PREFERENCES (ICLR 2025)](https://arxiv.org/pdf/2410.01257) (influential but not official paper)

- **Blogs/Tutorials:**  
  - None officially provided. (Needs more information)

## Metric Card Authors

- **Authors:** Michael J. Ryan  
- **Acknowledgment of AI Assistance:**  
  Portions of this metric card were drafted with assistance from OpenAI's ChatGPT, based on user-provided inputs and reference documentation. All content has been reviewed and curated by the author to ensure accuracy.  
- **Contact:** mryan0@stanford.edu"""
    def __init__(
        self,
        name: str = "INFORMRewardModel",
        description: str = "INF-ORM-Llama3.1-70B outcome reward model (reference-free).",
        model_name: str = "infly/INF-ORM-Llama3.1-70B",
        torch_dtype = torch.bfloat16,
        device_map: Union[str, dict] = "auto",
        attn_implementation: str = "flash_attention_2",
        num_labels: int = 1,
        batch_size: int = 2,
        persistent: bool = True
    ):
        super().__init__(name, description)
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.persistent = persistent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[INFORMForSequenceClassification] = None
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None

        if self.persistent:
            self._load_model()

    def _load_model(self):
        if self.model is None:
            self.model = INFORMForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                attn_implementation=self.attn_implementation,
                num_labels=self.num_labels
            )
            self.model.eval()
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_name)

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None

    def calculate(self, input: str, output: str, **kwargs) -> float:
        # ensure model & tokenizer loaded
        if self.model is None:
            self._load_model()

        # wrap into chat history
        conv = [
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ]
        tok = self.tokenizer.apply_chat_template(
            conv, tokenize=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tok).logits
            score = out.squeeze().cpu().item()

        if not self.persistent:
            self._unload_model()

        return score

    def calculate_batched(self, inputs: List[str], outputs: List[str], **kwargs) -> List[float]:
        if self.model is None:
            self._load_model()

        all_scores: List[float] = []
        # process in chunks
        for i in range(0, len(inputs), self.batch_size):
            batch_in = inputs[i : i + self.batch_size]
            batch_out = outputs[i : i + self.batch_size]
            for inp, out in zip(batch_in, batch_out):
                conv = [
                    {"role": "user", "content": inp},
                    {"role": "assistant", "content": out}
                ]
                tok = self.tokenizer.apply_chat_template(
                    conv, tokenize=True, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    sco = self.model(**tok).logits.squeeze().cpu().item()
                all_scores.append(sco)

        if not self.persistent:
            self._unload_model()

        return all_scores
