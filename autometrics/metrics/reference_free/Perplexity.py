"""
This file is based on the reference implementation of Fixed-Length Perplexity from:
https://huggingface.co/docs/transformers/en/perplexity
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from tqdm import tqdm
from accelerate.test_utils.testing import get_backend
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric


# -------------------------------
# Helper Functions
# -------------------------------

def maybe_tqdm(iterable, progress_bar=True, **tqdm_kwargs):
    """Wrap iterable with tqdm if progress_bar is True."""
    return tqdm(iterable, **tqdm_kwargs) if progress_bar else iterable


def tokenize_texts(texts, tokenizer):
    """
    Tokenize each document (string) using the provided tokenizer.
    Returns a list of 1D tensors (input_ids).
    """
    tokenized_texts = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)  # shape: (L,)
        tokenized_texts.append(input_ids)
    return tokenized_texts


def get_windows(input_ids, max_length, stride):
    """
    Given a 1D tensor of token ids, break it into overlapping windows.
    Each window is a dict containing:
      - "window": the slice of tokens,
      - "trg_len": number of new tokens (i.e. tokens not seen in the previous window),
      - "window_length": the length of the window.
    """
    windows = []
    L = input_ids.size(0)
    prev_end_loc = 0
    for begin in range(0, L, stride):
        end = min(begin + max_length, L)
        trg_len = end - prev_end_loc  # tokens not covered in the previous iteration
        window = input_ids[begin:end]
        windows.append({"window": window, "trg_len": trg_len, "window_length": end - begin})
        prev_end_loc = end
        if end == L:
            break
    return windows


def group_texts_by_windows(tokenized_texts, max_length, stride):
    """
    For each tokenized text, compute its sliding windows and group texts by the number of windows.
    Returns:
      - grouped: a dict mapping number-of-windows to a list of tuples (doc_id, windows)
      - texts_windows: a list of tuples (doc_id, windows) in original order.
    """
    texts_windows = []
    for idx, input_ids in enumerate(tokenized_texts):
        windows = get_windows(input_ids, max_length, stride)
        texts_windows.append((idx, windows))
    grouped = defaultdict(list)
    for doc_id, windows in texts_windows:
        grouped[len(windows)].append((doc_id, windows))
    return grouped, texts_windows


def compute_per_document_perplexities(grouped, model, tokenizer, device, batch_size, progress_bar):
    """
    Process groups of documents (each with the same number of sliding-window iterations)
    in batches, computing the cumulative negative log likelihood and token counts for each document.
    Returns a dictionary mapping document ID to its perplexity.
    """
    doc_nll = {}
    doc_tokens = {}
    # Initialize all document ids found in the groups.
    for group in grouped.values():
        for doc_id, _ in group:
            doc_nll[doc_id] = 0.0
            doc_tokens[doc_id] = 0

    for num_iters, group in maybe_tqdm(grouped.items(), progress_bar=progress_bar, desc="Groups"):
        for batch_start in range(0, len(group), batch_size):
            batch = group[batch_start: batch_start + batch_size]
            for iter_idx in range(num_iters):
                # Extract document ids and window information for the current iteration.
                batch_doc_ids = [item[0] for item in batch]
                windows_batch = [item[1][iter_idx] for item in batch]
                # Get raw token tensors for each window.
                window_tensors = [w["window"] for w in windows_batch]
                # Pad to the maximum length in this batch.
                padded_inputs = pad_sequence(window_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
                target_ids = padded_inputs.clone()
                # Mask tokens not part of the current sliding-window chunk.
                for i, w in enumerate(windows_batch):
                    valid_length = w["window_length"]
                    trg_len = w["trg_len"]
                    valid_start = valid_length - trg_len  # only the last trg_len tokens are new
                    target_ids[i, :valid_start] = -100
                    if valid_length < padded_inputs.size(1):
                        target_ids[i, valid_length:] = -100

                padded_inputs = padded_inputs.to(device)
                target_ids = target_ids.to(device)
                outputs = model(padded_inputs, labels=None)
                logits = outputs.logits
                # Shift logits and targets for computing cross-entropy loss.
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_targets = target_ids[:, 1:].contiguous()
                loss_tensor = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.size(-1)),
                    shifted_targets.view(-1),
                    reduction='none'
                )
                loss_tensor = loss_tensor.view(shifted_targets.size())
                # For each example in the batch, sum the loss over valid tokens.
                for i in range(loss_tensor.size(0)):
                    sample_losses = loss_tensor[i]
                    sample_targets = shifted_targets[i]
                    valid_mask = sample_targets != -100
                    token_loss_sum = sample_losses[valid_mask].sum().item()
                    num_valid_tokens = valid_mask.sum().item()
                    doc_id = batch_doc_ids[i]
                    doc_nll[doc_id] += token_loss_sum
                    doc_tokens[doc_id] += num_valid_tokens

    # Compute perplexity for each document.
    doc_perplexities = {}
    for doc_id in doc_nll:
        if doc_tokens[doc_id] > 0:
            avg_nll = doc_nll[doc_id] / doc_tokens[doc_id]
            perplexity = torch.exp(torch.tensor(avg_nll)).item()
            doc_perplexities[doc_id] = perplexity
        else:
            doc_perplexities[doc_id] = float('inf')
    return doc_perplexities


def calculate_perplexities(documents, model, tokenizer, device, batch_size=8, stride=512, progress_bar=True):
    """
    Given a list of document strings, compute and return a list of perplexities,
    one per document (in the same order as the input list).
    """
    tokenized_texts = tokenize_texts(documents, tokenizer)
    max_length = model.config.n_positions
    grouped, texts_windows = group_texts_by_windows(tokenized_texts, max_length, stride)
    doc_perplexities = compute_per_document_perplexities(grouped, model, tokenizer, device, batch_size, progress_bar)
    # Return perplexities in original document order.
    ordered = [doc_perplexities[i] for i in range(len(texts_windows))]
    return ordered


# -------------------------------
# Perplexity Class
# -------------------------------

class Perplexity(ReferenceFreeMetric):
    """
    Perplexity metric computed using a sliding-window approach.
    """
    def __init__(self, model="gpt2-large", batch_size=8, stride=512, progress_bar=True):
        name = "Perplexity_" + model
        description = (
            "Perplexity is a measure of how well a probability distribution predicts a sample. "
            "In the context of language models, it quantifies how well the model predicts a sequence of words. "
            "Lower perplexity indicates better performance."
        )
        super().__init__(name, description)
        self.model_name = model
        self.batch_size = batch_size
        self.stride = stride
        self.progress_bar = progress_bar

        device, _, _ = get_backend()
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the perplexity for a single document.
        Assumes `input` is a string.
        """
            
        perplexities = calculate_perplexities(
            [output],
            self.model,
            self.tokenizer,
            self.device,
            batch_size=self.batch_size,
            stride=self.stride,
            progress_bar=False
        )
        return perplexities[0]

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate perplexities for a batch of documents.
        Assumes `inputs` is a list of strings.
        """

        return calculate_perplexities(
            outputs,
            self.model,
            self.tokenizer,
            self.device,
            batch_size=self.batch_size,
            stride=self.stride,
            progress_bar=self.progress_bar
        )
    
# -------------------------------
# Main Execution for Benchmarking
# -------------------------------
if __name__ == "__main__":
    # Load the wikitext dataset and join the texts with "\n\n"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    joined_text = "\n\n".join(dataset["text"])

    # Create the Perplexity metric instance
    metric = Perplexity(model="gpt2-large", batch_size=8, stride=512, progress_bar=True)

    # Compute perplexity using the calculate method on the joined text
    ppl = metric.calculate(None, joined_text)
    print(f"Perplexity: {ppl}")
    print("Expected perplexity (from Hugging Face report): ~16.44")

    # RESULTS:
    # Perplexity: 16.44437599182129
    # Expected perplexity (from Hugging Face report): ~16.44