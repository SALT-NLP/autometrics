import os
import re
import fasttext
from platformdirs import user_data_dir
from huggingface_hub import hf_hub_download
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
from typing import List, Union

class FastTextEducationalValue(ReferenceFreeMetric):
    """
    Educational value scoring via fastText.  Downloads a multi-class model and computes an expected score
    based on probabilities over labels __label__Low (0), __label__Mid (1), __label__High (2).
    """
    def __init__(
        self,
        name: str = "FastTextEducationalValue",
        description: str = "fastText classifier for educational value scoring (Low/Mid/High)",
        repo_id: str = "kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2",
        filename: str = "fasttext_educational_value.bin",
        persistent: bool = True,
        data_dir: str = None
    ):
        super().__init__(name, description)
        self.repo_id = repo_id
        self.filename = filename
        self.persistent = persistent
        base_dir = data_dir or user_data_dir("autometrics")
        os.makedirs(base_dir, exist_ok=True)
        self.cache_dir = base_dir
        self.model = None
        if self.persistent:
            self._load_model()

    def _load_model(self):
        # Download via HF if not cached
        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            cache_dir=self.cache_dir
        )
        self.model = fasttext.load_model(model_path)

    def _unload_model(self):
        self.model = None

    def calculate(
        self,
        input_text: str,
        output: str,
        references: Union[List[str], str] = None,
        **kwargs
    ) -> float:
        # Lazy load
        if self.model is None:
            self._load_model()
        # Clean newlines
        text = re.sub(r"\n+", " ", output)
        # Predict over all labels
        labels_list, probs_list = self.model.predict([text], k=-1)
        labels = labels_list[0]
        probs = probs_list[0]
        # Mapping to numeric scores
        score_map = {
            '__label__': 0,
            '__label__Low': 0,
            '__label__Mid': 1,
            '__label__High': 2,
        }
        score = 0.0
        for l, p in zip(labels, probs):
            score += score_map.get(l, 0) * p
        # Optionally unload
        if not self.persistent:
            self._unload_model()
        return float(score)

    def calculate_batched(
        self,
        inputs: List[str],
        outputs: List[str],
        references=None,
        **kwargs
    ) -> List[float]:
        # Lazy load
        if self.model is None:
            self._load_model()
        # Clean each output
        cleaned = [re.sub(r"\n+", " ", o) for o in outputs]
        # Predict in batch
        labels_list, probs_list = self.model.predict(cleaned, k=-1)
        score_map = {
            '__label__': 0,
            '__label__Low': 0,
            '__label__Mid': 1,
            '__label__High': 2,
        }
        scores: List[float] = []
        for labels, probs in zip(labels_list, probs_list):
            s = 0.0
            for l, p in zip(labels, probs):
                s += score_map.get(l, 0) * p
            scores.append(float(s))
        # Optionally unload
        if not self.persistent:
            self._unload_model()
        return scores 