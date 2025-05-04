from evaluate import load
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class HuggingFaceReferenceBasedMetric(ReferenceBasedMetric):
    """
    Generic wrapper for HuggingFace Evaluate reference-based metrics.
    Loads the metric via evaluate.load and extracts a specified score key.
    """
    def __init__(
        self,
        name: str,
        description: str,
        metric_id: str,
        score_key: str = "score",
        load_kwargs: dict = None
    ):
        super().__init__(name, description)
        self.metric_id = metric_id
        self.score_key = score_key
        self.load_kwargs = load_kwargs or {}
        self.metric = None

    def _load_metric(self):
        if self.metric is None:
            self.metric = load(self.metric_id, **self.load_kwargs)

    def calculate(self, input: str, output: str, references=None, **kwargs):
        self._load_metric()
        # Expect references as list of strings
        refs = references if isinstance(references[0], list) or isinstance(references[0], tuple) else references
        result = self.metric.compute(predictions=[output], references=[refs], **kwargs)
        val = result.get(self.score_key)
        # broadcast scalar to single entry
        return float(val) if not isinstance(val, (list, tuple)) else float(val[0])

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Batch evaluation: loops over each sample and calls calculate().
        """
        # Prepare references list
        refs = references if references is not None else [None] * len(outputs)
        scores = []
        for inp, out, ref in zip(inputs, outputs, refs):
            scores.append(self.calculate(inp, out, ref, **kwargs))
        return scores 