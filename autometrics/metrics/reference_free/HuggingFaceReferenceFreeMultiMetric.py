from evaluate import load
from autometrics.metrics.reference_free.ReferenceFreeMultiMetric import ReferenceFreeMultiMetric
from typing import List

class HuggingFaceReferenceFreeMultiMetric(ReferenceFreeMultiMetric):
    """
    Generic wrapper for HuggingFace Evaluate reference-free multi-metrics.
    Allows loading any evaluate metric that returns multiple keys.
    """
    def __init__(
        self,
        name: str,
        description: str,
        metric_id: str,
        submetric_keys: List[str],
        load_kwargs: dict = None,
        compute_kwargs: dict = None
    ):
        super().__init__(name, description, submetric_names=submetric_keys)
        self.metric_id = metric_id
        self.submetric_keys = submetric_keys
        self.load_kwargs = load_kwargs or {}
        self.compute_kwargs = compute_kwargs or {}
        self.metric = None

    def _load_metric(self):
        if self.metric is None:
            self.metric = load(self.metric_id, **self.load_kwargs)

    def calculate(self, input_text: str, output: str, references=None, **kwargs):
        """
        Compute submetrics for a single output by calling the underlying HF evaluate metric.
        """
        self._load_metric()
        # Prepare single-item lists for compute
        preds = [output]
        args = {'predictions': preds}
        if references is not None:
            # Expect references as a list of references for this item
            refs = references if isinstance(references, list) and not isinstance(references[0], str) else [references]
            args['references'] = [refs]
        # Include any fixed compute kwargs
        args.update(self.compute_kwargs)
        result = self.metric.compute(**args)
        # Extract each submetric and return tuple
        values = []
        for key in self.submetric_keys:
            val = result.get(key)
            # Unpack single-element lists or return scalar
            if isinstance(val, (list, tuple)):
                values.append(float(val[0]))
            else:
                values.append(float(val))
        return tuple(values)

    def calculate_batched(self, inputs: List[str], outputs: List[str], references=None, **kwargs):
        """
        Batch evaluation: loops over each output and calls calculate(), assembling submetric lists.
        """
        # Prepare references list aligning with outputs
        refs_list = references if references is not None else [None] * len(outputs)
        # Initialize output lists for each submetric
        sub_results = [[] for _ in self.submetric_keys]
        # Evaluate each sample
        for out, ref in zip(outputs, refs_list):
            vals = self.calculate(None, out, ref, **kwargs)
            for idx, v in enumerate(vals):
                sub_results[idx].append(v)
        return sub_results 