from evaluate import load
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric

class HuggingFaceReferenceFreeMetric(ReferenceFreeMetric):
    """
    Generic wrapper for HuggingFace Evaluate reference-free metrics that return a single score per input.
    """
    def __init__(
        self,
        name: str,
        description: str,
        metric_id: str,
        score_key: str = None,
        load_kwargs: dict = None,
        compute_kwargs: dict = None,
        **kwargs
    ):
        # Pass ALL parameters to parent constructor
        super().__init__(
            name=name,
            description=description,
            metric_id=metric_id,
            score_key=score_key,
            load_kwargs=load_kwargs,
            compute_kwargs=compute_kwargs,
            **kwargs
        )
        self.metric_id = metric_id
        self.score_key = score_key
        self.load_kwargs = load_kwargs or {}
        self.compute_kwargs = compute_kwargs or {}
        self.metric = None

    def _load_metric(self):
        if self.metric is None:
            self.metric = load(self.metric_id, **self.load_kwargs)

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs) -> float:
        self._load_metric()
        # single prediction
        compute_args = {**self.compute_kwargs, **kwargs, 'predictions': [output]}
        result = self.metric.compute(**compute_args)
        val = result.get(self.score_key)
        # If list, take first element
        if isinstance(val, (list, tuple)):
            return float(val[0])
        return float(val)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs) -> list:
        self._load_metric()
        scores = []
        # Try vectorized compute for first two items
        if len(outputs) >= 2:
            try:
                small_preds = outputs[:2]
                small_args = {**self.compute_kwargs, **kwargs, 'predictions': small_preds}
                small_res = self.metric.compute(**small_args)
                val_small = small_res.get(self.score_key)
                if isinstance(val_small, (list, tuple)) and len(val_small) == 2:
                    # add first two
                    scores.extend([float(v) for v in val_small])
                    # compute for remainder
                    if len(outputs) > 2:
                        rest_preds = outputs[2:]
                        rest_args = {**self.compute_kwargs, **kwargs, 'predictions': rest_preds}
                        rest_res = self.metric.compute(**rest_args)
                        val_rest = rest_res.get(self.score_key)
                        if isinstance(val_rest, (list, tuple)) and len(val_rest) == len(rest_preds):
                            scores.extend([float(v) for v in val_rest])
                        else:
                            # fallback for each remaining
                            for i, out in enumerate(rest_preds):
                                # Use the parent's calculate method to leverage caching
                                scores.append(super().calculate(inputs[i+2] if i+2 < len(inputs) else None, out, None, **kwargs))
                    return scores
            except Exception:
                pass
        # Fallback to per-sample for entire batch
        scores = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            # Use the parent's calculate method to leverage caching
            scores.append(super().calculate(inp, out, None, **kwargs))
        return scores 