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
        Batch evaluation with fallback: test on first 2 outputs; if vectorized runs (returns lists of length 2), apply to full batch,
        else fallback to per-sample calculate loop.
        """
        self._load_metric()
        refs = references if references is not None else [None] * len(outputs)
        # Try vectorized compute for first two items
        sub_results = [[] for _ in self.submetric_keys]
        if len(outputs) >= 2:
            try:
                # Compute first two
                small_preds = outputs[:2]
                small_args = {**self.compute_kwargs, **kwargs, 'predictions': small_preds}
                if references is not None:
                    small_args['references'] = refs[:2]
                small_res = self.metric.compute(**small_args)
                # Validate and store first two
                ok = True
                for idx, key in enumerate(self.submetric_keys):
                    val2 = small_res.get(key)
                    if not (isinstance(val2, (list, tuple)) and len(val2) == 2):
                        ok = False
                        break
                    sub_results[idx].extend([float(v) for v in val2])
                if ok:
                    # Compute remaining items
                    if len(outputs) > 2:
                        rest_preds = outputs[2:]
                        rest_args = {**self.compute_kwargs, **kwargs, 'predictions': rest_preds}
                        if references is not None:
                            rest_args['references'] = refs[2:]
                        rest_res = self.metric.compute(**rest_args)
                        for idx, key in enumerate(self.submetric_keys):
                            val_rest = rest_res.get(key)
                            if isinstance(val_rest, (list, tuple)) and len(val_rest) == len(rest_preds):
                                sub_results[idx].extend([float(v) for v in val_rest])
                            else:
                                for out in rest_preds:
                                    sub_results[idx].append(self.calculate(None, out, None, **kwargs)[idx])
                    return sub_results
            except Exception:
                pass
        # Fallback: per-sample
        for inp, out, ref in zip(inputs, outputs, refs):
            vals = self.calculate(inp, out, ref, **kwargs)
            for idx, v in enumerate(vals):
                sub_results[idx].append(v)
        return sub_results 