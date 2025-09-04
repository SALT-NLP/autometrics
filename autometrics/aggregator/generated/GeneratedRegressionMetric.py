import numpy as np
import json
from typing import List, Optional, Tuple

from autometrics.aggregator.Aggregator import Aggregator
from autometrics.metrics.MultiMetric import MultiMetric


class GeneratedStaticRegressionAggregator(Aggregator):
    """
    A static, serialization-friendly regression aggregator that does not rely on
    scikit-learn at runtime. It uses stored coefficients, intercept, and
    StandardScaler statistics (feature means and scales) learned previously.

    This class expects metric dependencies (input metrics) to already be known
    and importable. During prediction, it ensures those metrics are available on
    the dataset (via ensure_dependencies), then computes:

        y = (X_clipped - mean) / scale @ coef + intercept

    where X_clipped follows the same clipping and NaN handling as in
    autometrics.aggregator.regression.Regression._predict_unsafe.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_metrics: List[object],
        feature_names: List[str],
        coefficients: List[float],
        intercept: float,
        scaler_mean: List[float],
        scaler_scale: List[float],
        dataset=None,
        **kwargs,
    ):
        super().__init__(name, description, input_metrics=input_metrics, dataset=dataset, **kwargs)

        # Persisted parameters
        self._feature_names: List[str] = list(feature_names)
        self._coef: np.ndarray = np.array(coefficients, dtype=float).reshape(-1)
        self._intercept: float = float(intercept)
        self._mean: np.ndarray = np.array(scaler_mean, dtype=float).reshape(-1)
        self._scale: np.ndarray = np.array(scaler_scale, dtype=float).reshape(-1)

        # Basic validation to guard against misalignment
        n = len(self._feature_names)
        if not (self._coef.shape[0] == self._mean.shape[0] == self._scale.shape[0] == n):
            raise ValueError(
                f"Static regression parameter length mismatch: features={n}, "
                f"coef={self._coef.shape[0]}, mean={self._mean.shape[0]}, scale={self._scale.shape[0]}"
            )

    def get_selected_columns(self) -> List[str]:
        """Return the exact feature column order used during training/export."""
        return list(self._feature_names)

    def _predict_unsafe(self, dataset, update_dataset: bool = True):
        import numpy as _np
        df = dataset.get_dataframe().copy()

        # Ensure dependencies are computed
        self.ensure_dependencies(dataset)
        df = dataset.get_dataframe().copy()

        input_columns = self.get_selected_columns()
        missing = [c for c in input_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Static regression input columns missing from dataset: {missing}")

        X = df[input_columns]

        # Clip +/-inf per-column to finite min/max, then fill NaN with 0 (matches Regression)
        X_clean = X.replace([_np.inf, -_np.inf], _np.nan)
        mins = X_clean.min()
        maxs = X_clean.max()
        X = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        # Standardize using stored statistics
        X_arr = X.values.astype(float)
        # Avoid division by zero: if scale == 0, treat as 1 (no scaling)
        safe_scale = _np.where(self._scale == 0, 1.0, self._scale)
        X_scaled = (X_arr - self._mean) / safe_scale

        y_pred = X_scaled.dot(self._coef) + self._intercept

        if update_dataset:
            df.loc[:, self.name] = y_pred
            dataset.set_dataframe(df)
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)

        return y_pred

    def identify_important_metrics(self) -> List[Tuple[float, str]]:
        pairs = list(zip(self._coef.tolist(), self.get_selected_columns()))
        return sorted(pairs, key=lambda x: abs(x[0]), reverse=True)


def generate_metric_constructor_code(metric_instance: object) -> Tuple[str, str]:
    """
    Produce an import statement and a constructor expression for a metric instance.

    Returns (import_line, constructor_code).

    Notes:
    - Defaults to using the class' module and zero-arg constructor.
    - For MultiMetric instances, the constructor remains the same; submetric
      expansion is handled by the aggregator using get_selected_columns order.
    """
    cls = metric_instance.__class__
    module = cls.__module__
    name = cls.__name__

    # Prefer import-by-path for generated metrics saved to disk by the pipeline
    python_file = getattr(metric_instance, "_python_file_path", None)
    exported_cls = getattr(metric_instance, "_exported_class_name", None)
    if python_file and exported_cls:
        dyn_import = (
            f"import importlib.util as _imp_util\n"
            f"_spec = _imp_util.spec_from_file_location('_{exported_cls}_mod', {json.dumps(python_file)})\n"
            f"_{exported_cls}_mod = _imp_util.module_from_spec(_spec); _spec.loader.exec_module(_{exported_cls}_mod)\n"
            f"from _{exported_cls}_mod import {exported_cls}"
        )
        import_line = dyn_import
        constructor_code = f"{exported_cls}()"
        return import_line, constructor_code

    # Helper to robustly render dspy.LM constructor from a live model without relying on utils
    def _render_llm(model_obj) -> str:
        try:
            import json as _json
            # Extract kwargs
            kwargs = {}
            if hasattr(model_obj, 'kwargs') and isinstance(getattr(model_obj, 'kwargs', None), dict):
                kwargs = {k: v for k, v in model_obj.kwargs.items() if v is not None}
            # Determine model name
            model_name = kwargs.pop('model', None)
            if model_name is None:
                raw = getattr(model_obj, 'model', None)
                model_name = getattr(raw, 'model', raw)
            if not model_name or str(model_name).lower() == 'none':
                model_name = 'gpt-3.5-turbo'
            # api_key fallback
            mn = str(model_name).lower()
            if 'api_key' not in kwargs:
                if 'openai' in mn:
                    kwargs['api_key'] = 'os.getenv("OPENAI_API_KEY")'
                elif 'anthropic' in mn:
                    kwargs['api_key'] = 'os.getenv("ANTHROPIC_API_KEY")'
                elif 'gemini' in mn or 'google' in mn:
                    kwargs['api_key'] = 'os.getenv("GEMINI_API_KEY")'
                else:
                    kwargs['api_key'] = 'None'
            parts = []
            for k, v in kwargs.items():
                if isinstance(v, str) and k != 'api_key':
                    parts.append(f"{k}={_json.dumps(v)}")
                else:
                    parts.append(f"{k}={v}")
            kwargs_str = ", ".join(parts)
            return f"dspy.LM(model={_json.dumps(str(model_name))}, {kwargs_str})"
        except Exception:
            return "dspy.OpenAI(model='gpt-3.5-turbo')"

    # Special-case: Generated LLM Judge metrics require constructor args
    if (
        module.endswith("autometrics.metrics.generated.GeneratedLLMJudgeMetric")
        or module == "autometrics.metrics.generated.GeneratedLLMJudgeMetric"
    ) and name in ("GeneratedRefFreeLLMJudgeMetric", "GeneratedRefBasedLLMJudgeMetric"):
        # Safely read attributes with defaults
        metric_name = getattr(metric_instance, "name", name)
        description = getattr(metric_instance, "description", "")
        axis = getattr(metric_instance, "axis", "")
        task_description = getattr(metric_instance, "task_description", None)
        max_workers = getattr(metric_instance, "max_workers", 32)
        model_obj = getattr(metric_instance, "model", None)

        # We'll inline a best-effort model constructor string and include dspy import in the file
        import_line = f"import dspy\nfrom {module} import {name}"

        model_expr = (
            "generate_llm_constructor_code(metric_model_placeholder)" if model_obj is None else f"generate_llm_constructor_code({repr(model_obj)})"
        )
        # We cannot repr(model_obj) directly in code; instead, we call generate_llm_constructor_code at runtime with a variable.
        # To avoid referencing unknown names, we inline the function call with the instance's model via placeholder replacement below.
        # Since we cannot pass the live object, we will construct the call string using the function itself at import-time in the exported file.
        # Simplify: call generate_llm_constructor_code on the exporting side and inline the resulting code.
        if model_obj is not None:
            model_code = _render_llm(model_obj)
        else:
            # Fallback: attempt to fetch DEFAULT_MODEL from the originating module
            try:
                import importlib
                mod = importlib.import_module(module)
                default_model_obj = getattr(mod, "DEFAULT_MODEL", None)
                if default_model_obj is not None:
                    from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code as _gen_llm_code
                    model_code = _gen_llm_code(default_model_obj)
                else:
                    model_code = "dspy.OpenAI(model='gpt-3.5-turbo')"
            except Exception:
                model_code = "dspy.OpenAI(model='gpt-3.5-turbo')"

        constructor_code = (
            f"{name}(name={json.dumps(str(metric_name))}, "
            f"description={json.dumps(str(description))}, "
            f"axis={json.dumps(str(axis))}, "
            f"model={model_code}, "
            f"task_description={json.dumps(str(task_description))}, "
            f"max_workers={int(max_workers)})"
        )
        return import_line, constructor_code

    # Example-based judge metrics also need constructor args
    if (
        module.endswith("autometrics.metrics.generated.GeneratedExampleRubric")
        or module == "autometrics.metrics.generated.GeneratedExampleRubric"
    ) and name in ("GeneratedRefFreeExampleRubricMetric", "GeneratedRefBasedExampleRubricMetric"):
        import_line = f"import dspy\nfrom {module} import {name}"
        metric_name = getattr(metric_instance, "name", name)
        description = getattr(metric_instance, "description", "")
        axis = getattr(metric_instance, "axis", "")
        task_description = getattr(metric_instance, "task_description", None)
        suggested_range = getattr(metric_instance, "suggested_range", (1,5))
        seed = getattr(metric_instance, "seed", 42)
        max_workers = getattr(metric_instance, "max_workers", 32)
        model_obj = getattr(metric_instance, "model", None)
        if model_obj is not None:
            model_code = _render_llm(model_obj)
        else:
            # Fallback: attempt to fetch DEFAULT_MODEL from the originating module
            try:
                import importlib
                mod = importlib.import_module(module)
                default_model_obj = getattr(mod, "DEFAULT_MODEL", None)
                if default_model_obj is not None:
                    from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code as _gen_llm_code
                    model_code = _gen_llm_code(default_model_obj)
                else:
                    model_code = "dspy.OpenAI(model='gpt-3.5-turbo')"
            except Exception:
                model_code = "dspy.OpenAI(model='gpt-3.5-turbo')"
        constructor_code = (
            f"{name}(name={json.dumps(str(metric_name))}, "
            f"description={json.dumps(str(description))}, "
            f"axis={json.dumps(str(axis))}, "
            f"model={model_code}, "
            f"task_description={json.dumps(str(task_description))}, "
            f"suggested_range={repr(tuple(suggested_range))}, "
            f"seed={int(seed)}, max_workers={int(max_workers)})"
        )
        return import_line, constructor_code

    # Default path: zero-arg constructor
    import_line = f"from {module} import {name}"
    constructor_code = f"{name}()"
    return import_line, constructor_code


