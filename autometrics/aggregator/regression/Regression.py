import numpy as np
from autometrics.aggregator.Aggregator import Aggregator
from sklearn.preprocessing import StandardScaler

class Regression(Aggregator):
    """
    Class for regression aggregation
    """
    def __init__(self, name, description, input_metrics=None, model=None, dataset=None, **kwargs):
        super().__init__(name, description, input_metrics, dataset, **kwargs)
        self.model = model
        self.scaler = None  # Will store the StandardScaler for consistent scaling
        self._selected_columns = None  # Persist exact feature order used during training

    def learn(self, dataset, target_column=None):
        """
        Learn the regression model with proper feature scaling
        """
        self.ensure_dependencies(dataset)
        df = dataset.get_dataframe()

        input_columns = self.get_input_columns()
        if not target_column:
            target_column = dataset.get_target_columns()[0]

        # Pull out X and y
        X = df[input_columns]
        y = df[target_column]

        # —— clip any +/-inf in X to the finite min/max of each column
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        mins = X_clean.min()
        maxs = X_clean.max()
        X = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        # —— same for y (a Series)
        y_clean = y.replace([np.inf, -np.inf], np.nan)
        if y_clean.isna().all():
            # if everything was infinite, just zero out
            y = y.fillna(0)
        else:
            y = y.clip(lower=y_clean.min(), upper=y_clean.max()).fillna(0)

        # Apply StandardScaler to handle scale differences between metrics
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Now safe to fit on scaled data
        self.model.fit(X_scaled, y)

        # Persist exact feature order for downstream export/repro
        self._selected_columns = list(input_columns)

    def _predict_unsafe(self, dataset, update_dataset=True):
        """
        Predict the target column using the same scaling as training
        """
        df = dataset.get_dataframe().copy()
        input_columns = self.get_input_columns()

        # Ensure dependencies are computed (some callers may bypass predict())
        # Note: not calling self.ensure_dependencies here to avoid double-work,
        # but we will fail fast if columns are still missing.
        missing_inputs = [c for c in input_columns if c not in df.columns]
        if missing_inputs:
            # Try once to compute via ensure_dependencies in case predict() caller forgot
            self.ensure_dependencies(dataset)
            df = dataset.get_dataframe().copy()
            missing_inputs = [c for c in input_columns if c not in df.columns]
        if missing_inputs:
            raise KeyError(f"Regression input columns missing from dataset: {missing_inputs}")

        X = df[input_columns]

        # —— clip any +/-inf or too large values in X before predict
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        mins = X_clean.min()
        maxs = X_clean.max()
        X = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        # Apply the same scaling used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            # Fallback if no scaler (for backward compatibility)
            X_scaled = X

        y_pred = self.model.predict(X_scaled)

        if update_dataset:
            df.loc[:, self.name] = y_pred
            dataset.set_dataframe(df)
            # Keep dataset.metric_columns metadata in sync so downstream correlation sees this column
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)

        return y_pred
    
    def identify_important_metrics(self):
        '''
            Identify the most important metrics depending on the model.
            For linear models: Use standardized coefficients (already in standardized form when fit on scaled data).
            For tree-based models: Use feature importances.
        '''
        metric_columns = self.get_input_columns()

        # Linear models (Ridge, Lasso, ElasticNet, PLS)
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim == 1:
                pairs = zip(coef, metric_columns)
            else:
                pairs = zip(coef[0], metric_columns)
            return sorted(pairs, key=lambda x: abs(x[0]), reverse=True)

        # Tree-based models (RandomForest, GradientBoosting)
        if hasattr(self.model, 'feature_importances_'):
            pairs = zip(self.model.feature_importances_, metric_columns)
            return sorted(pairs, key=lambda x: abs(x[0]), reverse=True)

        raise ValueError(
            "The model does not support extracting feature importances or coefficients."
        )

    # --- Export helpers -------------------------------------------------
    def get_selected_columns(self):
        """
        Return the exact feature column order used during training if available;
        otherwise fall back to current input columns.
        """
        return list(self._selected_columns) if self._selected_columns else self.get_input_columns()

    def _extract_static_params(self):
        """
        Extract coefficients, intercept, and StandardScaler statistics in a
        consistent shape. Coefficients are 1-D length n_features.
        """
        if self.scaler is None:
            raise ValueError("Scaler is not fitted; run learn() before exporting.")
        if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
            raise ValueError("Scaler is missing mean_/scale_ attributes.")

        # Coefficients/intercept from sklearn-like models
        coef = getattr(self.model, 'coef_', None)
        intercept = getattr(self.model, 'intercept_', 0.0)
        if coef is None:
            raise ValueError("Model is missing coef_.")
        # Flatten shapes (n_features,) or (1, n_features)
        try:
            import numpy as _np
            coef_arr = _np.array(coef, dtype=float).reshape(-1)
        except Exception:
            coef_arr = list(coef)[0] if isinstance(coef, (list, tuple)) and len(coef) == 1 else list(coef)
        # Intercept may be array-like
        try:
            import numpy as _np
            if isinstance(intercept, (list, tuple)):
                intercept_val = float(intercept[0]) if len(intercept) > 0 else 0.0
            else:
                intercept_val = float(_np.array(intercept).reshape(-1)[0])
        except Exception:
            try:
                intercept_val = float(intercept)
            except Exception:
                intercept_val = 0.0

        mean = getattr(self.scaler, 'mean_', None)
        scale = getattr(self.scaler, 'scale_', None)
        if mean is None or scale is None:
            raise ValueError("Scaler statistics are not available (mean_/scale_).")
        return coef_arr, intercept_val, list(mean), list(scale)

    def _generate_python_code(self, inline_generated_metrics: bool = False, name_salt: str | None = None) -> str:
        """
        Generate a standalone Python module that rebuilds this regression as a
        static aggregator using stored coefficients and scaler statistics.
        """
        # Gather static params
        coef_arr, intercept_val, mean_list, scale_list = self._extract_static_params()
        feature_names = self.get_selected_columns()

        # Build import lines and constructor code for input metrics
        from autometrics.aggregator.generated.GeneratedRegressionMetric import generate_metric_constructor_code
        import re as _re

        import_lines = []
        inline_blocks = []
        ctor_exprs = []
        seen_imports = set()

        def _safe_class_name_from_block(code_block: str, fallback: str) -> str:
            m = _re.search(r"class\s+([A-Za-z_]\w*)\(", code_block)
            return m.group(1) if m else fallback

        def _uniquify_constants(code_block: str, class_name: str) -> str:
            # Avoid top-level constant collisions across concatenated blocks
            replacements = [
                (r"\bDEFAULT_MODEL\b", f"DEFAULT_MODEL_{class_name}"),
                (r"\bOPTIMIZED_EXAMPLES\b", f"OPTIMIZED_EXAMPLES_{class_name}"),
                (r"\bOPTIMIZED_PROMPT_DATA\b", f"OPTIMIZED_PROMPT_DATA_{class_name}"),
            ]
            out = code_block
            for pattern, repl in replacements:
                out = _re.sub(pattern, repl, out)
            return out

        for m in (self.input_metrics or []):
            if inline_generated_metrics and hasattr(m, '_generate_python_code'):
                try:
                    raw = m._generate_python_code(include_metric_card=False)
                    # Derive class name from block; fallback to sanitized metric name
                    fallback = str(getattr(m, 'name', m.__class__.__name__)).replace(' ', '_').replace('-', '_')
                    cls_name = _safe_class_name_from_block(raw, fallback)
                    # Uniquify top-level constants inside the block
                    block = _uniquify_constants(raw, cls_name)
                    inline_blocks.append(block)
                    ctor_exprs.append(f"{cls_name}()")
                    continue
                except Exception:
                    # Fallback to import path
                    pass
            imp, ctor = generate_metric_constructor_code(m)
            if imp not in seen_imports:
                import_lines.append(imp)
                seen_imports.add(imp)
            ctor_exprs.append(ctor)

        import_block = "\n".join(import_lines)
        ctor_list = ",\n        ".join(ctor_exprs) if ctor_exprs else ""
        inline_block = ("\n\n".join(inline_blocks) + ("\n\n" if inline_blocks else ""))

        # Build code string
        # Use a single underscore when salting to avoid double underscores in class names
        salted_name = self.name if not name_salt else f"{self.name}_{name_salt}"
        class_base_unsalted = self.name
        def _sanitize(s: str) -> str:
            return s.replace(' ', '_').replace('-', '_')
        class_def_name = f"{_sanitize(class_base_unsalted)}_StaticRegression"
        code = f"""# Auto-generated static regression for {self.name}
from typing import ClassVar
import numpy as np
from autometrics.aggregator.generated.GeneratedRegressionMetric import GeneratedStaticRegressionAggregator

{import_block}

{inline_block}

INPUT_METRICS = [
        {ctor_list}
]

class {class_def_name}(GeneratedStaticRegressionAggregator):
    \"\"\"Static regression aggregator generated from a trained Regression.\"\"\"

    description: ClassVar[str] = {repr(self.description)}

    def __init__(self):
        super().__init__(
            name={repr(salted_name)},
            description={repr(self.description)},
            input_metrics=INPUT_METRICS,
            feature_names={repr(list(feature_names))},
            coefficients={repr([float(x) for x in list(coef_arr)])},
            intercept={float(intercept_val)},
            scaler_mean={repr([float(x) for x in list(mean_list)])},
            scaler_scale={repr([float(x) for x in list(scale_list)])},
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"
"""
        return code

    def export_python(self, output_path: str, *, inline_generated_metrics: bool = True, name_salt: str | None = None) -> str:
        """Write generated code to output_path and return the path.

        If name_salt is provided, the exported static metric's internal name
        will be suffixed with the salt to avoid collisions with any existing
        AutoMetrics caches. If None, no salt is applied.
        """
        code = self._generate_python_code(inline_generated_metrics=inline_generated_metrics, name_salt=name_salt)
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return output_path
