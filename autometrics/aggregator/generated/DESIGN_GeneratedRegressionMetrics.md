## Generated Regression Aggregators – Design Spec

Goal

- Provide a robust, importable, static representation of a trained regression aggregator that can be exported to a Python file and re-used without scikit-learn at runtime.
- Ensure out-of-the-box imports for submetrics and stable reproduction of row-wise predictions on any dataset with the same feature columns.

Key Concepts

- Static Aggregator Class: `GeneratedStaticRegressionAggregator` implements the prediction math using stored coefficients, intercept, and StandardScaler statistics. It does not depend on scikit-learn, making exported files minimal and portable.
- Export from Trained Regression: The existing `Regression` class learns with scikit-learn and StandardScaler. After training, it can emit a standalone Python module that reconstitutes the behavior with frozen parameters and imports for the submetrics.

Placement

- Core static class and utilities live in `autometrics/aggregator/generated/GeneratedRegressionMetric.py`.
- This spec sits alongside it for discoverability.

Compatibility and Behavior

- Matches Regression.predict logic:
  - Ensures submetric dependencies are computed (`Aggregator.ensure_dependencies`).
  - Clips +/-inf per column to the finite min/max; then fills NaNs with 0.
  - Applies the same StandardScaler transform (using learned mean/scale) to features in the same column order used during training.
  - Computes `y = X_scaled @ coef + intercept`.
- Works with any `Regression` subclass (PLS, Lasso, Ridge, ElasticNet, Linear, etc.) after learning, as long as the underlying model exposes `coef_` and `intercept_` in scikit-learn style.

Feature Column Order

- During `Regression.learn`, we persist the exact `input_columns` used for training as `_selected_columns`.
- The exporter uses this order to serialize `feature_names`. The static aggregator uses the same order at inference time.

Submetrics and Imports

- Each input metric instance is converted to an import line and a zero-argument constructor expression using `generate_metric_constructor_code`. This ensures the exported module includes proper imports and can create fresh metric instances (e.g., `BLEU()`, `SARI()`, `ROUGE()`).
- For `MultiMetric` inputs, no special-case is needed at export time; the aggregator depends only on the column names present in the dataset and the order in `feature_names`.

Exported Module Structure

- Imports: core class `GeneratedStaticRegressionAggregator` and each referenced submetric class.
- `INPUT_METRICS`: a list of instantiated submetrics constructed with zero-arg constructors.
- A concrete class named after the trained regression (e.g., `MyPLS_StaticRegression`) that passes:
  - `name`, `description`
  - `input_metrics`: `INPUT_METRICS`
  - `feature_names`, `coefficients`, `intercept`, `scaler_mean`, `scaler_scale`

Serialization Format Choices

- All numeric arrays are serialized as Python lists of floats for simplicity and to avoid numpy dependency in the exported file.
- Strings use `repr` to preserve exact content.

Cross-Validation of Assumptions

- Coefficients API: We use `model.coef_` and `model.intercept_` (see `util/report_card.extract_regression_coefficients`). Shapes are normalized to 1-D `n_features`.
- Standardization: We reuse the clipping/fillna logic in `Regression._predict_unsafe` and the StandardScaler statistics from the trained instance.

Future Extensions

- Metric Cards: A minimal card could be produced similarly to generated metrics, listing component metrics and coefficients. The exported class can be extended to include docstrings and a small `_generate_metric_card` stub if desired later.
- Nonlinear Models: For tree-based or nonlinear regressors, provide an abstract export interface; this design targets linear-like models where `coef_` exists.

Summary of Changes

- Added `GeneratedStaticRegressionAggregator` with pure-numpy inference.
- Augmented `Regression` with:
  - `_selected_columns` persistence during `learn`
  - `_extract_static_params`, `_generate_python_code`, and `export_python` helpers
- Added utility `generate_metric_constructor_code` for reliable imports.


