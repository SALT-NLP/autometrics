### Metric Report Card: Design Spec and Execution Plan

#### Goals
- Generate a rich HTML report card for a trained regression metric and its input metrics.
- If an evaluation dataset is provided, include all sections; otherwise include a subset.

#### Inputs
- regression_metric: Trained regression aggregator instance (e.g., PLS/HotellingPLS) with fitted model and accessible `input_metrics`.
- metrics: List of metric instances used by the regression (regular and MultiMetric).
- target_measure: String name of the ground-truth target column.
- eval_dataset (optional): Dataset for evaluation-only sections.
- train_dataset (optional): Dataset used during regression (optional reference in the card).
- lm (optional): dspy.LM for robustness and summary generation.
- output_path (optional): Path to save the rendered HTML report.

#### Outputs
- Path to saved HTML file and/or the HTML string.

#### Sections
1) Regression Coefficients
   - Extract linear coefficients from `regression_metric.model` if available (e.g., `coef_`, `intercept_`).
   - Map coefficients to feature names:
     - If `regression_metric` exposes `get_selected_columns()`, use that order.
     - Else derive from dataset columns corresponding to `input_metrics` (expand `MultiMetric` submetric names when present).
   - Fallback to importance/weights when coefficients not available.

2) Correlation (eval-only)
   - For each metric column and the regression score, compute Pearson r (and optionally Kendall's tau) with the ground truth on `eval_dataset`.
   - Produce scatter data arrays and a small results list for display.

3) Robustness (eval-only)
   - Prefer running `RobustnessExperiment` from `autometrics.experiments.robustness.robustness` with provided metrics ("obvious" mode by default).
   - Compute per-metric Stability and Sensitivity from the returned full table: 
     - Use `group` in {original, worse_obvious, same_obvious}.
     - sensitivity := max(0, (mean(original) - mean(worse_obvious)) / max(|mean(original)|, eps)).
     - stability := max(0, 1 - |mean(original) - mean(same_obvious)| / max(|mean(original)|, eps)).
   - Include the regression metric if feasible (compute column on eval copy first).

4) Run Time Distribution (eval-only)
   - Subsample 30 rows (or entire eval dataset if < 30).
   - Run all metrics sequentially on that subsample, timing per-example execution per metric.
   - Drop the shortest and longest two timings (if n >= 6), and create a box-and-whisker dataset per metric.
   - Compute aggregate time per example:
     - sequence = sum of per-metric times
     - parallel = max of per-metric times
   - Separately, compute all metric values on full eval set using parallel strategy (reuse Autometrics evaluator when possible) for correlation and examples sections.

5) Metric Details
   - Parse metric class docstrings formatted as metric cards. Extract:
     - Description
     - Usage/Tasks
     - Limitations
   - Fallback to classvar `description` and minimal bullets if parsing fails.

6) Compute Requirements
   - Read ClassVar `gpu_mem` and `cpu_mem` if present; otherwise "--".

7) Metric Summary (LLM)
   - Build a DSPy signature that accepts: Task description, dataset sample, target column, and a list of metrics with name, coefficient, description, usage, limitations.
   - Produce a concise 5–10 sentence summary.

8) Examples
   - Show a 5-row sample table with columns: Input, Output, References (optional), Ground Truth, all metric columns, and the Regression Score. Make it easily extensible for a future AutoFeedback column.

#### Module API (autometrics/util/report_card.py)
- generate_metric_report_card(
  regression_metric,
  metrics,
  target_measure: str,
  eval_dataset: Optional[Dataset] = None,
  train_dataset: Optional[Dataset] = None,
  lm: Optional[dspy.LM] = None,
  output_path: Optional[str] = None,
) -> dict
  - Returns: { 'html': str, 'path': Optional[str], 'artifacts': dict }

Internal helpers (separation of concerns):
- extract_feature_names(regression_metric, metrics, dataset) -> List[str]
- extract_regression_coefficients(regression_metric, feature_names) -> List[(name, coeff)]
- ensure_eval_metrics(eval_dataset, metrics, parallel=True) -> Dataset
- compute_correlation(eval_dataset, feature_names, target_measure, include_regression=True) -> dict
- run_robustness(eval_dataset, metrics, regression_metric, lm) -> dict(name -> {stability, sensitivity})
- measure_runtime(eval_sample_dataset, metrics) -> dict with per-metric times and aggregate stats
- parse_metric_cards(metric_classes) -> dict(name -> {description, usage, limitations})
- compute_requirements(metric_classes) -> List[{name, gpu_mem, cpu_mem}]
- build_examples_table(eval_dataset, feature_names, target_measure, include_regression=True) -> str (HTML)
- render_html(context: dict) -> str

#### Integration
- Update `Autometrics.run(...)` to accept `eval_dataset: Optional[Dataset] = None`.
- After regression, call `generate_metric_report_card(...)` and attach HTML string to pipeline return ('report_card'). Optionally save to `artifacts/`.

#### Testing Plan
1) Minimal test script: SimpDA + BLEU/ROUGE/SARI, train PLS, generate report (no generated metrics).
2) Integration test: Full pipeline with BLEU/ROUGE/SARI + 2 llm_judge generated metrics, produce report.

Notes
- Be robust to missing attributes and heterogeneous metrics (MultiMetric vs regular).
- Avoid heavy dependencies; render with vanilla HTML + Plotly.
- Keep functions small and unit-testable.


