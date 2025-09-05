## AutoFeedback: surfacing metric reasoning alongside scores

### Goal
Allow every metric to return not just a numeric score but also free-form feedback (e.g., Chain-of-Thought rationales from LLM-judge metrics). Public flows will request and propagate feedback by default (metrics without reasoning return an empty string). Preserve backward compatibility for existing metrics and code paths; enable aggregators (e.g., Regression) to summarize and importance-rank feedback.

---

### A) Inventory of affected APIs and call sites

Core definitions that define/shape outputs today:

- `autometrics/metrics/Metric.py`
  - `def _calculate_impl(self, input, output, references=None, **kwargs)`
  - `def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs)`
  - `def calculate(self, input, output, references=None, **kwargs)`
  - `def calculate_batched(self, inputs, outputs, references=None, **kwargs)`
  - Caching in both calculate paths currently assumes scalar or list-of-scalar results.

- `autometrics/metrics/MultiMetric.py`
  - Batched flow returns `List[List[float]]` (shape `[submetric][example]`).

Representative implementations overriding `_calculate_impl` / `_calculate_batched_impl`:

- Reference-free: `INFORMRewardModel.py`, `LDLRewardModel.py`, `GRMRewardModel.py`, `SelfBLEU.py`, `Sentiment.py`, `FKGL.py`, `DistinctNGram.py`, `Perplexity.py`, `SummaQA.py`, `FactCC.py`, `HuggingFaceReferenceFreeMetric.py`, `UniEvalFact.py`, etc.
- Reference-based: `BLEU.py`, `ROUGE.py`, `METEOR.py`, `BARTScore.py`, `BERTScore.py`, `CharCut.py`, `MAUVE.py`, `YiSi.py`, `MOVERScore.py`, `InfoLM.py`, `ParaScore.py`, `UpdateROUGE.py`, `UniEvalSum.py`, `UniEvalDialogue.py`, etc.
- LLM-judge: `LLMJudge.py`, `LLMJudgeGEval.py`, `llm_judge/LLMJudgeRubric*.py`, generated LLM-judge variants (e.g., `GeneratedLLMJudgeMetric.py`, `GeneratedPrometheus.py`, `GeneratedOptimizedJudge.py`, `GeneratedExampleRubric.py`, `GeneratedFinetunedMetric.py`, `GeneratedCodeMetric.py`, `GeneratedRefBasedMetric.py`, `GeneratedRefFreeMetric.py`).
- Pairwise: `PairwiseMetric.py`, `PairwiseMultiMetric.py`.

Call sites using `calculate`:

- Utility and examples: `autometrics/util/report_card.py`, many tests under `autometrics/test/**`, examples, and experiment scripts.
- Metric wrappers: `ReferenceBasedMetric.py`, `ReferenceFreeMetric.py`, `ReferenceFreeMultiMetric.py`, `ReferenceBasedMultiMetric.py`, `MultiMetric.py`.
- Generator: `generator/CodeGenerator.py`.
- Experiments: `experiments/utilization/**`, `experiments/timing/timing.py`.

Call sites using `calculate_batched`:

- `MultiMetric.py` (core), reference-based and free variants listed above, tests, examples, robustness experiments.

Dataset/aggregation paths that indirectly depend on the shape of results:

- `Aggregator` base and `Regression` family do not directly call `calculate`, but rely on `Dataset` columns populated via `Metric.predict(...)` or `evaluate_metric_instances(...)`.
- `metric_eval_utils.evaluate_metric_instances`: invokes `metric.predict(dataset, update_dataset=True)`, where `predict` for single-/multi-metric writes numeric columns onto the `Dataset` DataFrame. No current pathway for feedback storage.

---

### B) Three design alternatives (with recommendation)

1) Extend return values in place (tuple or dataclass)
   - Change `calculate` to return `Union[float, MetricResult]` where `MetricResult = { score: float, feedback: Optional[str] }` (prefer a typed dataclass/TypedDict).
   - `calculate_batched` returns `List[Union[float, MetricResult]]` (single-metric) and in `MultiMetric` returns `List[List[Union[float, MetricResult]]]`.
   - Back-compat: call sites that expect `float` can unwrap via helper `get_score(result)` or continue working if we default to returning `float` for metrics without feedback.
   - Pros: Simple mental model; feedback travels with the score through caches and batch logic; minimal parallel APIs.
   - Cons: All metric subclass implementations and caches must adapt to support tuple/object results; more churn in tests; MultiMetric nested shapes become more complex.

2) Side-channel feedback collector
   - Keep `calculate` numeric. Add optional callback/hook interface on `Metric` like `on_feedback(example_id, text)` or a thread-local feedback buffer that metrics write to during computation; callers can retrieve aggregated feedback per example afterward.
   - Pros: Minimal signature changes; avoids cache-key/value changes; least breakage to consumers.
   - Cons: Harder to reason about lifecycle; coupling to external state/thread-locals; brittle under parallelism; persistence to `Dataset` non-trivial.

3) Dual-mode result adapter layer
   - Introduce new methods: `calculate_with_feedback` and `calculate_batched_with_feedback` returning `MetricResult`/lists, implemented by LLM-judge metrics; default implementation wraps legacy `calculate` and inserts empty feedback. Aggregators and advanced flows opt in to the richer API; legacy code remains untouched for now.
   - Pros: Highly incremental; no immediate breakage; targeted adoption by components that need feedback (e.g., Regression aggregator, report cards).
   - Cons: Two parallel APIs may drift; some duplication; eventual migration still needed to unify.

Recommendation: Start with (3) Dual-mode adapter to de-risk. Once stable and adopted in core flows (dataset population, report cards, aggregators), migrate toward (1) unified return types, retiring legacy methods behind deprecation warnings. Avoid (2) due to complexity with concurrency and state.

---

### C) Execution plan

Phase 0 — Types and shims
- Add `MetricResult` type (dataclass or TypedDict): `{ score: float, feedback: str }` with `feedback=""` by default.
- Add `has_feedback: ClassVar[bool] = False` on `Metric` (and inherited by all metrics). Implementations that truly provide feedback (LLM-judge, generated*) must set `has_feedback = True`.
- Add default implementations to `Metric`:
  - `calculate_with_feedback(self, input, output, references=None, **kwargs) -> MetricResult` (wraps `calculate` and returns empty feedback).
  - `calculate_batched_with_feedback(self, inputs, outputs, references=None, **kwargs) -> List[MetricResult]` (wraps legacy batched results with empty feedbacks).
- For `MultiMetric`, mirror: return `List[List[MetricResult]]` keeping `[submetric][example]` layout.
- Add helpers: `get_score(x)`, `get_feedback(x)` tolerant to legacy floats.

Phase 1 — Producer implementation
- Update LLM-judge metrics (DSPy Chain-of-Thought) to populate `feedback` in `_calculate_impl` and `_calculate_batched_impl` but surfaced via the new `*_with_feedback` methods. Keep legacy `calculate` behavior unchanged.
- Ensure caching keys include inputs but cache values remain numeric for legacy `calculate`. The new `*_with_feedback` path MUST be cached as well (storing `MetricResult`), using method-name disambiguation in keys to avoid collisions with legacy caches.

Feedback-first implementation pattern (LLM-judge and generated metrics)
- Introduce `_calculate_with_feedback_impl(self, input, output, references=None, **kwargs) -> MetricResult` in metrics where feedback is first-class.
- Rewrite the existing `_calculate_impl` to simply call `_calculate_with_feedback_impl` and return only `.score` (drop `.feedback`).
- Similarly for batches: `_calculate_batched_with_feedback_impl(self, inputs, outputs, references=None, **kwargs) -> List[MetricResult]` and redefine `_calculate_batched_impl` to call the with-feedback variant and strip to scores.
- Rationale: these metrics become source-of-truth for feedback; legacy paths remain thin wrappers so older callers continue to see floats without change.
- Set `has_feedback: ClassVar[bool] = True` on LLM-judge and generated metrics so downstream layers know to persist feedback columns.

Phase 2 — Dataset integration (non-breaking)
- Extend `Metric.predict` to accept a flag `with_feedback: bool = True`.
  - If `False`: no behavior change.
  - If `True` (default): call the `*_with_feedback` variants; write score to the usual column; conditionally store feedback to `Dataset` if and only if `getattr(self, 'has_feedback', False)` is True. Use a parallel column convention `{metric_name}__feedback` (or a nested store like `dataset.feedback[metric_name] = List[str]`).
- Update `metric_eval_utils.evaluate_metric_instances` to pass through `with_feedback` and to collect/store feedback columns when `True`.
  - Do not allocate feedback columns for metrics where `has_feedback` is False or absent, even if `with_feedback=True`.

Phase 3 — Aggregator support
- Extend `Aggregator.ensure_dependencies`/`Regression` to request feedback by default (`with_feedback=True`) for input metrics.
- Do not change `Regression.identify_important_metrics` (it remains numeric-only and unchanged). Aggregation of feedback will occur in `Regression.predict` (or a helper), using the learned coefficients to rank which metric feedback to surface, without modifying the identify method itself.
  - For each example, gather feedback strings from all metrics that have feedback (ordered by |coefficient|; optionally include sign labels). Allow an optional user-specified limit (top-k) to truncate, but default is to include all feedback.
  - Produce an aggregated rationale per example: concise, weight-tagged bullet lines sorted by importance; deduplicate near-identical feedback.
- Expose aggregated feedback via `Regression.predict(..., return_feedback: bool=True)` or by storing in `Dataset` as `{aggregator_name}__feedback`.

Phase 4 — Reporting and UX
- Update `util/report_card.py` to render feedback by default, showing only the aggregator’s concatenated rationale by default. Provide an option to include individual metric feedback (off by default to reduce clutter).
- Provide toggles for verbosity and maximum characters.

Phase 5 — Migration to unified API (optional, later)
- Deprecate `calculate`/`calculate_batched` in favor of always returning `MetricResult`/lists. Provide deprecation warnings and a codemod.
- Collapse `*_with_feedback` once downstream callers all accept `MetricResult`.

---

### Testing strategy
- Unit tests for `MetricResult` helpers and the new `*_with_feedback` methods on base `Metric` and `MultiMetric`.
- Tests for LLM-judge metrics confirming non-empty `feedback` and consistent `score` with legacy path.
- Dataset integration tests: ensure both score and feedback columns are written; caching still hits on score path.
 - Dataset integration tests: ensure both score and feedback columns are written; caching hits on both legacy and with-feedback paths.
- Aggregator tests: verify concatenation ordering matches coefficient importances; validate output stability across seeds.
- Backward-compat tests: legacy callers receive identical numeric outputs and unchanged DataFrame schemas when not opting into feedback.

---

### D) Feedback acquisition from DSPy Chain-of-Thought

THE FUNDAMENTAL IDEA IS TO KEEP IT SIMPLE. We will use only the `.reasoning` attribute.
- For DSPy `ChainOfThought`, extract feedback from the `dspy.Prediction.reasoning` field.
  - Example:
    ```python
    import dspy
    program = dspy.ChainOfThought('question -> answer')
    pred = program(question='Why is the sky blue?')
    feedback = getattr(pred, 'reasoning', '')  # primary CoT text
    # score = ...  # metric-specific numeric computation
    # return MetricResult(score=score, feedback=feedback)
    ```
- Privacy and size controls: implement max-length truncation with ellipsis, and allow a redaction toggle if needed.

### Open questions / decisions
- Column naming for feedback: `__feedback` suffix vs. structured dataset field.
- Storage size and truncation policy for long reasoning strings (defaults and overrides).
- Caching policy details for feedback paths: value type (`MetricResult`), TTL/eviction, and ensuring consistent hashing across with/without feedback (method name included in key prevents collisions).
- Pairwise metrics: default to per-comparison feedback. If an implementation evaluates pointwise sub-calls for `output_1` and `output_2`, optionally store side-specific feedback as `{metric_name}__feedback_1` and `{metric_name}__feedback_2`. Low priority initially; implement when needed.

---

### Work items (tracked)
1. Types and shims in `Metric`/`MultiMetric` (with-feedback variants, helpers).
2. Update LLM-judge producers to emit feedback.
3. Dataset predict/evaluation opt-in flag and feedback storage.
4. Regression aggregator: importance-weighted feedback concatenation.
5. Report card rendering toggles.
6. Tests for all above; ensure zero regressions on legacy flows.


### E) Detailed migration plan and callsite changes

0) Types and helpers
- Add `MetricResult` (e.g., dataclass) in `autometrics/metrics/Metric.py` with fields: `score: float`, `feedback: str` (default "").
- Add helpers in `Metric.py` (module-level): `get_score(x) -> float`, `get_feedback(x) -> str`. These accept either `float` or `MetricResult` and return appropriately.

1) Core `Metric` base (`autometrics/metrics/Metric.py`)
- Add public methods:
  - `calculate_with_feedback(self, input, output, references=None, **kwargs) -> MetricResult`.
  - `calculate_batched_with_feedback(self, inputs, outputs, references=None, **kwargs) -> List[MetricResult]`.
- Default implementations wrap legacy `calculate` / `calculate_batched` and attach empty feedback strings.
- Caching:
  - Use distinct method names in cache keys: `'calculate_with_feedback'` vs `'calculate'` to avoid collisions.
  - Caching logic will need to be similar to the caching logic for `calculate` / `calculate_batched`
  - Store `MetricResult` instances (or JSON-serializable dict) for with-feedback paths; leave legacy unchanged.
- Predict integration:
  - `predict(..., with_feedback: bool = True)`; when `True`, call with-feedback paths and write both score column and `{metric_name}__feedback` column to the dataset.

2) `MultiMetric` (`autometrics/metrics/MultiMetric.py`)
- Add:
  - `calculate_batched_with_feedback(...) -> List[List[MetricResult]]` with shape `[submetric][example]`.
  - Default path: wrap numeric results in `MetricResult(score=v, feedback="")`.
- `predict(..., with_feedback: bool = True)` writes each submetric score as before; when `with_feedback`, also writes `{submetric_name}__feedback` columns.
- Only add `{submetric_name}__feedback` columns if the has_feedback classvar is present and set to true.

3) Dataset evaluation utilities (`autometrics/util/metric_eval_utils.py`)
- Thread `with_feedback=True` through all `metric.predict` calls (sequential and parallel waves).
- In the parallel path, when merging results back into the main dataset, also collect any `{metric_name}__feedback` or `{submetric_name}__feedback` columns and write them.  Only for metrics with the classvar has_feedback set to True.
- Ensure unload/GC behavior unchanged.

4) Aggregators
- `autometrics/aggregator/Aggregator.py`:
  - In `ensure_dependencies`, when computing missing metrics, use `metric.predict(dataset, update_dataset=True, with_feedback=True)` so feedback columns are materialized by default for metrics which have the class var has_feedback set to True.
- `autometrics/aggregator/regression/Regression.py`:
  - Do not change `identify_important_metrics`.
  - In `predict`, after obtaining numeric prediction, assemble aggregator-level feedback by ranking component metric feedback by |coefficient| and concatenating all lines by default. Support an optional limit parameter to cap to top-k if specified (default no limit). Store to `{aggregator_name}__feedback` and/or return via `return_feedback=True`.

5) Report card (`autometrics/util/report_card.py`)
- Default to showing only the aggregator feedback column(s). Provide a flag to include per-metric feedback columns.
- Callsite adjustments:
  - Where the report card currently calls `m.calculate(...)` for per-row previews, switch to `m.calculate_with_feedback(...)` when feedback display is enabled; otherwise leave intact.
  - For batch previews using `m.calculate_batched(...)`, switch to `m.calculate_batched_with_feedback(...)` when feedback is desired, but only display aggregator feedback by default.
 - Only display or allocate metric feedback columns when the metric’s `has_feedback` is True.

6) LLM-judge and generated metrics (feedback-first pattern)
- Files to update to the `_with_feedback_impl` pattern (prioritized):
  - `autometrics/metrics/llm_judge/LLMJudge.py`
  - `autometrics/metrics/llm_judge/LLMJudgeGEval.py`
  - `autometrics/metrics/llm_judge/LLMJudgeRubric.py`
  - `autometrics/metrics/llm_judge/LLMJudgeRubricDSPy.py`
  - Generated families under `autometrics/metrics/generated/`: `GeneratedLLMJudgeMetric.py`, `GeneratedPrometheus.py`, `GeneratedOptimizedJudge.py`, `GeneratedExampleRubric.py`, `GeneratedFinetunedMetric.py`, `GeneratedCodeMetric.py`, `GeneratedRefBasedMetric.py`, `GeneratedRefFreeMetric.py`.
- For each:
  - Implement `_calculate_with_feedback_impl(...) -> MetricResult` extracting DSPy reasoning via `.reasoning`. Ensure return of both score and feedback.
  - `_calculate_impl(...)` calls the above and returns `.score` only.
  - Batched: add `_calculate_batched_with_feedback_impl(...) -> List[MetricResult]` and have `_calculate_batched_impl(...)` derive only scores.
  - Set `has_feedback = True` on these classes.

7) Wrapper metrics
- `autometrics/metrics/reference_based/ReferenceBasedMetric.py` and `autometrics/metrics/reference_free/ReferenceFreeMetric.py`:
  - Keep existing `result = self.calculate(...)` behavior for numeric-only flows.
  - Optionally add parallel `result = self.calculate_with_feedback(...)` paths for callers that opt in; not required for backward compatibility.
  - Leave `has_feedback` as default False so the dataset will not create empty feedback columns.

8) Pairwise metrics
- Minimal initial support: return per-comparison feedback string when with-feedback paths are used. Defer side-specific feedback until needed; when implemented, use `{metric_name}__feedback_1` and `{metric_name}__feedback_2`.
- Files:
  - `autometrics/metrics/PairwiseMetric.py`
  - `autometrics/metrics/PairwiseMultiMetric.py`

9) Scripts/examples/generator
- `generator/CodeGenerator.py`: leave numeric behavior; optionally add a flag to request feedback and log it when helpful.
- Examples under `autometrics/examples/**`: add a new example demonstrating feedback flows and report-card rendering.

10) Tests
- Extend tests to cover:
  - Base helpers: `MetricResult`, `get_score`, `get_feedback`.
  - With-feedback methods on `Metric`/`MultiMetric` (shape and caching behavior).
  - LLM-judge metrics produce non-empty feedback and identical scores vs legacy.
  - Dataset integration writes `{name}__feedback` columns by default; lengths match dataset.
  - No feedback columns are created for metrics with `has_feedback=False`.
  - Aggregator concatenation ordering matches coefficient magnitudes; output column `{aggregator}__feedback` present.
  - Report card shows aggregator feedback by default; option toggles per-metric feedback.
  - Pairwise minimal behavior sanity.
- Keep all legacy tests that assert numeric outputs via `calculate`/`calculate_batched` unchanged.

11) Caching details
- Cache keys include method name so legacy and with-feedback entries are separate.
- Cache values for with-feedback paths store `MetricResult`(s) (or dict). Ensure serialization is supported by disk cache.

12) Rollout order (PRs)
- PR1: Types, base methods, caching, dataset/report-card toggles behind default-True flags; no metric-specific changes.
- PR2: LLM-judge family: implement with-feedback impl pattern; tests.
- PR3: Aggregator feedback concatenation and tests.
- PR4: Optional: pairwise enhancements; new examples; broader docs.

Acceptance criteria checklist
- With-feedback enabled by default, dataset gains feedback columns for metrics that produce them; scores unchanged.
 - No empty feedback columns are created for metrics without feedback; `has_feedback` gates persistence and display.
- Caching reduces recomputation for both numeric-only and with-feedback paths.
- Regression aggregator produces and exposes an aggregated feedback column.
- Report card defaults to aggregator feedback only; can optionally include per-metric feedback.
- All existing tests pass; new tests validate feedback behavior and caching.

