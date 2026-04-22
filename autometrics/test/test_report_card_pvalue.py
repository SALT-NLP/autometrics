"""
Tests for the p-value surfacing and banner logic in the HTML report card.

These tests exercise the two pieces added in response to the ICLR review:
  1. compute_correlation() now returns Pearson/Kendall p-values alongside
     effect sizes for both per-feature and regression rows.
  2. render_html() renders a tiered warning banner driven by the regression
     Pearson p-value: yellow caution for 0.05 < p ≤ 0.30, red alarm for
     p > 0.30, and nothing for p ≤ 0.05. The scatter's stats footer always
     shows the p-value alongside r and τ regardless of tier.

We stub out the full pipeline and call report_card helpers directly with
hand-built dataframes so these tests run in milliseconds and need no LLM.
"""

import numpy as np
import pandas as pd
import pytest

from autometrics.dataset.Dataset import Dataset
from autometrics.util.report_card import compute_correlation, render_html


def _make_dataset(rows: int, seed: int, noise: float = 0.0):
    """Build a tiny Dataset with a regression column whose correlation with
    the target is controlled via ``noise``. ``noise=0`` → perfect correlation,
    ``noise`` large → essentially random.
    """
    rng = np.random.default_rng(seed)
    target = rng.uniform(0.0, 5.0, size=rows)
    if noise == 0.0:
        predicted = target.copy()
    else:
        predicted = target + rng.normal(0.0, noise, size=rows)
    df = pd.DataFrame({
        "id": [str(i) for i in range(rows)],
        "input": [f"in {i}" for i in range(rows)],
        "output": [f"out {i}" for i in range(rows)],
        "human_score": target,
        "Autometrics_Regression_human_score": predicted,
    })
    return Dataset(
        dataframe=df,
        target_columns=["human_score"],
        ignore_columns=["id"],
        metric_columns=["Autometrics_Regression_human_score"],
        name="PvalueTest",
        data_id_column="id",
        input_column="input",
        output_column="output",
        reference_columns=[],
        task_description="synthetic",
    )


def _render(correlation, used_train_as_eval: bool = False) -> str:
    """Render a minimal HTML page with just the parts the banner cares about."""
    return render_html({
        "coefficients": [],
        "correlation": correlation,
        "robustness": {"available": False},
        "runtime": {},
        "details": {},
        "requirements": [],
        "examples_html": "",
        "summary": "",
        "target_measure": "human_score",
        "metrics_for_docs": [],
        "python_code": "",
        "python_filename": "Agg.py",
        "used_train_as_eval": used_train_as_eval,
    })


# ---------------------------------------------------------------------------
# compute_correlation() now returns p-values
# ---------------------------------------------------------------------------

def test_compute_correlation_returns_pvalues_for_regression_row():
    ds = _make_dataset(rows=40, seed=0, noise=0.3)
    corr = compute_correlation(
        ds, feature_names=[], target_measure="human_score",
        include_regression=True,
        regression_col_name="Autometrics_Regression_human_score",
    )
    reg = corr["regression"]
    assert "r_pvalue" in reg and "tau_pvalue" in reg
    assert isinstance(reg["r_pvalue"], float)
    assert isinstance(reg["tau_pvalue"], float)
    # Strong correlation with n=40 → very small p
    assert reg["r_pvalue"] < 0.05
    assert reg["tau_pvalue"] < 0.05


def test_compute_correlation_returns_pvalues_for_feature_rows():
    ds = _make_dataset(rows=40, seed=1, noise=0.3)
    # Reuse the regression column as a "feature" so we know it's populated.
    corr = compute_correlation(
        ds,
        feature_names=["Autometrics_Regression_human_score"],
        target_measure="human_score",
        include_regression=False,
    )
    assert corr["metrics"]
    row = corr["metrics"][0]
    assert "r_pvalue" in row and "tau_pvalue" in row


# ---------------------------------------------------------------------------
# Banner tiers
# ---------------------------------------------------------------------------

_ALARM_HEADLINE = "AutoMetrics did not find useful metrics for this task"
_CAUTION_HEADLINE = "Check these metrics carefully before trusting them"


def test_no_banner_when_pvalue_is_below_point05():
    ds = _make_dataset(rows=40, seed=2, noise=0.3)
    corr = compute_correlation(
        ds, feature_names=[], target_measure="human_score",
        include_regression=True,
        regression_col_name="Autometrics_Regression_human_score",
    )
    assert corr["regression"]["r_pvalue"] < 0.05, "sanity: pvalue band"
    html = _render(corr)
    assert _ALARM_HEADLINE not in html
    assert _CAUTION_HEADLINE not in html


def test_caution_banner_when_pvalue_between_point05_and_point3():
    # Hand-craft a correlation dict so the test doesn't depend on finding
    # real data that lands in the narrow (0.05, 0.30] band.
    corr = {
        "metrics": [],
        "regression": {
            "name": "Autometrics_Regression_human_score",
            "r": 0.45, "tau": 0.30,
            "r_pvalue": 0.12, "tau_pvalue": 0.15,
            "x": [1.0, 2.0, 3.0], "x_norm": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0], "ids": ["1", "2", "3"],
            "y_min": 0.0, "y_max": 5.0,
        },
    }
    html = _render(corr)
    assert _CAUTION_HEADLINE in html
    assert _ALARM_HEADLINE not in html
    # The numeric p shows up in the banner text
    assert "0.12" in html


def test_alarm_banner_when_pvalue_above_point3():
    corr = {
        "metrics": [],
        "regression": {
            "name": "Autometrics_Regression_human_score",
            "r": 0.05, "tau": 0.02,
            "r_pvalue": 0.72, "tau_pvalue": 0.80,
            "x": [1.0, 2.0, 3.0], "x_norm": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0], "ids": ["1", "2", "3"],
            "y_min": 0.0, "y_max": 5.0,
        },
    }
    html = _render(corr)
    assert _ALARM_HEADLINE in html
    assert _CAUTION_HEADLINE not in html
    assert "0.72" in html


def test_train_as_eval_and_alarm_banners_coexist():
    corr = {
        "metrics": [],
        "regression": {
            "name": "Autometrics_Regression_human_score",
            "r": 0.05, "tau": 0.02,
            "r_pvalue": 0.72, "tau_pvalue": 0.80,
            "x": [1.0, 2.0], "x_norm": [1.0, 2.0],
            "y": [1.0, 2.0], "ids": ["1", "2"],
            "y_min": 0.0, "y_max": 5.0,
        },
    }
    html = _render(corr, used_train_as_eval=True)
    assert "these numbers are too good to be true" in html.lower()
    assert _ALARM_HEADLINE in html
    # Order: train-as-eval banner appears before the p-value alarm.
    assert html.lower().index("these numbers are too good to be true") < html.index(_ALARM_HEADLINE)


# ---------------------------------------------------------------------------
# Stats footer
# ---------------------------------------------------------------------------

def test_stats_footer_always_shows_pvalue():
    corr = {
        "metrics": [],
        "regression": {
            "name": "Autometrics_Regression_human_score",
            "r": 0.85, "tau": 0.70,
            "r_pvalue": 0.001, "tau_pvalue": 0.002,
            "x": [1.0, 2.0], "x_norm": [1.0, 2.0],
            "y": [1.0, 2.0], "ids": ["1", "2"],
            "y_min": 0.0, "y_max": 5.0,
        },
    }
    html = _render(corr)
    # The JS that builds the footer text references both p-value fields.
    assert "r_pvalue" in html
    assert "tau_pvalue" in html
    # The footer string template contains '(p=' — rendered at runtime by JS,
    # but the literal belongs in the page source.
    assert "(p=" in html
