"""
Tests for the generation/regression dataset split surfaced in Autometrics.run().

Added in response to the ICLR reviewer ask to let callers use different data
for criteria proposal (Step 1) vs. regression fitting (Steps 4–5).

These tests stub every LLM-touching and heavy-computation method on the
orchestrator so they run in milliseconds and need no API key. We're checking
pure dispatch: which dataset each stage receives, and how on-disk identity
is chosen when the splits have different names.
"""

from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import pandas as pd
import pytest

from autometrics.autometrics import Autometrics
from autometrics.dataset.Dataset import Dataset


def _ds(name: str, rows: int, seed: int = 0) -> Dataset:
    """Build a minimal Dataset with ``rows`` rows and a unique name."""
    df = pd.DataFrame({
        "id": [f"{name}-{i}" for i in range(rows)],
        "input": [f"prompt {i}" for i in range(rows)],
        "output": [f"response {i}" for i in range(rows)],
        "score": [float((i + seed) % 5) + 1.0 for i in range(rows)],
    })
    return Dataset(
        dataframe=df,
        target_columns=["score"],
        ignore_columns=["id"],
        metric_columns=[],
        name=name,
        data_id_column="id",
        input_column="input",
        output_column="output",
        reference_columns=[],
        task_description=f"{name} task",
    )


class _Recorder:
    """Monkeypatches the three dataset-receiving pipeline stages to record
    which Dataset object each stage was handed.
    """
    def __init__(self, monkeypatch, am: Autometrics):
        self.gen_dataset = None
        self.gen_save_name = None
        self.eval_dataset = None
        self.reg_dataset = None
        self.report_train = None

        def fake_generate_or_load(_self, ds, target_measure, *a, save_name_dataset=None, **kw):
            self.gen_dataset = ds
            self.gen_save_name = save_name_dataset
            return []  # no generated metrics → empty pipeline downstream

        def fake_eval(_self, ds, metric_classes, **kw):
            self.eval_dataset = ds
            return []

        def fake_regress(_self, ds, *a, **kw):
            self.reg_dataset = ds
            return {
                "top_metrics": [],
                "regression_metric": SimpleNamespace(
                    get_name=lambda: "fake_regression",
                    get_description=lambda: "fake",
                    predict=lambda _d: [],
                ),
                "importance_scores": [],
            }

        def fake_report_card(_self, *a, **kw):
            return ""

        def fake_html_report(**kw):
            self.report_train = kw.get("train_dataset")
            return {"html": "", "path": None, "artifacts": {}}

        monkeypatch.setattr(Autometrics, "_generate_or_load_metrics", fake_generate_or_load)
        monkeypatch.setattr(Autometrics, "_process_metric_priors",
                            lambda _s, ds, *a, save_name_dataset=None, **kw: [])
        monkeypatch.setattr(Autometrics, "_evaluate_metrics_on_dataset", fake_eval)
        monkeypatch.setattr(Autometrics, "_regress_and_select_top_n", fake_regress)
        monkeypatch.setattr(Autometrics, "_generate_report_card", fake_report_card)
        # HTML report generator is a module-level function looked up inside run().
        monkeypatch.setattr(
            "autometrics.util.report_card.generate_metric_report_card",
            fake_html_report,
        )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def _make_am() -> Autometrics:
    """Autometrics instance configured so the generated-only short-circuit
    fires on any small regression dataset — this skips retriever setup so
    these tests don't need Java/pyserini/pylate."""
    return Autometrics(metric_generation_configs={}, full_bank_data_cutoff=100)


def test_single_dataset_routes_to_all_stages(monkeypatch):
    """No split kwargs → every stage sees the same dataset. The run-id name
    used for on-disk artifacts is also the original dataset's name."""
    am = _make_am()
    rec = _Recorder(monkeypatch, am)
    ds = _ds("OnlyOne", rows=8)

    am.run(
        dataset=ds, target_measure="score",
        generator_llm=None, judge_llm=None,
        num_to_retrieve=0, num_to_regress=1,
    )

    assert rec.gen_dataset is ds
    assert rec.gen_save_name is ds
    assert rec.eval_dataset is ds
    assert rec.reg_dataset is ds
    assert rec.report_train is ds


def test_split_routes_each_stage_to_its_split(monkeypatch):
    """With both split kwargs set, generation and regression stages see
    different datasets."""
    am = _make_am()
    rec = _Recorder(monkeypatch, am)
    gen = _ds("GenSplit", rows=12, seed=1)
    reg = _ds("RegSplit", rows=8, seed=2)

    am.run(
        dataset=gen, target_measure="score",
        generator_llm=None, judge_llm=None,
        generation_dataset=gen,
        regression_dataset=reg,
        num_to_retrieve=0, num_to_regress=1,
    )

    # Generation stage (Step 1) must see the generation split
    assert rec.gen_dataset is gen
    # On-disk naming must follow the regression split so every run artifact
    # shares an identity.
    assert rec.gen_save_name is reg
    # Evaluation (Step 4), regression fit (Step 5), and the HTML report's
    # "train" dataset all key off the regression split.
    assert rec.eval_dataset is reg
    assert rec.reg_dataset is reg
    assert rec.report_train is reg


def test_only_regression_dataset_override(monkeypatch):
    """Passing only regression_dataset: generation falls back to dataset,
    regression goes to the explicit override."""
    am = _make_am()
    rec = _Recorder(monkeypatch, am)
    main = _ds("Main", rows=8)
    reg = _ds("RegOverride", rows=20)

    am.run(
        dataset=main, target_measure="score",
        generator_llm=None, judge_llm=None,
        regression_dataset=reg,
        num_to_retrieve=0, num_to_regress=1,
    )

    assert rec.gen_dataset is main
    assert rec.gen_save_name is reg
    assert rec.reg_dataset is reg


def test_only_generation_dataset_override(monkeypatch):
    """Passing only generation_dataset: regression falls back to dataset."""
    am = _make_am()
    rec = _Recorder(monkeypatch, am)
    main = _ds("Main", rows=8)
    gen = _ds("GenOverride", rows=20)

    am.run(
        dataset=main, target_measure="score",
        generator_llm=None, judge_llm=None,
        generation_dataset=gen,
        num_to_retrieve=0, num_to_regress=1,
    )

    assert rec.gen_dataset is gen
    assert rec.gen_save_name is main
    assert rec.reg_dataset is main


# ---------------------------------------------------------------------------
# Cutoff is evaluated against the regression split
# ---------------------------------------------------------------------------

def test_cutoff_fires_on_small_regression_dataset_even_if_generation_is_large(monkeypatch, capsys):
    """Cutoff triggers generated-only mode based on regression size (the
    labeled-data size that matters for fitting), not generation size."""
    am = Autometrics(metric_generation_configs={}, full_bank_data_cutoff=100)
    _Recorder(monkeypatch, am)

    am.run(
        dataset=_ds("x", 500),
        generation_dataset=_ds("BigGen", rows=500),
        regression_dataset=_ds("SmallReg", rows=8),
        target_measure="score",
        generator_llm=None, judge_llm=None,
        num_to_retrieve=0, num_to_regress=1,
    )

    out = capsys.readouterr().out
    assert "Regression dataset size (8) <= cutoff (100)" in out
    assert "using generated metrics only" in out


def test_cutoff_does_not_fire_when_regression_is_large(monkeypatch, capsys):
    """The reverse: a big regression dataset keeps the full pipeline even
    if the generation dataset is small. We stub the bank-loading and retriever
    construction so this test doesn't require Java/pylate."""
    am = Autometrics(metric_generation_configs={}, full_bank_data_cutoff=100)
    _Recorder(monkeypatch, am)
    # Keep the pipeline out of the retrieval code path entirely.
    monkeypatch.setattr(Autometrics, "_load_metric_bank", lambda _s, _ds: [])
    monkeypatch.setattr(
        Autometrics, "_validate_and_adjust_retriever_config",
        lambda _self, kw, *a, **kwargs: kw,
    )
    am.retriever = lambda **kwargs: SimpleNamespace(recommend=lambda **kw: [])

    am.run(
        dataset=_ds("x", 8),
        generation_dataset=_ds("SmallGen", rows=8),
        regression_dataset=_ds("BigReg", rows=500),
        target_measure="score",
        generator_llm=None, judge_llm=None,
        num_to_retrieve=0, num_to_regress=1,
    )

    out = capsys.readouterr().out
    assert "using generated metrics only" not in out


# ---------------------------------------------------------------------------
# Results dict exposes both splits
# ---------------------------------------------------------------------------

def test_results_dict_exposes_both_splits(monkeypatch):
    am = _make_am()
    _Recorder(monkeypatch, am)
    gen = _ds("Gen", rows=8)
    reg = _ds("Reg", rows=8)

    results = am.run(
        dataset=gen, target_measure="score",
        generator_llm=None, judge_llm=None,
        generation_dataset=gen, regression_dataset=reg,
        num_to_retrieve=0, num_to_regress=1,
    )

    # `dataset` is the regression split (the canonical one for identifying the run)
    assert results["dataset"] is reg
    assert results["regression_dataset"] is reg
    assert results["generation_dataset"] is gen
