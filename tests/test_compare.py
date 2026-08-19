"""Tests for the deployment-selection rule.

This rule decides which model treats patients. It previously ranked candidates
by inference latency alone and selected the worst-recall architecture - 17
missed abnormalities against the best model's 6 - to save 360 ms per scan.
"""

from __future__ import annotations

import pandas as pd
import pytest

from neuroscan.evaluation.compare import (
    RECALL_TOLERANCE,
    SELECTION_RECALL_FLOOR,
    select_best_architecture,
)


def make_table(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "accuracy": 0.95, "precision": 0.95, "recall": 0.95, "specificity": 0.95,
        "f1": 0.95, "auc_roc": 0.99, "params_millions": 5.0,
        "cpu_latency_ms": 100.0, "train_minutes": 5.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class TestSelection:
    def test_prefers_recall_over_speed(self, cfg):
        """The regression this rule exists to prevent."""
        table = make_table([
            {"architecture": "fast_but_misses", "recall": 0.9264,
             "accuracy": 0.9632, "cpu_latency_ms": 47.0},
            {"architecture": "slower_but_catches", "recall": 0.9740,
             "accuracy": 0.9762, "cpu_latency_ms": 409.0},
        ])
        selection = select_best_architecture(table, cfg)
        assert selection["selected"] == "slower_but_catches"

    def test_breaks_near_ties_on_efficiency(self, cfg):
        """Within the tolerance, the cheaper model is the right call for a
        CPU-only district hospital."""
        table = make_table([
            {"architecture": "cheap", "recall": 0.960, "cpu_latency_ms": 40.0},
            {"architecture": "expensive", "recall": 0.965, "cpu_latency_ms": 400.0},
        ])
        selection = select_best_architecture(table, cfg)
        assert selection["selected"] == "cheap"
        assert selection["recall_traded"] <= RECALL_TOLERANCE

    def test_does_not_trade_beyond_the_tolerance(self, cfg):
        table = make_table([
            {"architecture": "cheap", "recall": 0.92, "cpu_latency_ms": 40.0},
            {"architecture": "expensive", "recall": 0.99, "cpu_latency_ms": 400.0},
        ])
        assert select_best_architecture(table, cfg)["selected"] == "expensive"

    def test_flags_when_nothing_is_eligible(self, cfg):
        table = make_table([
            {"architecture": "weak", "recall": 0.60, "accuracy": 0.70},
            {"architecture": "weaker", "recall": 0.50, "accuracy": 0.65},
        ])
        selection = select_best_architecture(table, cfg)
        assert selection["constraint_met"] is False
        # Still names the best available, and says so.
        assert selection["selected"] == "weak"
        assert "No architecture met" in selection["reason"]

    def test_excludes_models_below_the_recall_floor(self, cfg):
        table = make_table([
            {"architecture": "high_accuracy_low_recall",
             "recall": SELECTION_RECALL_FLOOR - 0.05, "accuracy": 0.99,
             "cpu_latency_ms": 10.0},
            {"architecture": "balanced", "recall": 0.95, "accuracy": 0.95,
             "cpu_latency_ms": 300.0},
        ])
        assert select_best_architecture(table, cfg)["selected"] == "balanced"

    def test_ties_break_on_auc_before_latency(self, cfg):
        """When two models tie at the operating point, the one with better
        threshold-independent separability wins - it has more margin and should
        degrade more gracefully under distribution shift. Latency is measured on
        a shared machine and moves 4x with unrelated load, so it must not
        outrank a stable statistic."""
        table = make_table([
            {"architecture": "tied_lower_auc", "recall": 0.9827, "accuracy": 0.9870,
             "auc_roc": 0.9969, "cpu_latency_ms": 166.0},
            {"architecture": "tied_higher_auc", "recall": 0.9827, "accuracy": 0.9870,
             "auc_roc": 0.9994, "cpu_latency_ms": 191.0},
        ])
        assert select_best_architecture(table, cfg)["selected"] == "tied_higher_auc"

    def test_latency_still_breaks_an_auc_tie(self, cfg):
        table = make_table([
            {"architecture": "slow", "recall": 0.98, "auc_roc": 0.999, "cpu_latency_ms": 400.0},
            {"architecture": "fast", "recall": 0.98, "auc_roc": 0.999, "cpu_latency_ms": 40.0},
        ])
        assert select_best_architecture(table, cfg)["selected"] == "fast"

    def test_auc_never_outranks_recall(self, cfg):
        """Separability is a tiebreak, not a substitute for catching tumours."""
        table = make_table([
            {"architecture": "great_auc_poor_recall", "recall": 0.91,
             "auc_roc": 0.9999, "cpu_latency_ms": 40.0},
            {"architecture": "good_recall", "recall": 0.98,
             "auc_roc": 0.9900, "cpu_latency_ms": 400.0},
        ])
        assert select_best_architecture(table, cfg)["selected"] == "good_recall"

    def test_empty_table(self, cfg):
        assert select_best_architecture(pd.DataFrame(), cfg)["selected"] is None

    def test_reason_quantifies_the_decision(self, cfg):
        """A selection a supervisor cannot interrogate is not auditable."""
        table = make_table([{"architecture": "only", "recall": 0.96}])
        reason = select_best_architecture(table, cfg)["reason"]
        assert "%" in reason
        assert "recall" in reason.lower()


class TestLatencyMeasurement:
    @pytest.mark.slow
    def test_measures_cpu_latency(self, cfg):
        from neuroscan.evaluation.compare import measure_cpu_latency
        from neuroscan.models.factory import build_model

        model = build_model(cfg, architecture="baseline_cnn")
        timing = measure_cpu_latency(model, cfg.preprocessing.image_size, n_runs=3, n_warmup=1)
        assert timing["median_ms"] > 0
        assert timing["min_ms"] <= timing["median_ms"] <= timing["max_ms"]
