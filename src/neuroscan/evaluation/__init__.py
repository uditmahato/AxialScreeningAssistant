"""Metrics, curves and cross-architecture comparison.

Project Design 4.1 commits to evaluating with "accuracy, precision, recall, f1
score, AUC-ROC, and cross validation". This package implements those, plus the
two things that make the numbers defensible on a dataset of a few hundred
images: bootstrap confidence intervals, and a threshold chosen for clinical
cost rather than left at the 0.5 default.
"""

from neuroscan.evaluation.metrics import (
    ClassificationMetrics,
    bootstrap_confidence_interval,
    compute_metrics,
    evaluate_model,
    tune_decision_threshold,
)

__all__ = [
    "ClassificationMetrics",
    "bootstrap_confidence_interval",
    "compute_metrics",
    "evaluate_model",
    "tune_decision_threshold",
]
