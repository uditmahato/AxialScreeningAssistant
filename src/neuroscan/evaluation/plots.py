"""Publication-quality figures for the final report.

Everything writes to file through the Agg backend - no interactive display -
so the same code runs identically in a notebook, from a script, and on a
headless machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)

from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from neuroscan.config import Config
    from neuroscan.evaluation.metrics import ClassificationMetrics
    from neuroscan.training.callbacks import MetricHistory
    from neuroscan.training.trainer import TrainingResult

log = get_logger("evaluation.plots")

PALETTE = {
    "baseline_cnn": "#8c8c8c",
    "vgg16": "#e07b39",
    "efficientnet_b0": "#2c7fb8",
}
DEFAULT_COLOR = "#4a4a4a"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure written to %s", path)
    return path


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    path: Path,
    *,
    title: str = "Confusion matrix (test set)",
    normalise: bool = False,
) -> Path:
    """Confusion matrix with counts and row percentages.

    Both are shown because on an imbalanced split the raw counts and the
    per-class rates tell different stories, and the clinically relevant one -
    what fraction of abnormal scans were caught - is the percentage.
    """
    cm = np.array(metrics.confusion, dtype=float)
    if cm.size == 0:
        raise ValueError("No confusion matrix available")

    row_sums = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0) * 100
    display = percentages if normalise else cm

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)

    names = metrics.class_names
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if percentages[i, j] > 55 else "black"
            label = f"{display[i, j]:.1f}%" if normalise else f"{int(cm[i, j])}\n{percentages[i, j]:.1f}%"
            ax.text(j, i, label, ha="center", va="center", color=colour, fontsize=10)

    fig.colorbar(im, ax=ax, label="% of true class")
    return _save(fig, path)


def plot_roc_curve(metrics: ClassificationMetrics, path: Path, *, title: str = "ROC curve") -> Path:
    """ROC curve with the operating threshold marked."""
    if metrics.y_true is None or metrics.y_score is None:
        raise ValueError("Metrics do not carry the raw scores needed for a ROC curve")
    if len(np.unique(metrics.y_true)) < 2:
        raise ValueError("ROC curve requires both classes to be present")

    fpr, tpr, thresholds = roc_curve(metrics.y_true, metrics.y_score[:, 1])
    area = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, color="#2c7fb8", lw=2, label=f"AUC = {area:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1, label="Chance")

    # Mark where the deployed threshold actually sits on the curve.
    index = int(np.argmin(np.abs(thresholds - metrics.threshold)))
    ax.plot(fpr[index], tpr[index], "o", color="#d62728", ms=8, zorder=5,
            label=f"Operating point (t={metrics.threshold:.2f})")

    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, path)


def plot_precision_recall_curve(
    metrics: ClassificationMetrics, path: Path, *, title: str = "Precision-Recall curve"
) -> Path:
    """Precision-recall curve.

    More informative than ROC when the positive class is rare, because ROC's
    false-positive rate is diluted by a large negative class.
    """
    if metrics.y_true is None or metrics.y_score is None:
        raise ValueError("Metrics do not carry the raw scores needed for a PR curve")

    precision, recall, _ = precision_recall_curve(metrics.y_true, metrics.y_score[:, 1])
    baseline = float(np.mean(metrics.y_true))

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(recall, precision, color="#2c7fb8", lw=2, label=f"AP = {metrics.auc_pr:.4f}")
    ax.axhline(baseline, ls="--", color="#bbbbbb", lw=1,
               label=f"Chance (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", frameon=False)
    return _save(fig, path)


def plot_training_history(history: MetricHistory, path: Path, *, title: str = "Training history") -> Path:
    """Loss and validation metric curves, with the stage boundary marked."""
    records = history.records
    if not records:
        raise ValueError("History is empty")

    epochs = [r.epoch for r in records]
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax_loss.plot(epochs, [r.train_loss for r in records], label="train", color="#e07b39", lw=1.8)
    ax_loss.plot(epochs, [r.val_loss for r in records], label="validation", color="#2c7fb8", lw=1.8)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(frameon=False)

    for key, colour, label in [
        ("val_accuracy", "#2c7fb8", "accuracy"),
        ("val_recall", "#d62728", "recall"),
        ("val_f1", "#7b3294", "F1"),
        ("val_auc", "#1a9641", "AUC"),
    ]:
        ax_metric.plot(epochs, history.series(key), label=label, color=colour, lw=1.6)
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_title("Validation metrics")
    ax_metric.set_ylim(0, 1.02)
    ax_metric.legend(frameon=False, ncol=2)

    # Where stage 1 (frozen backbone) handed over to stage 2 (fine-tuning).
    stages = [r.stage for r in records]
    if "head" in stages and "finetune" in stages:
        boundary = epochs[stages.index("finetune")] - 0.5
        for ax in (ax_loss, ax_metric):
            ax.axvline(boundary, ls=":", color="#888888", lw=1.4)
            ax.text(boundary, ax.get_ylim()[1] * 0.97, " fine-tune",
                    fontsize=8, color="#666666", va="top")

    best = history.best
    if best is not None:
        for ax in (ax_loss, ax_metric):
            ax.axvline(best.epoch, ls="--", color="#1a9641", lw=1.2, alpha=0.7)

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_comparison(results: list[TrainingResult], path: Path, cfg: Config) -> Path:
    """Overlay the test ROC curves of every architecture."""
    fig, ax = plt.subplots(figsize=(5.8, 5.4))

    plotted = 0
    for result in results:
        metrics = result.test_metrics
        if metrics.y_true is None or metrics.y_score is None:
            continue
        if len(np.unique(metrics.y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(metrics.y_true, metrics.y_score[:, 1])
        ax.plot(
            fpr, tpr, lw=2,
            color=PALETTE.get(result.architecture, DEFAULT_COLOR),
            label=f"{result.architecture} (AUC = {auc(fpr, tpr):.4f})",
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("No architecture produced a plottable ROC curve")

    ax.plot([0, 1], [0, 1], "--", color="#cccccc", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC comparison - {cfg.dataset.name} test set")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    return _save(fig, path)


#: Error-type colours. These encode *status*, not identity: a missed
#: abnormality and a false alarm are different kinds of failure with very
#: different costs, so they get a critical hue and a neutral-informational hue
#: rather than two arbitrary categorical colours. Validated for colour-vision
#: deficiency separation (deutan dE 18.8, normal dE 27.0) and for contrast
#: against a light surface.
ERROR_COLOURS = {"missed": "#9E2B22", "false_alarm": "#1C7FA8"}


def plot_comparison_bars(table: pd.DataFrame, path: Path) -> Path:
    """Clinical error counts per architecture.

    **Why not the headline rates.** Every rate in this comparison falls between
    0.95 and 1.00. Plotted on an honest 0-1 axis the three architectures are
    visually identical, and the chart tells the reader nothing; plotted on a
    truncated axis it exaggerates hairline differences into apparent gulfs.
    Neither is worth printing.

    What actually differs between these models is the *shape of their errors* -
    VGG16 misses 2 abnormal scans but raises 11 false alarms, EfficientNetB0
    misses 4 and raises 2. That is the trade-off a clinician is being asked to
    accept, it is legible in absolute counts, and it needs no axis trickery.
    """
    if "missed" not in table.columns or "false_alarms" not in table.columns:
        raise ValueError(
            "comparison table lacks 'missed'/'false_alarms' columns; "
            "rebuild it with build_comparison_table"
        )

    ordered = table.sort_values("missed", ascending=False).reset_index(drop=True)
    architectures = ordered["architecture"].tolist()
    y = np.arange(len(architectures))
    height = 0.36

    fig, ax = plt.subplots(figsize=(8.6, 0.95 * len(architectures) + 2.1))

    series = [
        ("missed", "Abnormal scans missed", ERROR_COLOURS["missed"]),
        ("false_alarms", "Normal scans falsely flagged", ERROR_COLOURS["false_alarm"]),
    ]

    for index, (column, label, colour) in enumerate(series):
        # +/- half a bar height, with a small gap so adjacent fills never touch.
        offset = (0.5 - index) * (height + 0.03)
        values = ordered[column].astype(float).tolist()
        bars = ax.barh(y + offset, values, height, label=label, color=colour)
        # Direct labels: the counts are small integers, so every bar is
        # labelled rather than making the reader measure against the axis.
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=9,
                     color="#33454E", fontweight="medium")

    ax.set_yticks(y, architectures, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Scans, out of 231 abnormal and 231 normal in the test set")
    ax.set_title("Where each architecture makes its mistakes", pad=34, loc="left")

    ax.set_xlim(0, max(float(ordered[["missed", "false_alarms"]].to_numpy().max()) * 1.25, 4))
    ax.xaxis.grid(True, alpha=0.25)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Above the plot area rather than inside it: the longest bar plus its
    # direct label reaches far enough right that an in-axes legend crowds it.
    ax.legend(
        frameon=False, fontsize=9, ncol=2,
        loc="lower left", bbox_to_anchor=(0, 1.01),
    )

    fig.text(
        0.5, 0.008,
        "A missed abnormality and a false alarm are not equivalent: one delays a "
        "diagnosis, the other costs a review.",
        ha="center", fontsize=8, color="#5C6B73",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return _save(fig, path)


def generate_run_figures(result: TrainingResult, cfg: Config) -> dict[str, Path]:
    """Produce the standard figure set for one completed run.

    A plotting failure is logged and skipped rather than raised: figures are a
    reporting convenience, and losing one must never invalidate a training run
    that has already completed.
    """
    figures_dir = result.run_dir / "figures"
    written: dict[str, Path] = {}

    jobs = [
        ("confusion_matrix", lambda: plot_confusion_matrix(
            result.test_metrics, figures_dir / "confusion_matrix.png",
            title=f"Confusion matrix - {result.architecture}")),
        ("roc_curve", lambda: plot_roc_curve(
            result.test_metrics, figures_dir / "roc_curve.png",
            title=f"ROC - {result.architecture}")),
        ("pr_curve", lambda: plot_precision_recall_curve(
            result.test_metrics, figures_dir / "pr_curve.png",
            title=f"Precision-Recall - {result.architecture}")),
        ("history", lambda: plot_training_history(
            result.history, figures_dir / "training_history.png",
            title=f"Training history - {result.architecture}")),
    ]

    for name, job in jobs:
        try:
            written[name] = job()
        except Exception as exc:
            log.warning("Could not generate %s: %s", name, exc)

    return written


__all__ = [
    "generate_run_figures",
    "plot_comparison_bars",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_comparison",
    "plot_roc_curve",
    "plot_training_history",
]
