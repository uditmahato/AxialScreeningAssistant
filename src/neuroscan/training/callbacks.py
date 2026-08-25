"""Early stopping and epoch-history tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from neuroscan.utils import get_logger, write_json

log = get_logger("training.callbacks")

#: Metrics where lower is better. Everything else is maximised.
MINIMISE_METRICS = {"val_loss", "train_loss", "loss"}


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    On a few hundred MRI images a transfer-learning model reaches its best
    validation score within a handful of epochs and then memorises the training
    set. Early stopping is what keeps the reported result honest rather than
    letting the run continue to a checkpoint that only looks better on train.

    Args:
        patience: Epochs without improvement before stopping.
        metric: Metric key to monitor, e.g. ``'val_auc'``.
        min_delta: Minimum change that counts as an improvement. Guards against
            declaring victory on floating-point noise.
    """

    def __init__(
        self,
        patience: int = 8,
        metric: str = "val_auc",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.minimise = metric in MINIMISE_METRICS

        self.best_value: float = math.inf if self.minimise else -math.inf
        self.best_epoch: int = -1
        self.counter: int = 0
        self.should_stop: bool = False

    def is_improvement(self, value: float) -> bool:
        if self.minimise:
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record an epoch's metric.

        Returns:
            True if this epoch is a new best (the caller should checkpoint).
        """
        if not math.isfinite(value):
            log.warning("Non-finite value for %s at epoch %d; treating as no improvement",
                        self.metric, epoch)
            self.counter += 1
            self.should_stop = self.counter >= self.patience
            return False

        if self.is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            log.info(
                "Early stopping: %s has not improved on %.5f (epoch %d) for %d epochs",
                self.metric, self.best_value, self.best_epoch, self.patience,
            )
        return False

    def reset(self) -> None:
        """Clear state between training stages.

        Called at the stage 1 -> stage 2 boundary. Without it, the patience
        counter accumulated while the backbone was frozen would carry into
        fine-tuning and could stop it before it has taken a single step.
        """
        self.best_value = math.inf if self.minimise else -math.inf
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False


@dataclass
class EpochRecord:
    """One row of the training history."""

    epoch: int  # global, across both stages - used for plotting
    stage: Literal["head", "finetune"]
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_recall: float
    val_f1: float
    val_auc: float
    learning_rate: float
    duration_seconds: float
    # Position within the current stage, for logging. Distinct from ``epoch``,
    # which is global across both stages so the history plots continuously.
    stage_epoch: int = 0
    is_best: bool = False


@dataclass
class MetricHistory:
    """Per-epoch record of a training run.

    Persisted as JSON so the learning curves in the final report are generated
    from data rather than redrawn by hand, and so a run can be re-plotted
    without retraining.
    """

    records: list[EpochRecord] = field(default_factory=list)

    def append(self, record: EpochRecord) -> None:
        self.records.append(record)

    def series(self, key: str) -> list[float]:
        return [getattr(r, key) for r in self.records]

    @property
    def best(self) -> EpochRecord | None:
        best_records = [r for r in self.records if r.is_best]
        return best_records[-1] if best_records else None

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": [vars(r) for r in self.records],
            "best_epoch": self.best.epoch if self.best else None,
        }

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())
        log.info("Training history written to %s", path)

    def log_epoch(self, record: EpochRecord, total_epochs: int) -> None:
        # Report the stage-local epoch. The record also carries a global epoch
        # for plotting across both stages, but printing that against the
        # stage's own total reads as "epoch 25/20".
        log.info(
            "[%s] epoch %2d/%-2d | train_loss=%.4f val_loss=%.4f | "
            "acc=%.4f rec=%.4f f1=%.4f auc=%.4f | lr=%.2e | %.1fs%s",
            record.stage,
            record.stage_epoch or record.epoch,
            total_epochs,
            record.train_loss,
            record.val_loss,
            record.val_accuracy,
            record.val_recall,
            record.val_f1,
            record.val_auc,
            record.learning_rate,
            record.duration_seconds,
            "  <-- best" if record.is_best else "",
        )


__all__ = ["MINIMISE_METRICS", "EarlyStopping", "EpochRecord", "MetricHistory"]
