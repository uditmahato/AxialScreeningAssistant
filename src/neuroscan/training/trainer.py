"""The two-stage transfer-learning training loop.

**Stage 1 - head only.** The backbone is frozen and only the new classifier
head trains, at a relatively high learning rate. The head starts from random
weights, so its initial gradients are large; allowing those to propagate into
pre-trained features would destroy the representations that transfer learning
exists to reuse. Skipped entirely when ``epochs_head`` is 0, which is the
correct setting for the scratch baseline.

**Stage 2 - fine-tuning.** The deepest ``unfreeze_layers`` blocks reopen and
train at a much lower rate, alongside the head at its own rate. Shallow layers
stay frozen: edge and texture detectors transfer from ImageNet to MRI
essentially unchanged, and re-training them on a few hundred images only
invites overfitting.

Selection is on validation AUC by default rather than accuracy, because AUC is
threshold-independent - the operating threshold is chosen afterwards, once, on
the validation split, and then frozen before test is ever touched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from neuroscan.data.splits import compute_class_weights
from neuroscan.evaluation.metrics import (
    ClassificationMetrics,
    compute_metrics,
    evaluate_model,
    find_misclassified,
    predict,
    tune_decision_threshold,
)
from neuroscan.models.factory import save_checkpoint
from neuroscan.training.callbacks import EarlyStopping, EpochRecord, MetricHistory
from neuroscan.utils import (
    count_parameters,
    describe_device,
    format_duration,
    get_logger,
    resolve_device,
    set_seed,
    write_json,
)

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import DataLoader

    from neuroscan.config import Config
    from neuroscan.data.splits import SplitResult
    from neuroscan.models.base import BaseClassifier

log = get_logger("training.trainer")


@dataclass
class TrainingResult:
    """Everything a completed run produced."""

    architecture: str
    run_id: str
    run_dir: Path
    best_checkpoint: Path
    history: MetricHistory
    val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    decision_threshold: float
    threshold_diagnostics: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    device: str = ""
    model_summary: dict[str, object] = field(default_factory=dict)
    split_summary: dict[str, object] = field(default_factory=dict)

    def meets_objective(self, target_accuracy: float) -> bool:
        """Whether the run satisfies the >=90% accuracy target."""
        return self.test_metrics.accuracy >= target_accuracy

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "decision_threshold": self.decision_threshold,
            "threshold_diagnostics": self.threshold_diagnostics,
            "total_seconds": round(self.total_seconds, 1),
            "device": self.device,
            "model_summary": self.model_summary,
            "split_summary": self.split_summary,
            "val_metrics": self.val_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "history": self.history.to_dict(),
        }


class Trainer:
    """Orchestrates a single architecture's training run.

    Args:
        model: The classifier to train.
        loaders: ``{'train', 'val', 'test'}`` DataLoaders.
        cfg: Resolved configuration.
        splits: Used for class weighting and provenance.
        run_dir: Destination for checkpoints, history and metrics.
        run_id: Identifier for this run.
    """

    def __init__(
        self,
        model: BaseClassifier,
        loaders: dict[str, DataLoader],
        cfg: Config,
        splits: SplitResult,
        run_dir: Path,
        run_id: str,
        *,
        export_for_serving: bool = True,
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.cfg = cfg
        self.splits = splits
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        # Cross-validation folds set this False. A fold is an experiment on a
        # deliberately different partition, not a deployment candidate; without
        # the flag the last fold trained silently became the served model, so
        # which model answered a clinician depended on the order folds
        # finished in.
        self.export_for_serving = export_for_serving

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.training.device)
        self.model.to(self.device)

        self.criterion = self._build_criterion()
        self.history = MetricHistory()
        self.early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping_patience,
            metric=cfg.training.early_stopping_metric,
        )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.training.mixed_precision and self.device.type == "cuda"
        )
        self.best_checkpoint_path = self.run_dir / f"best_{model.architecture_name}.pt"
        # (primary_metric, -val_loss). See _run_stage for why the tie-break matters.
        self._best_selection: tuple[float, float] = (-float("inf"), -float("inf"))
        self._global_epoch = 0

    # -- setup ---------------------------------------------------------------

    def _build_criterion(self) -> nn.Module:
        """Cross-entropy, optionally class-weighted and label-smoothed."""
        weight = None
        if self.cfg.training.class_weighting:
            weights = compute_class_weights(self.splits.train, self.cfg.dataset.class_names)
            weight = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return nn.CrossEntropyLoss(
            weight=weight, label_smoothing=self.cfg.training.label_smoothing
        )

    def _build_optimizer(self, stage: Literal["head", "finetune"]) -> torch.optim.Optimizer:
        """Build an optimizer over the parameters trainable in this stage."""
        tcfg = self.cfg.training

        if stage == "head":
            params: list[dict] = [
                {
                    "params": [p for m in self.model.head_modules() for p in m.parameters()
                               if p.requires_grad],
                    "lr": tcfg.lr_head,
                    "name": "head",
                }
            ]
        else:
            # Discriminative rates: the reopened backbone moves slowly, the
            # head keeps a rate an order of magnitude higher.
            params = self.model.parameter_groups(
                lr_backbone=tcfg.lr_finetune,
                lr_head=max(tcfg.lr_finetune * 10, tcfg.lr_head / 10),
            )

        params = [g for g in params if g["params"]]
        if not params:
            raise RuntimeError(
                f"No trainable parameters for stage {stage!r}. Check freeze/unfreeze settings."
            )

        if tcfg.optimizer == "sgd":
            return torch.optim.SGD(params, momentum=0.9, weight_decay=tcfg.weight_decay, nesterov=True)
        if tcfg.optimizer == "adam":
            return torch.optim.Adam(params, weight_decay=tcfg.weight_decay)
        return torch.optim.AdamW(params, weight_decay=tcfg.weight_decay)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        """Learning-rate schedule for a stage."""
        if self.cfg.training.scheduler == "none" or epochs <= 1:
            return None
        if self.cfg.training.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # -- loops ---------------------------------------------------------------

    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> float:
        """One pass over the training loader. Returns mean loss."""
        self.model.train()
        loader = self.loaders["train"]
        amp_enabled = self.scaler.is_enabled()

        total_loss = 0.0
        total_count = 0
        skipped = 0

        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = self.criterion(self.model(x), y)

            if not torch.isfinite(loss):
                # A non-finite loss under AMP usually means an overflow that the
                # GradScaler will handle by skipping the step; propagating NaN
                # into the weights instead would poison the whole run.
                skipped += 1
                continue

            self.scaler.scale(loss).backward()

            if self.cfg.training.grad_clip_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), self.cfg.training.grad_clip_norm
                )

            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += float(loss.detach()) * x.size(0)
            total_count += x.size(0)

        if skipped:
            log.warning("Skipped %d batch(es) with non-finite loss this epoch", skipped)

        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        return total_loss / total_count if total_count else float("nan")

    def _validate(self) -> ClassificationMetrics:
        """Evaluate on validation at the fixed configured threshold.

        Deliberately not the tuned threshold: tuning inside the epoch loop
        would let the selection metric drift epoch to epoch and make early
        stopping compare incomparable numbers.
        """
        y_true, y_score, loss, _ = predict(
            self.model,
            self.loaders["val"],
            self.device,
            criterion=self.criterion,
            use_amp=self.cfg.training.mixed_precision,
        )
        return compute_metrics(
            y_true,
            y_score,
            self.cfg.dataset.class_names,
            threshold=self.cfg.evaluation.decision_threshold,
            loss=loss,
        )

    def _selection_value(self, metrics: ClassificationMetrics) -> float:
        key = self.cfg.training.early_stopping_metric
        if key == "val_loss":
            return metrics.loss
        if key == "val_f1":
            return metrics.f1
        return metrics.auc_roc

    def _run_stage(
        self,
        stage: Literal["head", "finetune"],
        epochs: int,
    ) -> None:
        """Run one training stage to completion or early stop."""
        if epochs <= 0:
            log.info("Stage %r has 0 epochs configured - skipping", stage)
            return

        total, trainable = count_parameters(self.model)
        log.info(
            "=== Stage: %s | %d epoch(s) | %s of %s parameters trainable (%.1f%%) ===",
            stage, epochs, f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1),
        )

        optimizer = self._build_optimizer(stage)
        scheduler = self._build_scheduler(optimizer, epochs)
        self.early_stopping.reset()

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()

            train_loss = self._train_one_epoch(optimizer, scheduler)
            val_metrics = self._validate()

            selection_value = self._selection_value(val_metrics)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(selection_value)

            self._global_epoch += 1
            is_stage_best = self.early_stopping.step(selection_value, epoch)

            # Checkpoint against the run-wide best, not the stage-local best.
            # Stage 2 occasionally never beats stage 1 on a small dataset, and
            # in that case the stage-1 weights are the ones worth keeping.
            #
            # The tie-break on validation loss is essential, not cosmetic. On a
            # validation split of a few dozen images, AUC saturates at 1.0
            # within the first epoch or two and then cannot improve. A strict
            # comparison on AUC alone would freeze the checkpoint at that first
            # saturated epoch and discard every subsequent epoch of
            # fine-tuning, shipping a barely-trained model whose class
            # probabilities sit a hair either side of the threshold. Validation
            # loss keeps falling after the metric ceilings, so it ranks the
            # tied epochs by how well-separated the model actually is.
            minimising = self.cfg.training.early_stopping_metric in {"val_loss"}
            comparable = -selection_value if minimising else selection_value
            candidate = (comparable, -val_metrics.loss)
            is_run_best = candidate > self._best_selection
            if is_run_best:
                self._best_selection = candidate
                save_checkpoint(
                    self.best_checkpoint_path,
                    self.model,
                    self.cfg,
                    metrics=val_metrics.to_dict(),  # type: ignore[arg-type]
                    epoch=self._global_epoch,
                )

            record = EpochRecord(
                epoch=self._global_epoch,
                stage_epoch=epoch,
                stage=stage,
                train_loss=train_loss,
                val_loss=val_metrics.loss,
                val_accuracy=val_metrics.accuracy,
                val_recall=val_metrics.recall,
                val_f1=val_metrics.f1,
                val_auc=val_metrics.auc_roc,
                learning_rate=optimizer.param_groups[0]["lr"],
                duration_seconds=time.perf_counter() - started,
                is_best=is_run_best,
            )
            self.history.append(record)
            self.history.log_epoch(record, epochs)

            if self.early_stopping.should_stop:
                log.info("Stage %r stopped early at epoch %d", stage, epoch)
                break

            del is_stage_best  # stage-local best is tracked by EarlyStopping only

    # -- entry point ---------------------------------------------------------

    def fit(self) -> TrainingResult:
        """Run both stages, then tune the threshold and evaluate on test."""
        set_seed(self.cfg.training.seed)
        started = time.perf_counter()

        log.info("Run %s | architecture=%s | device=%s",
                 self.run_id, self.model.architecture_name, describe_device(self.device))
        self.cfg.dump_yaml(self.run_dir / "config.yaml")
        self.splits.save(self.run_dir / "split_manifest.json")

        # -- Stage 1: head only ---------------------------------------------
        if self.cfg.training.epochs_head > 0:
            frozen = self.model.freeze_backbone()
            log.info("Backbone frozen (%s parameters)", f"{frozen:,}")
            self._run_stage("head", self.cfg.training.epochs_head)

        # -- Stage 2: fine-tune ---------------------------------------------
        if self.cfg.training.epochs_finetune > 0:
            if self.cfg.training.epochs_head > 0:
                unfrozen = self.model.unfreeze_last(self.cfg.training.unfreeze_layers)
                log.info(
                    "Reopened the deepest %d layer(s) (%s parameters) for fine-tuning",
                    self.cfg.training.unfreeze_layers, f"{unfrozen:,}",
                )
            else:
                # Scratch baseline: everything trains from the start.
                self.model.unfreeze_all()
            self._run_stage("finetune", self.cfg.training.epochs_finetune)

        # -- Restore the best weights before any reporting -------------------
        if self.best_checkpoint_path.exists():
            from neuroscan.models.factory import load_checkpoint

            self.model, _ = load_checkpoint(self.best_checkpoint_path, device=self.device)
            log.info("Restored best checkpoint for final evaluation")

        # -- Threshold selection: validation only ----------------------------
        threshold = self.cfg.evaluation.decision_threshold
        diagnostics: dict[str, float] = {}
        if self.cfg.evaluation.tune_threshold_on_val and self.cfg.dataset.num_classes == 2:
            y_true, y_score, _, _ = predict(
                self.model, self.loaders["val"], self.device,
                use_amp=self.cfg.training.mixed_precision,
            )
            threshold, diagnostics = tune_decision_threshold(
                y_true,
                y_score,
                min_recall=self.cfg.evaluation.min_recall,
                metric=self.cfg.evaluation.threshold_metric,
            )

        val_metrics = evaluate_model(
            self.model, self.loaders["val"], self.device, self.cfg,
            criterion=self.criterion, threshold=threshold,
        )

        # -- Test: touched exactly once, at the frozen threshold -------------
        test_metrics = evaluate_model(
            self.model, self.loaders["test"], self.device, self.cfg,
            criterion=self.criterion, threshold=threshold,
            with_confidence_intervals=True,
        )

        elapsed = time.perf_counter() - started

        result = TrainingResult(
            architecture=self.model.architecture_name,
            run_id=self.run_id,
            run_dir=self.run_dir,
            best_checkpoint=self.best_checkpoint_path,
            history=self.history,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            decision_threshold=threshold,
            threshold_diagnostics=diagnostics,
            total_seconds=elapsed,
            device=describe_device(self.device),
            model_summary=self.model.describe(),
            split_summary=self.splits.summary(),
        )

        self._persist(result)
        self._log_summary(result)
        return result

    def _persist(self, result: TrainingResult) -> None:
        """Write history, metrics and error analysis into the run directory."""
        self.history.save(self.run_dir / "history.json")
        write_json(self.run_dir / "metrics.json", result.to_dict())

        errors = find_misclassified(result.test_metrics, limit=25)
        if errors:
            write_json(self.run_dir / "misclassified.json", errors)
            log.info("Recorded %d misclassified test image(s) for error analysis", len(errors))

        # Figures are a reporting convenience; a failure here must not discard
        # a run that has already trained and evaluated successfully.
        try:
            from neuroscan.evaluation.plots import generate_run_figures

            figures = generate_run_figures(result, self.cfg)
            log.info("Generated %d figure(s) in %s", len(figures), self.run_dir / "figures")
        except Exception as exc:
            log.warning("Figure generation failed: %s", exc)

        # Re-save the best checkpoint carrying the tuned threshold, so the web
        # application serves the model at the same operating point it was
        # evaluated at rather than falling back to the config default.
        if self.best_checkpoint_path.exists():
            from neuroscan.models.factory import load_checkpoint

            model, _ = load_checkpoint(self.best_checkpoint_path, device="cpu")
            extra = {
                "tuned_threshold": result.decision_threshold,
                "threshold_diagnostics": result.threshold_diagnostics,
                "run_id": self.run_id,
            }
            save_checkpoint(
                self.best_checkpoint_path,
                model,
                self.cfg,
                metrics=result.test_metrics.to_dict(),  # type: ignore[arg-type]
                epoch=self.history.best.epoch if self.history.best else None,
                extra=extra,
            )

            # Export the finished model to models_dir, which is where the web
            # application looks first. Without this it falls back to scanning
            # runs/ for the newest checkpoint, and while another training job
            # is in flight that is a partially-trained model saved mid-epoch -
            # served with the config's default threshold rather than the tuned
            # one. Only completed, non-fold runs are exported here.
            if self.export_for_serving:
                export_path = self.cfg.paths.models_dir / f"best_{model.architecture_name}.pt"
                save_checkpoint(
                    export_path,
                    model,
                    self.cfg,
                    metrics=result.test_metrics.to_dict(),  # type: ignore[arg-type]
                    epoch=self.history.best.epoch if self.history.best else None,
                    extra=extra,
                )
                log.info("Exported completed model for serving: %s", export_path)
            else:
                log.info(
                    "Not exporting %s for serving - this run is a cross-validation "
                    "fold, not a deployment candidate.", self.run_id,
                )

    def _log_summary(self, result: TrainingResult) -> None:
        target = self.cfg.evaluation.target_accuracy
        met = result.meets_objective(target)

        log.info("=" * 78)
        log.info("RUN COMPLETE: %s (%s)", result.run_id, format_duration(result.total_seconds))
        log.info("  validation : %s", result.val_metrics.summary_line())
        log.info("  test       : %s", result.test_metrics.summary_line())
        log.info("  threshold  : %.3f", result.decision_threshold)
        log.info("  %s", result.test_metrics.clinical_summary())
        log.info(
            "  Accuracy target (>=%.0f%% accuracy): %s (test accuracy %.2f%%)",
            target * 100, "MET" if met else "NOT MET", result.test_metrics.accuracy * 100,
        )

        intervals = result.test_metrics.per_class.get("_confidence_intervals")
        if isinstance(intervals, dict):
            for name in ("accuracy", "recall", "auc_roc"):
                ci = intervals.get(name)
                if ci:
                    log.info(
                        "  %-9s: %.4f (95%% CI %.4f - %.4f)",
                        name, ci["point"], ci["lower"], ci["upper"],
                    )
        log.info("=" * 78)


def train_from_config(
    cfg: Config,
    *,
    architecture: str | None = None,
    run_id: str | None = None,
) -> TrainingResult:
    """Build everything from a config and run one training job end to end."""
    from neuroscan.data.adapters import discover_records, summarise_records
    from neuroscan.data.datamodule import build_dataloaders
    from neuroscan.data.splits import build_splits
    from neuroscan.models.factory import build_model
    from neuroscan.utils import make_run_id

    set_seed(cfg.training.seed)

    arch = architecture or cfg.training.architecture
    identifier = run_id or make_run_id(arch, cfg.dataset.name)
    run_dir = cfg.paths.runs_dir / identifier

    records = discover_records(cfg)
    log.info("Dataset provenance: %s", summarise_records(records))

    splits = build_splits(records, cfg)
    loaders = build_dataloaders(splits, cfg)
    model = build_model(cfg, architecture=arch)

    trainer = Trainer(model, loaders, cfg, splits, run_dir, identifier)
    return trainer.fit()


__all__ = ["Trainer", "TrainingResult", "train_from_config"]
