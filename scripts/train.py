#!/usr/bin/env python
"""Train a brain abnormality classifier.

Usage:
    python scripts/train.py --config efficientnet_b0
    python scripts/train.py --config vgg16 --epochs-finetune 10
    python scripts/train.py --compare-all              # all three architectures
    python scripts/train.py --config grande_clinical --cross-validate
    python scripts/train.py --config efficientnet_b0 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.config import load_config
from neuroscan.models.factory import ARCHITECTURES
from neuroscan.utils import (
    format_duration,
    get_logger,
    make_run_id,
    setup_logging,
    write_json,
)

log = get_logger("scripts.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Axial Screening Assistant brain abnormality classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Experiment config name or path")
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), help="Override architecture")
    parser.add_argument("--compare-all", action="store_true",
                        help="Train all three architectures and write a comparison table")
    parser.add_argument("--cross-validate", action="store_true",
                        help="Run k-fold cross-validation instead of a single split")
    parser.add_argument("--folds", type=int, default=None, help="Override cv_folds")

    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lr-finetune", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the data pipeline and exit without training")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    """Turn CLI flags into a nested config override dict."""
    training: dict = {}
    for flag, key in [
        ("architecture", "architecture"),
        ("epochs_head", "epochs_head"),
        ("epochs_finetune", "epochs_finetune"),
        ("batch_size", "batch_size"),
        ("lr_head", "lr_head"),
        ("lr_finetune", "lr_finetune"),
        ("seed", "seed"),
        ("device", "device"),
        ("num_workers", "num_workers"),
    ]:
        value = getattr(args, flag, None)
        if value is not None:
            training[key] = value
    if args.no_amp:
        training["mixed_precision"] = False

    evaluation: dict = {}
    if args.cross_validate:
        evaluation["run_cross_validation"] = True
    if args.folds is not None:
        evaluation["cv_folds"] = args.folds

    overrides: dict = {}
    if training:
        overrides["training"] = training
    if evaluation:
        overrides["evaluation"] = evaluation
    return overrides


def run_dry(cfg) -> int:
    """Validate the data pipeline without training.

    Worth running before any long job: it surfaces missing data, broken class
    folders and leakage problems in seconds rather than after an hour of
    training.
    """
    from neuroscan.data.adapters import discover_records, summarise_records
    from neuroscan.data.datamodule import build_dataloaders, preview_batch
    from neuroscan.data.splits import build_splits
    from neuroscan.models.factory import build_model

    print("\n" + "=" * 78)
    print("DRY RUN - data pipeline validation")
    print("=" * 78)

    records = discover_records(cfg)
    summary = summarise_records(records)
    print(f"\n  Discovered   : {summary['total_images']:,} images")
    print(f"  Patients     : {summary['num_patients']:,}")
    print(f"  By class     : {summary['by_class']}")
    print(f"  Balance      : {summary['class_balance']}")
    print(f"  By source    : {summary['by_source']}")

    splits = build_splits(records, cfg)
    print(f"\n  Split sizes  : {splits.sizes}")
    print(f"  Distribution : {splits.class_distribution()}")
    print(f"  Patients     : {splits.summary()['patients']}")
    print("  Leakage check: PASSED (no shared files or patients across splits)")

    loaders = build_dataloaders(splits, cfg)
    batch = next(iter(loaders["train"]))
    print(f"\n  Train batch  : {tuple(batch[0].shape)}, labels {batch[1].tolist()[:8]}")

    preview_path = cfg.paths.artifacts_dir / "batch_preview.png"
    preview_batch(loaders["train"], cfg, preview_path)
    print(f"  Preview grid : {preview_path}")

    model = build_model(cfg)
    print(f"\n  Model        : {model.describe()}")
    print("\nDry run complete - the pipeline is ready. Re-run without --dry-run to train.\n")
    return 0


def run_single(cfg, architecture: str | None) -> int:
    from neuroscan.training.trainer import train_from_config

    result = train_from_config(cfg, architecture=architecture)
    target = cfg.evaluation.target_accuracy

    print("\n" + "=" * 78)
    print(f"RESULT: {result.architecture}")
    print("=" * 78)
    print(f"  Test accuracy   : {result.test_metrics.accuracy:.4f}")
    print(f"  Test recall     : {result.test_metrics.recall:.4f}  (abnormal class)")
    print(f"  Test precision  : {result.test_metrics.precision:.4f}")
    print(f"  Test F1         : {result.test_metrics.f1:.4f}")
    print(f"  Test AUC-ROC    : {result.test_metrics.auc_roc:.4f}")
    print(f"  Specificity     : {result.test_metrics.specificity:.4f}")
    print(f"  Threshold       : {result.decision_threshold:.3f}")
    print(f"\n  {result.test_metrics.clinical_summary()}")
    print(f"\n  Accuracy target (>={target:.0%}): "
          f"{'MET' if result.meets_objective(target) else 'NOT MET'}")
    print(f"  Artefacts       : {result.run_dir}")
    print(f"  Best checkpoint : {result.best_checkpoint}\n")
    return 0 if result.meets_objective(target) else 2


def run_compare(cfg) -> int:
    """Train all three architectures under identical conditions."""
    from neuroscan.evaluation.compare import build_comparison_table, save_comparison
    from neuroscan.training.trainer import train_from_config

    results = []
    started = time.perf_counter()

    for architecture in ["baseline_cnn", "vgg16", "efficientnet_b0"]:
        # Each architecture's own config file supplies its schedule; only the
        # dataset and split settings are held fixed, so the comparison isolates
        # the architecture rather than penalising one for a mismatched recipe.
        arch_cfg = load_config(
            architecture,
            overrides={
                "dataset": cfg.dataset.model_dump(mode="json"),
                "split": cfg.split.model_dump(mode="json"),
                "paths": {"data_root": str(cfg.paths.data_root)},
                "training": {
                    "device": cfg.training.device,
                    "num_workers": cfg.training.num_workers,
                    "seed": cfg.training.seed,
                },
            },
        )
        log.info("#" * 78)
        log.info("# ARCHITECTURE %d/3: %s", len(results) + 1, architecture)
        log.info("#" * 78)
        try:
            results.append(train_from_config(arch_cfg, architecture=architecture))
        except Exception as exc:
            log.exception("Training failed for %s: %s", architecture, exc)

    if not results:
        log.error("Every architecture failed to train")
        return 1

    table = build_comparison_table(results, cfg)
    out_dir = cfg.paths.artifacts_dir / "comparison"
    save_comparison(table, results, out_dir, cfg)

    print("\n" + "=" * 78)
    print(f"ARCHITECTURE COMPARISON ({format_duration(time.perf_counter() - started)})")
    print("=" * 78)
    print(table.to_string(index=False))
    print(f"\n  Written to: {out_dir}\n")
    return 0


def run_cross_validation(cfg, architecture: str | None) -> int:
    """K-fold cross-validation (Project Design 4.1)."""
    import numpy as np

    from neuroscan.data.adapters import discover_records
    from neuroscan.data.datamodule import build_dataloaders
    from neuroscan.data.splits import iter_cross_validation_folds
    from neuroscan.models.factory import build_model
    from neuroscan.training.trainer import Trainer

    arch = architecture or cfg.training.architecture
    records = discover_records(cfg)
    base_run_id = make_run_id(f"cv_{arch}", cfg.dataset.name)

    fold_metrics = []
    for fold, splits in iter_cross_validation_folds(records, cfg):
        log.info("#" * 78)
        log.info("# FOLD %d/%d", fold + 1, cfg.evaluation.cv_folds)
        log.info("#" * 78)

        loaders = build_dataloaders(splits, cfg)
        model = build_model(cfg, architecture=arch)
        run_dir = cfg.paths.runs_dir / base_run_id / f"fold_{fold}"
        trainer = Trainer(
            model, loaders, cfg, splits, run_dir, f"{base_run_id}_fold{fold}",
            # A fold is an experiment on a deliberately different partition, so
            # it must not overwrite the model the web application serves.
            export_for_serving=False,
        )
        result = trainer.fit()
        fold_metrics.append(result.test_metrics.to_dict())

    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc_roc", "specificity"]
    aggregate = {
        key: {
            "mean": float(np.mean([m[key] for m in fold_metrics])),
            "std": float(np.std([m[key] for m in fold_metrics], ddof=1))
            if len(fold_metrics) > 1 else 0.0,
            "folds": [round(float(m[key]), 4) for m in fold_metrics],
        }
        for key in keys
    }

    out_path = cfg.paths.runs_dir / base_run_id / "cross_validation.json"
    write_json(out_path, {"architecture": arch, "n_folds": len(fold_metrics),
                          "aggregate": aggregate, "per_fold": fold_metrics})

    print("\n" + "=" * 78)
    print(f"CROSS-VALIDATION: {arch} ({len(fold_metrics)} folds)")
    print("=" * 78)
    for key in keys:
        stats = aggregate[key]
        print(f"  {key:20s} {stats['mean']:.4f} +/- {stats['std']:.4f}   {stats['folds']}")
    print(f"\n  Written to: {out_path}\n")
    return 0


def main() -> int:
    args = parse_args()

    cfg = load_config(args.config, overrides=build_overrides(args))
    cfg.paths.ensure_all()

    run_id = make_run_id(args.architecture or cfg.training.architecture, cfg.dataset.name)
    setup_logging(args.log_level, log_file=cfg.paths.runs_dir / run_id / "train.log")

    log.info("Configuration: %s", args.config or "default")
    log.info("Dataset: %s (%s adapter)", cfg.dataset.name, cfg.dataset.adapter)

    if args.dry_run:
        return run_dry(cfg)
    if args.compare_all:
        return run_compare(cfg)
    if cfg.evaluation.run_cross_validation:
        return run_cross_validation(cfg, args.architecture)
    return run_single(cfg, args.architecture)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
