#!/usr/bin/env python
"""Quantify how split protocol inflates reported performance on Br35H.

Trains the same architecture under the same schedule with three split
protocols, five seed-varied folds each:

    random         file-wise random split - what most published Br35H
                   results use
    grouped        patient-grouped split without duplicate detection. Br35H
                   publishes no patient metadata, so every file is its own
                   group and this is expected to behave like `random`; the
                   arm exists to demonstrate that patient grouping alone
                   cannot fix a benchmark that ships no patient identifiers
    grouped_dedup  near-duplicate clusters merged into shared groups before
                   splitting - this project's protocol

Alongside the test metrics, the test-set contamination rate (test images
with a Hamming-distance-1 twin in their own training split) is measured per
fold: the mechanism, not just the symptom.

Usage:
    python scripts/ablate_splits.py --config efficientnet_b0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from neuroscan.config import load_config
from neuroscan.utils import format_duration, get_logger, make_run_id, setup_logging, write_json

log = get_logger("scripts.ablate_splits")

PROTOCOLS: dict[str, dict[str, bool]] = {
    "random": {"group_by_patient": False, "detect_near_duplicates": False},
    "grouped": {"group_by_patient": True, "detect_near_duplicates": False},
    "grouped_dedup": {"group_by_patient": True, "detect_near_duplicates": True},
}

METRIC_KEYS = ("accuracy", "recall", "specificity", "auc_roc", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="efficientnet_b0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS),
                        choices=list(PROTOCOLS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    from neuroscan.data.adapters import discover_records
    from neuroscan.data.datamodule import build_dataloaders
    from neuroscan.data.splits import iter_cross_validation_folds
    from neuroscan.evaluation.integrity import test_set_contamination
    from neuroscan.models.factory import build_model
    from neuroscan.training.trainer import Trainer

    base_cfg = load_config(args.config)
    arch = base_cfg.training.architecture
    out_dir = Path(args.out) if args.out else base_cfg.paths.artifacts_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    out_path = out_dir / "split_ablation.json"

    # Resume-friendly: a rerun of one protocol merges into the existing file
    # instead of discarding the arms that already finished.
    results: dict[str, object] = {
        "architecture": arch,
        "folds": args.folds,
        "protocols": {},
    }
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
            if previous.get("architecture") == arch and previous.get("folds") == args.folds:
                results["protocols"].update(previous.get("protocols", {}))
                log.info("Merging into existing results: %s", sorted(results["protocols"]))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read existing %s (%s) - starting fresh", out_path, exc)

    for protocol in args.protocols:
        overrides = PROTOCOLS[protocol]
        cfg = base_cfg.model_copy(
            update={"split": base_cfg.split.model_copy(update=overrides)}
        )
        records = discover_records(cfg)
        base_run_id = make_run_id(f"ablate_{protocol}_{arch}", cfg.dataset.name)

        fold_metrics: list[dict] = []
        fold_contamination: list[dict] = []

        for fold, splits in iter_cross_validation_folds(records, cfg, n_folds=args.folds):
            log.info("#" * 78)
            log.info("# PROTOCOL %s | FOLD %d/%d", protocol, fold + 1, args.folds)
            log.info("#" * 78)

            # Measured on the raw split, before any training: this is a
            # property of the protocol, not of the model.
            contamination = test_set_contamination(splits.train, splits.test)
            fold_contamination.append(contamination)
            log.info(
                "Test-set contamination under %r: %d/%d images (%.1f%%) have a "
                "near-twin in train.",
                protocol, contamination["contaminated"], contamination["test_images"],
                contamination["contamination_rate"] * 100,
            )

            loaders = build_dataloaders(splits, cfg)
            model = build_model(cfg, architecture=arch)
            run_dir = cfg.paths.runs_dir / base_run_id / f"fold_{fold}"
            trainer = Trainer(
                model, loaders, cfg, splits, run_dir, f"{base_run_id}_fold{fold}",
                # An ablation arm must never overwrite the served model.
                export_for_serving=False,
            )
            result = trainer.fit()
            fold_metrics.append(result.test_metrics.to_dict())

        aggregate = {
            key: {
                "mean": float(np.mean([m[key] for m in fold_metrics])),
                "std": float(np.std([m[key] for m in fold_metrics], ddof=1))
                if len(fold_metrics) > 1 else 0.0,
            }
            for key in METRIC_KEYS
        }
        contamination_rates = [c["contamination_rate"] for c in fold_contamination]
        results["protocols"][protocol] = {
            "aggregate": aggregate,
            "contamination": {
                "mean_rate": float(np.mean(contamination_rates)),
                "per_fold": fold_contamination,
            },
            "per_fold": fold_metrics,
            "run_id": base_run_id,
        }

        write_json(out_path, results)  # checkpoint after every protocol

    print("\n" + "=" * 78)
    print(f"SPLIT-PROTOCOL ABLATION: {arch}, {args.folds} folds "
          f"({format_duration(time.perf_counter() - started)})")
    print("=" * 78)
    header = f"{'protocol':<16}{'contam.':>9}"
    for key in METRIC_KEYS:
        header += f"{key:>16}"
    print(header)
    for protocol, data in results["protocols"].items():
        row = f"{protocol:<16}{data['contamination']['mean_rate'] * 100:>8.1f}%"
        for key in METRIC_KEYS:
            stats = data["aggregate"][key]
            row += f"{stats['mean'] * 100:>10.2f}±{stats['std'] * 100:<5.2f}"
        print(row)
    print(f"\nWritten to: {out_dir / 'split_ablation.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
