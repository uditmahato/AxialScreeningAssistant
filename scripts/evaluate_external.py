#!/usr/bin/env python
"""Zero-shot external validation of the served checkpoint, decontaminated.

Evaluates the Br35H-trained model on external public datasets, twice per
dataset: on the full set, and after removing every image with a perceptual
twin (Hamming <= 1) in Br35H. The difference between those two rows measures
how much cross-dataset contamination flatters "external" results.

Metrics carry percentile-bootstrap 95% confidence intervals, and calibration
is reported as a reliability curve with expected calibration error - the
interface shows the classifier's confidence to clinicians, so its
calibration is a claim that needs evidence.

Usage:
    python scripts/evaluate_external.py
    python scripts/evaluate_external.py --datasets sartaj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from neuroscan.config import load_config
from neuroscan.utils import get_logger, setup_logging, write_json

log = get_logger("scripts.evaluate_external")

DEFAULT_DATASETS = ("brain_tumor_mri", "sartaj")
CI_METRICS = ("accuracy", "recall", "auc_roc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _plot_reliability(curve: dict, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, weights = [], [], []
    for row in curve["bins"]:
        if row["count"] > 0:
            xs.append(row["mean_predicted"])
            ys.append(row["fraction_positive"])
            weights.append(row["count"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#999999",
            label="Perfect calibration")
    ax.plot(xs, ys, marker="o", linewidth=1.5, color="#09534D",
            label=f"Model (ECE {curve['ece']:.3f})")
    for x, y, w in zip(xs, ys, weights, strict=True):
        ax.annotate(str(w), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Mean predicted probability (abnormal)")
    ax.set_ylabel("Observed abnormal fraction")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    from torch.utils.data import DataLoader

    from neuroscan.data.adapters import ImageFolderAdapter
    from neuroscan.data.datamodule import MRIDataset
    from neuroscan.data.dedup import compute_hashes
    from neuroscan.data.preprocessing import build_eval_transform
    from neuroscan.evaluation.integrity import calibration_curve_binary, min_hamming_to_reference
    from neuroscan.evaluation.metrics import (
        bootstrap_confidence_interval,
        compute_metrics,
        predict,
    )
    from neuroscan.models.factory import find_best_checkpoint, load_checkpoint
    from neuroscan.utils import resolve_device

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else cfg.paths.artifacts_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_best_checkpoint(cfg.paths.models_dir)
    if checkpoint is None:
        log.error("No served checkpoint found under %s", cfg.paths.models_dir)
        return 1

    device = resolve_device(cfg.training.device)
    model, metadata = load_checkpoint(checkpoint, device=device)
    extra = metadata.get("extra", {}) or {}
    threshold = float(
        extra.get("tuned_threshold")
        or metadata.get("decision_threshold")
        or cfg.evaluation.decision_threshold
    )
    log.info("Checkpoint %s at threshold %.3f on %s", checkpoint.name, threshold, device)

    dataset_cfg = cfg.dataset.model_copy(
        update={"adapter": "imagefolder", "patient_id_pattern": None}
    )

    # The training benchmark's hashes define contamination.
    br35h_records = ImageFolderAdapter(
        cfg.paths.raw_dir / "br35h", dataset_cfg, source="br35h"
    ).discover()
    br35h_hashes, _ = compute_hashes(br35h_records)
    log.info("Reference hashes: %d Br35H images", len(br35h_hashes))

    transform = build_eval_transform(cfg.preprocessing)
    report: dict[str, object] = {
        "checkpoint": checkpoint.name,
        "threshold": threshold,
        "trained_on": "br35h",
        "datasets": {},
    }

    def evaluate(records: list, label: str) -> dict:
        dataset = MRIDataset(records, cfg, transform, cache=False, return_metadata=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        y_true, y_score, _, _ = predict(model, loader, device, use_amp=False)
        metrics = compute_metrics(
            y_true, y_score, cfg.dataset.class_names, threshold=threshold
        )
        intervals = {
            name: bootstrap_confidence_interval(
                y_true, y_score, metric=name, threshold=threshold,
                n_samples=2000, seed=cfg.training.seed,
            )
            for name in CI_METRICS
        }
        curve = calibration_curve_binary(y_true, y_score[:, 1])
        _plot_reliability(curve, label, out_dir / f"reliability_{label}.png")
        summary = metrics.to_dict()
        summary["confidence_intervals"] = intervals
        summary["calibration"] = curve
        summary["n_images"] = len(y_true)
        return summary

    for key in args.datasets:
        root = cfg.paths.raw_dir / key
        if not root.exists():
            log.warning("Dataset %r not found at %s - skipping", key, root)
            continue
        records = ImageFolderAdapter(root, dataset_cfg, source=key).discover()
        hashes, valid = compute_hashes(records)
        minima = min_hamming_to_reference(hashes, br35h_hashes)

        contaminated_indices = {valid[i] for i in np.nonzero(minima <= 1)[0]}
        clean_records = [r for i, r in enumerate(records) if i not in contaminated_indices]
        log.info(
            "%s: %d images, %d (%.1f%%) share a near-twin with Br35H and are "
            "excluded from the decontaminated row.",
            key, len(records), len(contaminated_indices),
            100.0 * len(contaminated_indices) / max(len(records), 1),
        )

        report["datasets"][key] = {
            "total_images": len(records),
            "contaminated_with_br35h": len(contaminated_indices),
            "contamination_rate": round(len(contaminated_indices) / max(len(records), 1), 4),
            "full": evaluate(records, f"{key}_full"),
            "decontaminated": evaluate(clean_records, f"{key}_clean"),
        }

    out_path = out_dir / "external_validation.json"
    write_json(out_path, report)

    print("\n" + "=" * 78)
    print(f"EXTERNAL VALIDATION: {checkpoint.name} (threshold {threshold:.2f})")
    print("=" * 78)
    print(f"{'dataset':<22}{'row':<16}{'n':>7}{'acc':>18}{'recall':>18}{'auc':>18}{'ece':>7}")
    def fmt(interval: dict) -> str:
        return (
            f"{interval['point'] * 100:6.2f} "
            f"[{interval['lower'] * 100:5.2f},{interval['upper'] * 100:6.2f}]"
        )

    for key, data in report["datasets"].items():
        for row_name in ("full", "decontaminated"):
            row = data[row_name]
            ci = row["confidence_intervals"]
            print(
                f"{key:<22}{row_name:<16}{row['n_images']:>7}"
                f"{fmt(ci['accuracy']):>18}{fmt(ci['recall']):>18}{fmt(ci['auc_roc']):>18}"
                f"{row['calibration']['ece']:>7.3f}"
            )
    print(f"\nWritten to: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
