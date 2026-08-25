#!/usr/bin/env python
"""Audit the public brain-MRI benchmarks for duplication and cross-contamination.

Measures, per dataset, the near-duplicate rate at several Hamming thresholds,
and, per dataset pair, how many images of one dataset have a perceptual twin
in the other. The output is the evidence table for the claim that random
splits on these benchmarks - and "external" validation between them - are
contaminated.

Usage:
    python scripts/audit_datasets.py
    python scripts/audit_datasets.py --datasets br35h sartaj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.config import load_config
from neuroscan.data.adapters import ImageFolderAdapter
from neuroscan.data.dedup import compute_hashes
from neuroscan.evaluation.integrity import cross_dataset_overlap, within_dataset_duplication
from neuroscan.utils import get_logger, setup_logging, write_json

log = get_logger("scripts.audit_datasets")

DEFAULT_DATASETS = ("br35h", "brain_tumor_mri", "sartaj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="Output directory (default artifacts/audit)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    out_dir = Path(args.out) if args.out else cfg.paths.artifacts_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every dataset is read through the same binary folder-name mapping the
    # training pipeline uses, so the audit describes exactly the data a model
    # would see.
    dataset_cfg = cfg.dataset.model_copy(
        update={"adapter": "imagefolder", "patient_id_pattern": None}
    )

    records_by_key: dict[str, list] = {}
    hashes_by_key: dict[str, object] = {}
    report: dict[str, object] = {"within": {}, "cross": {}}

    for key in args.datasets:
        root = cfg.paths.raw_dir / key
        if not root.exists():
            log.warning("Dataset %r not found at %s - skipping", key, root)
            continue
        records = ImageFolderAdapter(root, dataset_cfg, source=key).discover()
        log.info("%s: %d binary-labelled images", key, len(records))
        records_by_key[key] = records

        log.info("Auditing within-dataset duplication for %s...", key)
        report["within"][key] = within_dataset_duplication(records)
        hashes, _ = compute_hashes(records)
        hashes_by_key[key] = hashes

    keys = list(records_by_key)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            log.info("Measuring cross-dataset overlap: %s vs %s...", a, b)
            report["cross"][f"{b}_in_{a}"] = cross_dataset_overlap(
                hashes_by_key[a], hashes_by_key[b]
            )
            report["cross"][f"{a}_in_{b}"] = cross_dataset_overlap(
                hashes_by_key[b], hashes_by_key[a]
            )

    out_path = out_dir / "dataset_audit.json"
    write_json(out_path, report)

    print("\n" + "=" * 78)
    print("WITHIN-DATASET DUPLICATION (Hamming <= t)")
    print("=" * 78)
    print(f"{'dataset':<18}{'images':>8}{'t':>4}{'clusters':>10}{'affected':>10}{'rate':>8}{'x-class':>9}")
    for key, stats in report["within"].items():
        for t, row in stats["thresholds"].items():
            print(
                f"{key:<18}{stats['hashed_images']:>8}{t:>4}"
                f"{row['duplicate_clusters']:>10}{row['images_in_clusters']:>10}"
                f"{row['duplication_rate'] * 100:>7.1f}%{row['cross_class_clusters']:>9}"
            )

    print("\n" + "=" * 78)
    print("CROSS-DATASET CONTAMINATION (images of B with a twin in A)")
    print("=" * 78)
    print(f"{'pair (B in A)':<38}{'B images':>10}{'t=0':>8}{'t=1':>8}{'t=2':>8}")
    for pair, stats in report["cross"].items():
        row = stats["overlap"]
        print(
            f"{pair:<38}{stats['b_images']:>10}"
            f"{row['0']['b_images_with_twin_in_a']:>8}"
            f"{row['1']['b_images_with_twin_in_a']:>8}"
            f"{row['2']['b_images_with_twin_in_a']:>8}"
        )

    print(f"\nWritten to: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
