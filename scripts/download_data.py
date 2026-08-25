#!/usr/bin/env python
"""Download and verify the public contingency MRI datasets.

Usage:
    python scripts/download_data.py                      # all datasets
    python scripts/download_data.py --dataset br35h      # one dataset
    python scripts/download_data.py --list               # show what is available
    python scripts/download_data.py --verify-only        # check what is on disk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.config import load_config
from neuroscan.data.download import (
    PUBLIC_DATASETS,
    DownloadError,
    download_all,
    download_dataset,
    verify_dataset,
)
from neuroscan.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public brain MRI datasets for Axial Screening Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=sorted(PUBLIC_DATASETS), help="Download one dataset")
    parser.add_argument("--config", default=None, help="Experiment config for path resolution")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    parser.add_argument("--verify-only", action="store_true", help="Verify without downloading")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def print_catalogue() -> None:
    print("\nAvailable public datasets\n" + "=" * 78)
    for key, dataset in PUBLIC_DATASETS.items():
        print(f"\n  {key}")
        print(f"    {dataset.title}")
        print(f"    ~{dataset.approx_images:,} images | classes: {', '.join(dataset.expected_classes)}")
        print(f"    {dataset.kaggle_url}")
        for line in _wrap(dataset.description, 70):
            print(f"    {line}")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if args.list:
        print_catalogue()
        return 0

    cfg = load_config(args.config)
    raw_dir = cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Raw data directory: {raw_dir}\n")

    keys = [args.dataset] if args.dataset else list(PUBLIC_DATASETS)

    if args.verify_only:
        failures = 0
        for key in keys:
            try:
                summary = verify_dataset(PUBLIC_DATASETS[key], raw_dir / key)
                print(f"  OK      {key}: {summary['total_images']:,} images {summary['by_folder']}")
            except DownloadError as exc:
                failures += 1
                print(f"  MISSING {key}: {exc}")
        return 1 if failures else 0

    results = (
        {args.dataset: download_dataset(args.dataset, raw_dir, force=args.force)}
        if args.dataset
        else download_all(raw_dir, force=args.force)
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    failed = 0
    for key, summary in results.items():
        if "error" in summary:
            failed += 1
            print(f"  FAILED  {key}")
        else:
            print(f"  OK      {key}: {summary['total_images']:,} images -> {summary['path']}")
            print(f"          {summary['by_folder']}")

    if failed:
        print(f"\n{failed} dataset(s) could not be downloaded. See the instructions above.")
        return 1

    print("\nNext: python scripts/train.py --config efficientnet_b0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
