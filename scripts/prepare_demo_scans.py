#!/usr/bin/env python
"""Prepare demonstration scans for usability sessions.

Selects a fixed, seed-stable set from the downloaded public dataset so every
participant sees the same images, and no real patient data is ever used in a
session.

Usage:
    python scripts/prepare_demo_scans.py
    python scripts/prepare_demo_scans.py --n-abnormal 4 --n-normal 4
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.config import load_config
from neuroscan.utils import setup_logging, write_json

OUT_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "usability" / "sample_scans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare usability demo scans")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-abnormal", type=int, default=3)
    parser.add_argument("--n-normal", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    source = cfg.paths.raw_dir / "br35h"

    if not source.exists():
        print(
            f"Public dataset not found at {source}.\n"
            f"Run: python scripts/download_data.py --dataset br35h",
            file=sys.stderr,
        )
        return 1

    abnormal = sorted((source / "yes").glob("*.jpg"))
    normal = sorted((source / "no").glob("*.jpg"))

    if not abnormal or not normal:
        print(f"Expected 'yes' and 'no' folders under {source}", file=sys.stderr)
        return 1

    # Seeded and sorted, so the same six images are chosen on every machine and
    # every participant in the study sees an identical set.
    rng = random.Random(args.seed)
    chosen = (
        [(p, "abnormal") for p in rng.sample(abnormal, min(args.n_abnormal, len(abnormal)))]
        + [(p, "normal") for p in rng.sample(normal, min(args.n_normal, len(normal)))]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in OUT_DIR.glob("demo_*.jpg"):
        existing.unlink()

    manifest = []
    for index, (path, label) in enumerate(chosen, start=1):
        destination = OUT_DIR / f"demo_{index:02d}.jpg"
        shutil.copy2(path, destination)
        manifest.append({
            "file": destination.name,
            "expected_label": label,
            "source_file": path.name,
            "source_dataset": "br35h",
        })
        print(f"  {destination.name}  <- {path.name}  ({label})")

    # The answer key is written separately so a facilitator can avoid opening
    # it; knowing the expected label makes it hard not to prime the
    # participant, and task 2 measures exactly that.
    write_json(OUT_DIR / "ANSWER_KEY.json", {
        "warning": "Facilitators should not consult this before a session.",
        "seed": args.seed,
        "scans": manifest,
    })

    print(f"\n{len(manifest)} demo scan(s) written to {OUT_DIR}")
    print("Answer key: ANSWER_KEY.json (do not read before facilitating)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
