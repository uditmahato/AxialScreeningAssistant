#!/usr/bin/env python
"""Score completed clinician-review rating sheets.

Reads every rating_sheet_<rater>.csv in the export directory alongside
key.json, and produces the analysis fixed in advance by
evaluation/clinician_review/instrument.md: Likert means and sds averaged
over raters, the acceptable-by-all criterion, the binary items, and
inter-rater agreement (Cohen's kappa for two raters, Fleiss' for more).

Usage:
    python scripts/score_clinician_review.py
    python scripts/score_clinician_review.py --export evaluation/clinician_review/export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.evaluation.review import load_rating_sheet, summarise_review
from neuroscan.utils import get_logger, setup_logging, write_json

log = get_logger("scripts.score_clinician_review")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="evaluation/clinician_review/export")
    parser.add_argument("--out", default=None,
                        help="Output JSON path; default <export>/clinician_review_scores.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    export = Path(args.export)
    key_path = export / "key.json"
    if not key_path.exists():
        log.error("No key.json under %s; run scripts/export_clinician_review.py first", export)
        return 1
    key = json.loads(key_path.read_text(encoding="utf-8"))["advisories"]

    sheet_paths = sorted(export.glob("rating_sheet_*.csv"))
    if len(sheet_paths) < 2:
        log.error(
            "Found %d completed sheet(s) matching rating_sheet_<rater>.csv under %s; "
            "the design requires at least two raters.", len(sheet_paths), export,
        )
        return 1

    sheets = {}
    for path in sheet_paths:
        rater = path.stem.replace("rating_sheet_", "")
        sheets[rater] = load_rating_sheet(path)
        log.info("Loaded %d rows from rater %r", len(sheets[rater]), rater)

    summary = summarise_review(sheets, key)
    out_path = Path(args.out) if args.out else export / "clinician_review_scores.json"
    write_json(out_path, summary)

    print("\n" + "=" * 60)
    print(f"CLINICIAN REVIEW: {summary['advisories']} advisories, "
          f"{summary['raters']} raters ({summary['kappa_statistic']} kappa)")
    print("=" * 60)
    for item, stats in summary["likert"].items():  # type: ignore[union-attr]
        print(f"{item:<28}{stats['mean']:.2f} (sd {stats['sd']:.2f})")
    acc = summary["acceptable_by_all_raters"]
    print(f"{'acceptable by all raters':<28}{acc['count']} of {acc['of']} ({acc['rate']:.1%})")
    inf = summary["infection_raised_abnormal_only"]
    print(f"{'infection raised (abnormal)':<28}{inf['count']} of {inf['of']}")
    blk = summary["should_have_been_blocked_any_rater"]
    print(f"{'should have been blocked':<28}{blk['count']} of {blk['of']}")
    print("kappa:")
    for item, value in summary["kappa"].items():  # type: ignore[union-attr]
        print(f"  {item:<26}{value if value is not None else 'undefined'}")
    print(f"\nWritten to: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
