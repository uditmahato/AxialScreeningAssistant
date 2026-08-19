#!/usr/bin/env python
"""Measure the safety architecture against the adversarial probe corpus.

Runs every input probe through the pre-retrieval question screen and every
output probe through the post-generation validator, then reports block and
flag rates per category and per script (English, Devanagari, romanised
Nepali), alongside the false-block rate on benign controls.

The probes encode required behaviour, not observed behaviour: a missed
probe is printed as a finding. The corpus is versioned so reported numbers
are reproducible.

Usage:
    python scripts/evaluate_safety.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.config import load_config
from neuroscan.rag.advisory import validate_generated_text
from neuroscan.safety import screen_question
from neuroscan.utils import get_logger, setup_logging, write_json

log = get_logger("scripts.evaluate_safety")

PROBES_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "safety" / "probes.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", default=str(PROBES_PATH))
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config()
    out_dir = Path(args.out) if args.out else cfg.paths.artifacts_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(Path(args.probes).read_text(encoding="utf-8"))

    # -- input screening ---------------------------------------------------
    input_rows = []
    for probe in corpus["input_probes"]:
        language = "ne" if probe["language"].startswith("ne") else "en"
        check = screen_question(probe["text"], language)
        blocked = not check.allowed
        input_rows.append({
            **probe,
            "blocked": blocked,
            "screen_category": check.category,
            "correct": blocked == probe["expect_block"],
        })

    # -- output validation -------------------------------------------------
    output_rows = []
    for probe in corpus["output_probes"]:
        flags = validate_generated_text(probe["text"])
        flagged = bool(flags)
        output_rows.append({
            **probe,
            "flagged": flagged,
            "validator_flags": flags,
            "correct": flagged == probe["expect_flag"],
        })

    def rates(rows: list[dict], outcome_key: str) -> dict:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["category"], row.get("language", "output"))].append(row)
        table = {}
        for (category, language), members in sorted(grouped.items()):
            hits = sum(1 for m in members if m[outcome_key])
            correct = sum(1 for m in members if m["correct"])
            table[f"{category}/{language}"] = {
                "n": len(members),
                outcome_key: hits,
                "rate": round(hits / len(members), 3),
                "correct": correct,
            }
        return table

    report = {
        "corpus_version": corpus["version"],
        "input": {
            "n": len(input_rows),
            "by_group": rates(input_rows, "blocked"),
            "misses": [r["id"] for r in input_rows if not r["correct"]],
            "rows": input_rows,
        },
        "output": {
            "n": len(output_rows),
            "by_group": rates(output_rows, "flagged"),
            "misses": [r["id"] for r in output_rows if not r["correct"]],
            "rows": output_rows,
        },
    }

    out_path = out_dir / "safety_eval.json"
    write_json(out_path, report)

    print("\n" + "=" * 74)
    print(f"SAFETY EVALUATION (probe corpus v{corpus['version']})")
    print("=" * 74)
    print("\nINPUT SCREEN (block rate; benign rows should be 0)")
    print(f"{'category/script':<34}{'n':>4}{'blocked':>9}{'rate':>8}{'correct':>9}")
    for group, row in report["input"]["by_group"].items():
        print(f"{group:<34}{row['n']:>4}{row['blocked']:>9}{row['rate']:>8.0%}{row['correct']:>9}")
    print("\nOUTPUT VALIDATOR (flag rate; benign rows should be 0)")
    print(f"{'category/script':<34}{'n':>4}{'flagged':>9}{'rate':>8}{'correct':>9}")
    for group, row in report["output"]["by_group"].items():
        print(f"{group:<34}{row['n']:>4}{row['flagged']:>9}{row['rate']:>8.0%}{row['correct']:>9}")

    misses = report["input"]["misses"] + report["output"]["misses"]
    if misses:
        print(f"\nFINDINGS - {len(misses)} probe(s) not handled as required: {misses}")
    else:
        print("\nAll probes handled as required.")
    print(f"\nWritten to: {out_path}\n")
    return 0 if not misses else 2


if __name__ == "__main__":
    raise SystemExit(main())
