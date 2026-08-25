#!/usr/bin/env python
"""Inject the architecture comparison into README.md.

The results table is generated from `artifacts/comparison/comparison.json`
rather than typed by hand, so the figures in the README cannot drift from the
run that produced them.

Usage:
    python scripts/update_readme_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroscan.config import load_config  # noqa: E402
from neuroscan.utils import read_json  # noqa: E402

MARKER = "<!-- RESULTS_TABLE -->"
END_MARKER = "<!-- /RESULTS_TABLE -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update README results from the last comparison")
    parser.add_argument("--config", default=None)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    return parser.parse_args()


def build_table(payload: dict) -> str:
    rows = payload.get("table", [])
    selection = payload.get("selection", {})

    lines = [
        MARKER,
        "",
        "| Architecture | Accuracy | Precision | Recall | Specificity | F1 | AUC-ROC | Params | CPU latency | Train time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        latency = row.get("cpu_latency_ms")
        latency_text = f"{latency:.0f} ms" if isinstance(latency, (int, float)) and latency == latency else "-"
        lines.append(
            "| `{arch}` | {acc:.4f} | {prec:.4f} | **{rec:.4f}** | {spec:.4f} | "
            "{f1:.4f} | {auc:.4f} | {params:.1f}M | {lat} | {mins:.0f} min |".format(
                arch=row["architecture"],
                acc=row["accuracy"], prec=row["precision"], rec=row["recall"],
                spec=row["specificity"], f1=row["f1"], auc=row["auc_roc"],
                params=row["params_millions"], lat=latency_text,
                mins=row["train_minutes"],
            )
        )

    lines += [
        "",
        f"**Selected for deployment: `{selection.get('selected')}`**. "
        f"{selection.get('reason', '')}",
        "",
        "Recall is bolded because it is the number that matters clinically: it is the",
        "fraction of abnormal scans the system flags. Accuracy averages a miss and a",
        "false alarm as though they cost the same, and in triage they do not.",
        "",
        END_MARKER,
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    source = cfg.paths.artifacts_dir / "comparison" / "comparison.json"
    if not source.exists():
        print(f"No comparison results at {source}.\n"
              f"Run: python scripts/train.py --compare-all", file=sys.stderr)
        return 1

    payload = read_json(source)
    table = build_table(payload)

    readme = args.readme
    text = readme.read_text(encoding="utf-8")

    if MARKER not in text:
        print(f"Marker {MARKER} not found in {readme}", file=sys.stderr)
        return 1

    start = text.index(MARKER)
    end = text.index(END_MARKER) + len(END_MARKER) if END_MARKER in text else start + len(MARKER)

    readme.write_text(text[:start] + table + text[end:], encoding="utf-8")
    print(f"Updated {readme} with {len(payload.get('table', []))} architecture result(s)")
    print(f"Selected: {payload.get('selection', {}).get('selected')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
