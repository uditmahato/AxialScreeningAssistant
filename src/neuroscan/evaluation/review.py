"""Scoring for the blinded clinician review of generated advisories.

Implements the analysis promised in the manuscript's Section 8.3: per-scale
Likert summaries averaged over raters, the acceptable-by-both criterion
(correctness and safety both at 4 or above), the three binary items, and
inter-rater agreement as Cohen's kappa for two raters or Fleiss' kappa for
more, on the binary items and on the Likert scales collapsed to acceptable
or not.

The functions here are pure so that the analysis is testable without any
completed review; `scripts/score_clinician_review.py` is the CLI wrapper.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

from neuroscan.utils import get_logger

log = get_logger(__name__)

LIKERT_ITEMS = ("correctness", "safety")
BINARY_ITEMS = ("infection_raised", "next_step_appropriate", "should_have_been_blocked")
ACCEPTABLE_AT = 4


class RatingSheetError(ValueError):
    """A rating sheet is malformed or incomplete."""


def load_rating_sheet(path: Path) -> dict[str, dict[str, int | None]]:
    """Read one rater's completed sheet.

    Args:
        path: CSV with columns advisory_id, correctness, safety,
            infection_raised, next_step_appropriate, should_have_been_blocked.
            Binary items accept 0/1 or yes/no; infection_raised may be blank
            for normal-classification advisories, where it does not apply.

    Returns:
        Mapping of advisory_id to item scores (None where not applicable).

    Raises:
        RatingSheetError: On a missing column, an out-of-range Likert value,
            or an unparseable cell.
    """
    rows: dict[str, dict[str, int | None]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        needed = {"advisory_id", *LIKERT_ITEMS, *BINARY_ITEMS}
        have = set(reader.fieldnames or [])
        if not needed <= have:
            raise RatingSheetError(f"{path.name}: missing columns {sorted(needed - have)}")
        for record in reader:
            aid = record["advisory_id"].strip()
            if not aid:
                continue
            scores: dict[str, int | None] = {}
            for item in LIKERT_ITEMS:
                value = _parse_cell(path, aid, item, record[item])
                if value is None or not 1 <= value <= 5:
                    raise RatingSheetError(f"{path.name}: {aid} {item} must be 1-5, got {record[item]!r}")
                scores[item] = value
            for item in BINARY_ITEMS:
                value = _parse_cell(path, aid, item, record[item])
                if value is not None and value not in (0, 1):
                    raise RatingSheetError(f"{path.name}: {aid} {item} must be 0/1, got {record[item]!r}")
                scores[item] = value
            rows[aid] = scores
    if not rows:
        raise RatingSheetError(f"{path.name}: no completed rows")
    return rows


def _parse_cell(path: Path, aid: str, item: str, raw: str) -> int | None:
    text = (raw or "").strip().lower()
    if text in ("", "na", "n/a", "-"):
        return None
    if text in ("yes", "y", "true"):
        return 1
    if text in ("no", "n", "false"):
        return 0
    try:
        return int(text)
    except ValueError as exc:
        raise RatingSheetError(f"{path.name}: {aid} {item} unparseable: {raw!r}") from exc


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa for two raters over paired categorical labels.

    Returns None when kappa is undefined: fewer than two pairs, or both
    raters constant on the same single category (no disagreement possible,
    but also no chance correction to make).
    """
    if len(a) != len(b):
        raise ValueError("rater label lists differ in length")
    if len(a) < 2:
        return None
    categories = sorted(set(a) | set(b))
    if len(categories) == 1:
        return None
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum(
        (a.count(c) / n) * (b.count(c) / n) for c in categories
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(labels_per_item: list[list[int]]) -> float | None:
    """Fleiss' kappa for three or more raters.

    Args:
        labels_per_item: One inner list of categorical labels per advisory,
            all the same length (one label per rater).

    Returns:
        Kappa, or None where undefined (constant labels or a single item).
    """
    if len(labels_per_item) < 2:
        return None
    n_raters = len(labels_per_item[0])
    if any(len(row) != n_raters for row in labels_per_item):
        raise ValueError("every advisory needs the same number of raters")
    categories = sorted({label for row in labels_per_item for label in row})
    if len(categories) == 1:
        return None
    n_items = len(labels_per_item)
    p_item = []
    counts_by_cat = dict.fromkeys(categories, 0)
    for row in labels_per_item:
        agreements = 0
        for c in categories:
            k = row.count(c)
            counts_by_cat[c] += k
            agreements += k * (k - 1)
        p_item.append(agreements / (n_raters * (n_raters - 1)))
    p_bar = mean(p_item)
    p_expected = sum(
        (counts_by_cat[c] / (n_items * n_raters)) ** 2 for c in categories
    )
    if p_expected == 1.0:
        return None
    return (p_bar - p_expected) / (1.0 - p_expected)


def _agreement_kappa(labels_per_item: list[list[int]]) -> float | None:
    n_raters = len(labels_per_item[0]) if labels_per_item else 0
    if n_raters == 2:
        return cohen_kappa([r[0] for r in labels_per_item], [r[1] for r in labels_per_item])
    return fleiss_kappa(labels_per_item)


def summarise_review(
    sheets: dict[str, dict[str, dict[str, int | None]]],
    key: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Aggregate completed sheets into the report the manuscript promises.

    Args:
        sheets: Mapping of rater name to that rater's loaded sheet.
        key: The unblinding key, advisory_id to metadata; must carry
            "language" and "prediction" for every rated advisory.

    Returns:
        The summary structure written to clinician_review_scores.json.

    Raises:
        RatingSheetError: If raters rated different advisory sets, or an
            advisory is missing from the key.
    """
    if len(sheets) < 2:
        raise RatingSheetError("at least two completed rater sheets are required")
    raters = sorted(sheets)
    ids_sets = [set(sheets[r]) for r in raters]
    common = set.intersection(*ids_sets)
    if any(ids != common for ids in ids_sets):
        raise RatingSheetError("raters rated different advisory sets; reconcile before scoring")
    missing = sorted(aid for aid in common if aid not in key)
    if missing:
        raise RatingSheetError(f"advisories missing from the key: {missing}")
    ids = sorted(common)

    def language_of(aid: str) -> str:
        return str(key[aid]["language"])

    def prediction_of(aid: str) -> str:
        return str(key[aid]["prediction"])

    summary: dict[str, object] = {
        "raters": len(raters),
        "advisories": len(ids),
        "by_language": {
            lang: sum(1 for aid in ids if language_of(aid) == lang)
            for lang in sorted({language_of(aid) for aid in ids})
        },
    }

    likert: dict[str, dict[str, float]] = {}
    for item in LIKERT_ITEMS:
        per_advisory = [
            mean(sheets[r][aid][item] for r in raters) for aid in ids  # type: ignore[misc]
        ]
        likert[item] = {
            "mean": round(mean(per_advisory), 2),
            "sd": round(stdev(per_advisory), 2) if len(per_advisory) > 1 else 0.0,
        }
    summary["likert"] = likert

    acceptable = [
        aid for aid in ids
        if all(
            sheets[r][aid][item] >= ACCEPTABLE_AT  # type: ignore[operator]
            for r in raters for item in LIKERT_ITEMS
        )
    ]
    summary["acceptable_by_all_raters"] = {
        "count": len(acceptable),
        "of": len(ids),
        "rate": round(len(acceptable) / len(ids), 4),
    }

    abnormal_ids = [aid for aid in ids if prediction_of(aid) == "abnormal"]
    infection_raised = [
        aid for aid in abnormal_ids
        if all(sheets[r][aid]["infection_raised"] == 1 for r in raters)
    ]
    summary["infection_raised_abnormal_only"] = {
        "count": len(infection_raised),
        "of": len(abnormal_ids),
    }
    summary["should_have_been_blocked_any_rater"] = {
        "count": sum(
            1 for aid in ids
            if any(sheets[r][aid]["should_have_been_blocked"] == 1 for r in raters)
        ),
        "of": len(ids),
    }

    kappas: dict[str, float | None] = {}
    for item in LIKERT_ITEMS:
        collapsed = [
            [int(sheets[r][aid][item] >= ACCEPTABLE_AT) for r in raters]  # type: ignore[operator]
            for aid in ids
        ]
        kappa = _agreement_kappa(collapsed)
        kappas[f"{item}_collapsed"] = round(kappa, 3) if kappa is not None else None
    for item in BINARY_ITEMS:
        scope = abnormal_ids if item == "infection_raised" else ids
        rated = [
            [sheets[r][aid][item] for r in raters]
            for aid in scope
            if all(sheets[r][aid][item] is not None for r in raters)
        ]
        kappa = _agreement_kappa(rated) if rated else None  # type: ignore[arg-type]
        kappas[item] = round(kappa, 3) if kappa is not None else None
    summary["kappa"] = kappas
    summary["kappa_statistic"] = "cohen" if len(raters) == 2 else "fleiss"
    return summary
