"""Dataset-integrity auditing and calibration measurement.

Research evidence for two claims the project makes about the public brain-MRI
benchmarks:

1. **Within-dataset duplication.** Br35H is heavily near-duplicated, so a
   random file-wise split scores the model on images it has effectively
   memorised. The duplication rate and the resulting test-set contamination
   are measured, not asserted.

2. **Cross-dataset contamination.** The public 4-class aggregate dataset was
   assembled from earlier public sets. If an "external" validation set shares
   images with the training benchmark, external validation on it is illusory.
   Overlap between datasets is measured pairwise with the same perceptual
   hash used for leakage-safe splitting.

Calibration is measured because the interface reports the classifier's
confidence to clinicians: a confidence figure is only meaningful if it is
calibrated, and that must be demonstrated rather than assumed.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from neuroscan.data.dedup import _UnionFind, compute_hashes
from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.data.adapters import DatasetRecord

log = get_logger("evaluation.integrity")

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _to_bytes(hashes: np.ndarray) -> np.ndarray:
    """View 64-bit hashes as byte rows for vectorised Hamming arithmetic."""
    return np.asarray(hashes, dtype=np.uint64).astype(">u8").view(np.uint8).reshape(-1, 8)


def min_hamming_to_reference(
    hashes: np.ndarray,
    reference: np.ndarray,
    *,
    block: int = 512,
) -> np.ndarray:
    """Minimum Hamming distance from each hash to any hash in ``reference``.

    Blockwise, because the full distance matrix between two multi-thousand
    image datasets would not fit in memory. Used both to count cross-dataset
    duplicates and to remove contaminated images from an external test set.
    """
    if len(hashes) == 0 or len(reference) == 0:
        return np.full(len(hashes), 64, dtype=np.int64)

    query = _to_bytes(hashes)
    ref = _to_bytes(reference)

    minima = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), block):
        chunk = query[start : start + block]
        distances = _POPCOUNT[chunk[:, None, :] ^ ref[None, :, :]].sum(axis=2)
        minima[start : start + len(chunk)] = distances.min(axis=1)
    return minima


def cluster_hashes(hashes: np.ndarray, *, threshold: int, block: int = 512) -> np.ndarray:
    """Union-find cluster id per hash, joining pairs within ``threshold``.

    Blockwise pair discovery for the same memory reason as
    :func:`min_hamming_to_reference`.
    """
    n = len(hashes)
    union_find = _UnionFind(n)
    if n == 0:
        return np.array([], dtype=np.int64)

    as_bytes = _to_bytes(hashes)
    for start in range(0, n, block):
        chunk = as_bytes[start : start + block]
        distances = _POPCOUNT[chunk[:, None, :] ^ as_bytes[None, :, :]].sum(axis=2)
        rows, cols = np.nonzero(distances <= threshold)
        for r, c in zip(rows, cols, strict=True):
            a = start + int(r)
            b = int(c)
            if a < b:
                union_find.union(a, b)

    return np.array([union_find.find(i) for i in range(n)], dtype=np.int64)


def within_dataset_duplication(
    records: list[DatasetRecord],
    *,
    thresholds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    """Duplication statistics for one dataset at several Hamming thresholds.

    Returns per-threshold: the number of duplicate clusters, how many images
    sit inside one, that count as a fraction of the dataset, the largest
    cluster, and how many clusters mix both class labels - the same image
    published under contradictory labels.
    """
    hashes, valid = compute_hashes(records)
    labels = [records[i].class_name for i in valid]

    result: dict[str, object] = {
        "total_images": len(records),
        "hashed_images": len(hashes),
        "by_class": dict(Counter(r.class_name for r in records)),
        "thresholds": {},
    }

    for threshold in thresholds:
        clusters = cluster_hashes(hashes, threshold=threshold)
        sizes = Counter(clusters.tolist())
        duplicated = {c for c, size in sizes.items() if size > 1}
        affected = sum(sizes[c] for c in duplicated)

        cross_class = 0
        for cluster in duplicated:
            members = {labels[i] for i in np.nonzero(clusters == cluster)[0]}
            if len(members) > 1:
                cross_class += 1

        result["thresholds"][str(threshold)] = {
            "duplicate_clusters": len(duplicated),
            "images_in_clusters": int(affected),
            "duplication_rate": round(affected / max(len(hashes), 1), 4),
            "largest_cluster": int(max(sizes.values())) if sizes else 0,
            "cross_class_clusters": cross_class,
        }

    return result


def cross_dataset_overlap(
    hashes_a: np.ndarray,
    hashes_b: np.ndarray,
    *,
    thresholds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    """How many images of dataset B have a near-twin in dataset A."""
    minima = min_hamming_to_reference(hashes_b, hashes_a)
    return {
        "b_images": len(hashes_b),
        "overlap": {
            str(t): {
                "b_images_with_twin_in_a": int((minima <= t).sum()),
                "rate": round(float((minima <= t).mean()) if len(minima) else 0.0, 4),
            }
            for t in thresholds
        },
    }


def calibration_curve_binary(
    y_true: np.ndarray,
    p_positive: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, object]:
    """Reliability curve and expected calibration error for a binary task.

    Equal-width bins over the predicted positive-class probability. ECE is
    the bin-count-weighted mean absolute gap between predicted probability
    and observed positive fraction - the number a clinician implicitly relies
    on when reading "97% confidence".
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    p_positive = np.asarray(p_positive).ravel().astype(float)
    if len(y_true) != len(p_positive):
        raise ValueError("y_true and p_positive must be the same length")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Right-inclusive top bin so p == 1.0 is counted.
        mask = (p_positive >= lo) & ((p_positive < hi) | ((i == n_bins - 1) & (p_positive <= hi)))
        count = int(mask.sum())
        if count == 0:
            bins.append({"lower": float(lo), "upper": float(hi), "count": 0,
                         "mean_predicted": None, "fraction_positive": None})
            continue
        mean_pred = float(p_positive[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += (count / total) * abs(mean_pred - frac_pos)
        bins.append({
            "lower": float(lo), "upper": float(hi), "count": count,
            "mean_predicted": round(mean_pred, 4),
            "fraction_positive": round(frac_pos, 4),
        })

    return {"ece": round(float(ece), 4), "n_bins": n_bins, "bins": bins}


def test_set_contamination(
    train_records: list[DatasetRecord],
    test_records: list[DatasetRecord],
    *,
    threshold: int = 1,
) -> dict[str, object]:
    """Fraction of a test split with a near-twin in its own training split.

    This is the mechanism behind split-protocol metric inflation, measured
    directly: the model is being scored on images it has effectively seen.
    """
    train_hashes, _ = compute_hashes(train_records)
    test_hashes, _ = compute_hashes(test_records)
    minima = min_hamming_to_reference(test_hashes, train_hashes)
    contaminated = int((minima <= threshold).sum())
    return {
        "threshold": threshold,
        "test_images": len(test_hashes),
        "contaminated": contaminated,
        "contamination_rate": round(contaminated / max(len(test_hashes), 1), 4),
    }


__all__ = [
    "calibration_curve_binary",
    "cluster_hashes",
    "cross_dataset_overlap",
    "min_hamming_to_reference",
    "test_set_contamination",
    "within_dataset_duplication",
]
