"""Tests for the dataset-integrity and calibration measurements.

These functions produce the research evidence tables, so they are verified
against hand-computable cases: known Hamming distances, synthetic duplicate
structures, and calibration curves with closed-form ECE.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroscan.evaluation.integrity import (
    calibration_curve_binary,
    cluster_hashes,
    cross_dataset_overlap,
    min_hamming_to_reference,
)


class TestMinHamming:
    def test_matches_naive_computation(self):
        rng = np.random.default_rng(7)
        a = rng.integers(0, 2**63, size=40, dtype=np.uint64)
        b = rng.integers(0, 2**63, size=25, dtype=np.uint64)

        naive = np.array([
            min(bin(int(x) ^ int(y)).count("1") for y in b) for x in a
        ])
        fast = min_hamming_to_reference(a, b, block=7)  # non-divisor block size
        assert np.array_equal(fast, naive)

    def test_identical_hash_gives_zero(self):
        a = np.array([0b1010, 0b1111], dtype=np.uint64)
        b = np.array([0b1010], dtype=np.uint64)
        minima = min_hamming_to_reference(a, b)
        assert minima[0] == 0
        assert minima[1] == 2  # 0b1111 ^ 0b1010 = 0b0101

    def test_empty_reference_returns_max_distance(self):
        a = np.array([1, 2], dtype=np.uint64)
        assert (min_hamming_to_reference(a, np.array([], dtype=np.uint64)) == 64).all()


class TestClusterHashes:
    def test_exact_duplicates_cluster_together(self):
        hashes = np.array([10, 10, 99, 10, 500], dtype=np.uint64)
        clusters = cluster_hashes(hashes, threshold=0)
        assert clusters[0] == clusters[1] == clusters[3]
        assert clusters[2] != clusters[0]
        assert clusters[4] != clusters[0]

    def test_transitive_chaining(self):
        """A within 1 of B, B within 1 of C: all three share a cluster even
        though A and C are 2 apart."""
        a, b, c = 0b000, 0b001, 0b011
        clusters = cluster_hashes(np.array([a, b, c], dtype=np.uint64), threshold=1)
        assert clusters[0] == clusters[1] == clusters[2]


class TestCrossDatasetOverlap:
    def test_counts_twins_at_each_threshold(self):
        a = np.array([0b0000, 0b1111_0000], dtype=np.uint64)
        b = np.array([0b0000, 0b0001, 0b1010_1010_1010], dtype=np.uint64)
        result = cross_dataset_overlap(a, b, thresholds=(0, 1))
        assert result["overlap"]["0"]["b_images_with_twin_in_a"] == 1
        assert result["overlap"]["1"]["b_images_with_twin_in_a"] == 2


class TestCalibration:
    def test_perfectly_calibrated_bins_give_zero_ece(self):
        # In each occupied bin the observed positive fraction equals the
        # predicted probability exactly.
        p = np.array([0.25] * 4 + [0.75] * 4)
        y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        curve = calibration_curve_binary(y, p, n_bins=2)
        assert curve["ece"] == 0.0

    def test_overconfident_model_scores_its_gap(self):
        # Every prediction says 0.95 abnormal but only half are abnormal:
        # ECE must equal the 0.45 gap.
        p = np.full(20, 0.95)
        y = np.array([1, 0] * 10)
        curve = calibration_curve_binary(y, p, n_bins=10)
        assert curve["ece"] == pytest.approx(0.45, abs=1e-6)

    def test_top_bin_includes_probability_one(self):
        curve = calibration_curve_binary(np.array([1]), np.array([1.0]), n_bins=10)
        assert curve["bins"][-1]["count"] == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calibration_curve_binary(np.array([1, 0]), np.array([0.5]))
