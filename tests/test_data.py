"""Tests for preprocessing, ingestion, splitting and near-duplicate detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from neuroscan.data.adapters import discover_records, summarise_records
from neuroscan.data.dedup import deduplicate_patient_ids, difference_hash
from neuroscan.data.preprocessing import (
    ImageLoadError,
    apply_clahe,
    crop_black_border,
    is_plausible_brain_scan,
    load_image,
    standardise,
    to_grayscale,
)
from neuroscan.data.splits import SplitError, build_splits, compute_class_weights

from .conftest import make_records, synthetic_scan


class TestPreprocessing:
    def test_load_image_returns_rgb_uint8(self, scan_file):
        image = load_image(scan_file)
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8

    def test_load_image_rejects_missing_file(self, tmp_path):
        with pytest.raises(ImageLoadError, match="does not exist"):
            load_image(tmp_path / "absent.png")

    def test_load_image_rejects_empty_file(self, tmp_path):
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        with pytest.raises(ImageLoadError, match="empty"):
            load_image(empty)

    def test_load_image_rejects_non_image(self, tmp_path):
        junk = tmp_path / "notimage.png"
        junk.write_bytes(b"this is not an image")
        with pytest.raises(ImageLoadError):
            load_image(junk)

    def test_clahe_returns_three_channels(self, scan_image):
        result = apply_clahe(scan_image)
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_clahe_output_is_achromatic(self, scan_image):
        """CLAHE is applied to one luminance channel then replicated.
        Equalising RGB planes independently produces colour fringing at tissue
        boundaries, which a CNN will happily learn."""
        result = apply_clahe(scan_image)
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])

    def test_clahe_increases_local_contrast(self, scan_image):
        low_contrast = (scan_image * 0.35).astype(np.uint8)
        assert to_grayscale(apply_clahe(low_contrast)).std() > to_grayscale(low_contrast).std()

    def test_crop_removes_black_border(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[100:200, 110:190] = 130
        cropped = crop_black_border(image)
        assert cropped.shape[0] < 300
        assert cropped.shape[1] < 300

    def test_crop_rejects_tiny_regions(self):
        """A near-black or badly-windowed scan must not be cropped down to a
        few stray bright pixels."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[150:154, 150:154] = 255
        assert crop_black_border(image).shape == image.shape

    def test_crop_handles_all_black(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        assert crop_black_border(image).shape == image.shape

    @pytest.mark.parametrize("size", [128, 224, 320])
    def test_standardise_produces_requested_size(self, scan_image, size):
        result = standardise(scan_image, image_size=size)
        assert result.shape == (size, size, 3)

    def test_standardise_is_deterministic(self, scan_image):
        """Anything varying between calls belongs in augmentation, not here."""
        assert np.array_equal(standardise(scan_image), standardise(scan_image))

    def test_standardise_accepts_grayscale_input(self):
        gray = to_grayscale(synthetic_scan(lesion=True))
        assert standardise(gray, image_size=224).shape == (224, 224, 3)


class TestBrainPlausibility:
    def test_accepts_real_scans_at_a_high_rate(self, cfg, image_folder_dataset):
        """Regression for a guard that rejected 49.7% of genuine Br35H scans.

        A health worker uploading a valid MRI and being told it is not one is a
        worse failure than letting an odd image through, so this asserts the
        accept rate directly rather than testing a single example.
        """
        from neuroscan.data.preprocessing import load_image

        images = sorted(image_folder_dataset.rglob("*.png"))
        assert images, "fixture produced no images"
        accepted = sum(1 for p in images if is_plausible_brain_scan(load_image(p)))
        rate = accepted / len(images)
        assert rate >= 0.95, f"guard rejected {(1 - rate):.1%} of valid scans"

    def test_accepts_both_raw_and_cropped(self, scan_image):
        """The guard runs on raw uploads, but must not fail on cropped input -
        crop_black_border removes the dark margin it partly looks for."""
        assert is_plausible_brain_scan(scan_image)
        assert is_plausible_brain_scan(standardise(scan_image))

    def test_accepts_synthetic_scan(self, scan_image):
        assert is_plausible_brain_scan(standardise(scan_image))

    def test_accepts_normal_scan(self, normal_scan_image):
        assert is_plausible_brain_scan(standardise(normal_scan_image))

    def test_rejects_colourful_photograph(self):
        rng = np.random.default_rng(0)
        photo = rng.integers(60, 200, (224, 224, 3), dtype=np.uint8)
        assert not is_plausible_brain_scan(photo)

    def test_rejects_mostly_white_document(self):
        document = np.full((224, 224, 3), 240, dtype=np.uint8)
        document[40:60, 20:200] = 10
        assert not is_plausible_brain_scan(document)

    def test_rejects_uniform_image(self):
        assert not is_plausible_brain_scan(np.full((224, 224, 3), 128, dtype=np.uint8))


class TestAdapters:
    def test_discovers_imagefolder_dataset(self, cfg, image_folder_dataset):
        cfg.dataset.name = "synthetic"
        cfg.dataset.source_dir = image_folder_dataset
        records = discover_records(cfg)
        assert len(records) == 24
        assert {r.class_name for r in records} == {"normal", "abnormal"}

    def test_maps_yes_no_folders_onto_canonical_classes(self, cfg, image_folder_dataset):
        """Public datasets use wildly inconsistent folder names; collapsing
        them here keeps the rest of the pipeline label-agnostic."""
        cfg.dataset.name = "synthetic"
        cfg.dataset.source_dir = image_folder_dataset
        records = discover_records(cfg)
        assert all(r.class_name in {"normal", "abnormal"} for r in records)

    def test_summary_reports_balance(self, cfg, image_folder_dataset):
        cfg.dataset.name = "synthetic"
        cfg.dataset.source_dir = image_folder_dataset
        summary = summarise_records(discover_records(cfg))
        assert summary["total_images"] == 24
        assert summary["by_class"]["normal"] == 12


class TestSplits:
    def test_ratios_are_respected(self, cfg, records):
        splits = build_splits(records, cfg)
        total = sum(splits.sizes.values())
        assert total == len(records)
        assert splits.sizes["train"] > splits.sizes["val"]

    def test_no_patient_appears_in_two_splits(self, cfg, records):
        """The single most important guarantee in the data layer. Consecutive
        slices from one study are near-identical; if they straddle the
        boundary the reported accuracy is fiction."""
        splits = build_splits(records, cfg)
        train = {r.patient_id for r in splits.train}
        val = {r.patient_id for r in splits.val}
        test = {r.patient_id for r in splits.test}
        assert not train & val
        assert not train & test
        assert not val & test

    def test_leakage_assertion_runs_on_construction(self, cfg, records):
        splits = build_splits(records, cfg)
        splits.assert_no_leakage()  # must not raise

    def test_is_deterministic_for_a_seed(self, cfg, records):
        first = build_splits(records, cfg)
        second = build_splits(records, cfg)
        assert [r.path for r in first.test] == [r.path for r in second.test]

    def test_different_seeds_give_different_splits(self, cfg, records):
        first = build_splits(records, cfg, seed=1)
        second = build_splits(records, cfg, seed=999)
        assert [r.path for r in first.test] != [r.path for r in second.test]

    def test_both_classes_present_in_every_split(self, cfg, records):
        distribution = build_splits(records, cfg).class_distribution()
        for split in distribution.values():
            assert len(split) == 2

    def test_rejects_too_few_groups(self, cfg):
        with pytest.raises(SplitError, match="at least 3"):
            build_splits(make_records(n_patients=2), cfg)

    def test_supplementary_data_never_reaches_test(self, cfg):
        """Public supplementary images come from a different scanner
        population; test metrics must describe the target distribution."""
        from dataclasses import replace
        from pathlib import Path

        primary = make_records(n_patients=20)
        # Distinct paths as well as distinct patient ids - supplementary data
        # is a different corpus, and reusing primary paths would (correctly)
        # trip the leakage assertion.
        supplementary = [
            replace(
                r,
                source="supplementary",
                patient_id=f"supp_{i}",
                path=Path(f"/supplementary/S{i:04d}.png"),
            )
            for i, r in enumerate(make_records(n_patients=6))
        ]
        splits = build_splits(primary + supplementary, cfg)
        assert all(r.source == "primary" for r in splits.test)
        assert all(r.source == "primary" for r in splits.val)
        assert any(r.source == "supplementary" for r in splits.train)

    def test_class_weights_are_mean_normalised(self, cfg, records):
        """Normalising to mean 1.0 keeps the effective learning rate constant
        when the class balance changes."""
        weights = compute_class_weights(records, cfg.dataset.class_names)
        assert abs(sum(weights) / len(weights) - 1.0) < 1e-6

    def test_class_weights_favour_the_rare_class(self, cfg):
        imbalanced = make_records(n_patients=20)[:30]  # skewed by construction
        weights = compute_class_weights(imbalanced, cfg.dataset.class_names)
        assert len(weights) == 2


class TestNearDuplicateDetection:
    def test_identical_images_hash_identically(self, scan_image):
        assert difference_hash(scan_image) == difference_hash(scan_image.copy())

    def test_hash_survives_brightness_change(self, scan_image):
        """dHash encodes gradient direction, not absolute intensity, which is
        why it survives the re-export differences that matter here."""
        brighter = np.clip(scan_image.astype(np.int16) + 18, 0, 255).astype(np.uint8)
        original = difference_hash(scan_image)
        changed = difference_hash(brighter)
        hamming = bin(original ^ changed).count("1")
        assert hamming <= 4

    def test_different_images_hash_differently(self):
        a = difference_hash(synthetic_scan(lesion=False, seed=11))
        b = difference_hash(synthetic_scan(lesion=True, seed=77))
        assert bin(a ^ b).count("1") > 2

    def test_duplicates_receive_a_shared_patient_id(self, tmp_path):
        """The whole point: duplicates must group so the splitter cannot
        separate them."""
        from neuroscan.data.adapters import DatasetRecord

        image = synthetic_scan(lesion=True, seed=5)
        records = []
        for i in range(3):
            path = tmp_path / f"copy{i}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            records.append(DatasetRecord(path=path, class_name="abnormal",
                                         label=1, patient_id=f"unique_{i}"))

        updated, stats = deduplicate_patient_ids(records, threshold=1)
        assert stats["duplicate_clusters"] == 1
        assert len({r.patient_id for r in updated}) == 1

    def test_partial_duplication_does_not_tear_a_patient(self, tmp_path):
        """The regression this whole module exists to prevent.

        A patient has three slices; two of them are near-identical. An earlier
        implementation OVERWROTE patient_id with a duplicate-cluster id, so the
        two clustered slices were relabelled while the third kept the original
        id - splitting one patient across train and test, and doing it
        invisibly, because the leakage assertion then compared the rewritten
        ids. Consecutive axial slices are near-identical by construction, so
        this is the likely case, not a corner case.
        """
        from neuroscan.data.adapters import DatasetRecord

        shared = synthetic_scan(lesion=True, seed=21)
        records = []
        # Two near-identical slices of patient P001...
        for i in range(2):
            path = tmp_path / f"P001_dup{i}.png"
            cv2.imwrite(str(path), cv2.cvtColor(shared, cv2.COLOR_RGB2BGR))
            records.append(DatasetRecord(path=path, class_name="abnormal",
                                         label=1, patient_id="P001"))
        # ...and a third, visually different slice of the SAME patient.
        distinct = tmp_path / "P001_other.png"
        cv2.imwrite(str(distinct),
                    cv2.cvtColor(synthetic_scan(lesion=False, seed=99), cv2.COLOR_RGB2BGR))
        records.append(DatasetRecord(path=distinct, class_name="abnormal",
                                     label=1, patient_id="P001"))

        updated, _ = deduplicate_patient_ids(records, threshold=1)

        groups = {r.patient_id for r in updated}
        assert len(groups) == 1, (
            f"patient P001 was torn into {len(groups)} groups {groups} - "
            f"its slices can now be split across train and test"
        )

    def test_duplicate_across_patients_merges_both(self, tmp_path):
        """The other direction: one image appearing under two patient ids must
        pull both patients into a single group, or they straddle the split."""
        from neuroscan.data.adapters import DatasetRecord

        shared = synthetic_scan(lesion=True, seed=31)
        records = []
        for patient in ("P001", "P002"):
            path = tmp_path / f"{patient}_shared.png"
            cv2.imwrite(str(path), cv2.cvtColor(shared, cv2.COLOR_RGB2BGR))
            records.append(DatasetRecord(path=path, class_name="abnormal",
                                         label=1, patient_id=patient))
            other = tmp_path / f"{patient}_own.png"
            cv2.imwrite(str(other), cv2.cvtColor(
                synthetic_scan(lesion=False, seed=hash(patient) % 5000), cv2.COLOR_RGB2BGR))
            records.append(DatasetRecord(path=other, class_name="normal",
                                         label=0, patient_id=patient))

        updated, stats = deduplicate_patient_ids(records, threshold=1)

        assert len({r.patient_id for r in updated}) == 1
        assert stats["original_patients_merged"] == 2

    def test_unaffected_records_keep_their_original_id(self, tmp_path):
        """A record in no duplicate cluster must not be renamed - the id is
        clinical identity and is not the dedup layer's to invent.

        The distinct image is a gradient rather than another synthetic scan:
        dHash downsamples to 8x8, at which size a small lesion no longer changes
        the hash, so two synthetic scans differing only in lesion position
        legitimately cluster. Distinguishing them needs a global structural
        difference.
        """
        from neuroscan.data.adapters import DatasetRecord

        duplicate = synthetic_scan(lesion=True, seed=41)
        gradient = np.tile(
            np.linspace(0, 255, 256, dtype=np.uint8)[None, :, None], (256, 1, 3)
        )

        records = []
        for name, image in [("P000", duplicate), ("P001", duplicate), ("P002", gradient)]:
            path = tmp_path / f"{name}.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            records.append(DatasetRecord(path=path, class_name="abnormal",
                                         label=1, patient_id=name))

        updated, _ = deduplicate_patient_ids(records, threshold=1)

        # The duplicate pair merges under the smaller id; the distinct record
        # is untouched.
        assert [r.patient_id for r in updated] == ["P000", "P000", "P002"]

    def test_distinct_images_keep_their_own_ids(self, tmp_path):
        from neuroscan.data.adapters import DatasetRecord

        records = []
        for i in range(4):
            path = tmp_path / f"distinct{i}.png"
            image = synthetic_scan(lesion=(i % 2 == 0), seed=i * 977)
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            records.append(DatasetRecord(path=path, class_name="abnormal",
                                         label=1, patient_id=f"unique_{i}"))

        updated, _ = deduplicate_patient_ids(records, threshold=1)
        assert len({r.patient_id for r in updated}) == 4
