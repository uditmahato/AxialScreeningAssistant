"""Tests for the clinician-review scoring maths.

The review itself has not been run; these tests pin the analysis down with
synthetic sheets so the eventual scoring is trustworthy and regression-safe.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neuroscan.evaluation.review import (
    RatingSheetError,
    cohen_kappa,
    fleiss_kappa,
    load_rating_sheet,
    summarise_review,
)

COLUMNS = ["advisory_id", "correctness", "safety", "infection_raised",
           "next_step_appropriate", "should_have_been_blocked", "comments"]


def write_sheet(path: Path, rows: list[list[object]]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


KEY = {
    "R01": {"language": "en", "prediction": "abnormal"},
    "R02": {"language": "ne", "prediction": "abnormal"},
    "R03": {"language": "en", "prediction": "normal"},
    "R04": {"language": "ne", "prediction": "normal"},
}


class TestKappa:
    def test_perfect_agreement_is_one(self):
        assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)

    def test_chance_only_agreement_is_zero(self):
        # Independent raters, each 50/50, agreeing on exactly half the items.
        assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

    def test_undefined_when_one_category(self):
        assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([1, 0], [1])

    def test_fleiss_perfect_agreement(self):
        assert fleiss_kappa([[1, 1, 1], [0, 0, 0], [1, 1, 1]]) == pytest.approx(1.0)

    def test_fleiss_undefined_for_constant_labels(self):
        assert fleiss_kappa([[1, 1], [1, 1]]) is None

    def test_fleiss_ragged_raises(self):
        with pytest.raises(ValueError):
            fleiss_kappa([[1, 1, 0], [1, 0]])


class TestLoadSheet:
    def test_reads_scores_and_blanks(self, tmp_path):
        sheet = load_rating_sheet(write_sheet(tmp_path / "s.csv", [
            ["R01", 5, 4, "yes", 1, 0, "fine"],
            ["R03", 4, 5, "", "no", "0", ""],
        ]))
        assert sheet["R01"]["correctness"] == 5
        assert sheet["R01"]["infection_raised"] == 1
        assert sheet["R03"]["infection_raised"] is None
        assert sheet["R03"]["next_step_appropriate"] == 0

    def test_out_of_range_likert_rejected(self, tmp_path):
        path = write_sheet(tmp_path / "s.csv", [["R01", 6, 4, 1, 1, 0, ""]])
        with pytest.raises(RatingSheetError, match="must be 1-5"):
            load_rating_sheet(path)

    def test_blank_likert_rejected(self, tmp_path):
        path = write_sheet(tmp_path / "s.csv", [["R01", "", 4, 1, 1, 0, ""]])
        with pytest.raises(RatingSheetError):
            load_rating_sheet(path)

    def test_missing_column_rejected(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_text("advisory_id,correctness\nR01,5\n", encoding="utf-8")
        with pytest.raises(RatingSheetError, match="missing columns"):
            load_rating_sheet(path)


class TestSummarise:
    def sheets(self, tmp_path) -> dict:
        a = load_rating_sheet(write_sheet(tmp_path / "a.csv", [
            ["R01", 5, 5, 1, 1, 0, ""],
            ["R02", 4, 4, 1, 1, 0, ""],
            ["R03", 4, 5, "", 1, 0, ""],
            ["R04", 2, 3, "", 0, 1, "overreach"],
        ]))
        b = load_rating_sheet(write_sheet(tmp_path / "b.csv", [
            ["R01", 5, 4, 1, 1, 0, ""],
            ["R02", 4, 5, 0, 1, 0, ""],
            ["R03", 5, 5, "", 1, 0, ""],
            ["R04", 3, 3, "", 0, 1, ""],
        ]))
        return {"a": a, "b": b}

    def test_report_shape_and_counts(self, tmp_path):
        report = summarise_review(self.sheets(tmp_path), KEY)
        assert report["advisories"] == 4
        assert report["raters"] == 2
        assert report["kappa_statistic"] == "cohen"
        assert report["by_language"] == {"en": 2, "ne": 2}
        # R01, R02, R03 have every rating >= 4 from both raters; R04 does not.
        assert report["acceptable_by_all_raters"]["count"] == 3
        # Infection: R01 both say yes; R02 raters disagree, so not counted.
        assert report["infection_raised_abnormal_only"] == {"count": 1, "of": 2}
        assert report["should_have_been_blocked_any_rater"]["count"] == 1

    def test_likert_mean_over_raters(self, tmp_path):
        report = summarise_review(self.sheets(tmp_path), KEY)
        # Correctness per advisory: 5, 4, 4.5, 2.5 -> mean 4.0.
        assert report["likert"]["correctness"]["mean"] == pytest.approx(4.0)

    def test_blocked_kappa_is_perfect(self, tmp_path):
        report = summarise_review(self.sheets(tmp_path), KEY)
        assert report["kappa"]["should_have_been_blocked"] == pytest.approx(1.0)

    def test_single_rater_rejected(self, tmp_path):
        with pytest.raises(RatingSheetError, match="two completed rater sheets"):
            summarise_review({"a": self.sheets(tmp_path)["a"]}, KEY)

    def test_mismatched_sets_rejected(self, tmp_path):
        sheets = self.sheets(tmp_path)
        del sheets["b"]["R04"]
        with pytest.raises(RatingSheetError, match="different advisory sets"):
            summarise_review(sheets, KEY)

    def test_missing_key_rejected(self, tmp_path):
        with pytest.raises(RatingSheetError, match="missing from the key"):
            summarise_review(self.sheets(tmp_path), {"R01": KEY["R01"]})
