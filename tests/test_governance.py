"""Tests for the code-enforced governance commitments.

An audit found two promises that lived only in documentation: the
ethics-approval marker was described as blocking clinical-data training but
nothing checked it, and generated PDF reports persisted indefinitely while
the uploads they came from purged after 24 hours. Both are enforced in code
now, and these tests keep them that way.
"""

from __future__ import annotations

import os
import time

import pytest

from neuroscan.data.adapters import DatasetDiscoveryError, discover_records


class TestEthicsApprovalGuard:
    @pytest.fixture
    def clinical_cfg(self, cfg, tmp_path):
        root = tmp_path / "grande"
        root.mkdir()
        return cfg.model_copy(update={
            "dataset": cfg.dataset.model_copy(update={
                "name": "grande",
                "source_dir": root,
                "requires_ethics_approval": True,
            })
        })

    def test_discovery_refuses_without_the_marker(self, clinical_cfg):
        with pytest.raises(DatasetDiscoveryError, match="ethics approval"):
            discover_records(clinical_cfg)

    def test_error_names_the_marker_path_and_the_policy(self, clinical_cfg):
        with pytest.raises(DatasetDiscoveryError, match="ETHICS_APPROVED"):
            discover_records(clinical_cfg)
        with pytest.raises(DatasetDiscoveryError, match=r"ETHICS\.md"):
            discover_records(clinical_cfg)

    def test_marker_lifts_the_block(self, clinical_cfg):
        """With the marker present the guard passes; discovery then fails for
        the ordinary reason that the directory holds no images, which proves
        the guard was the thing stopping it."""
        marker = clinical_cfg.dataset.source_dir / "ETHICS_APPROVED"
        marker.write_text("Approved 2026-01-01, ref ETH/0000", encoding="utf-8")
        with pytest.raises(DatasetDiscoveryError) as exc:
            discover_records(clinical_cfg)
        assert "ethics approval" not in str(exc.value)

    def test_a_directory_cannot_stand_in_for_the_marker(self, clinical_cfg):
        (clinical_cfg.dataset.source_dir / "ETHICS_APPROVED").mkdir()
        with pytest.raises(DatasetDiscoveryError, match="ethics approval"):
            discover_records(clinical_cfg)

    def test_public_datasets_are_unaffected(self, cfg):
        """The default configuration carries no ethics requirement, so the
        guard must not fire for Br35H-style data."""
        assert cfg.dataset.requires_ethics_approval is False

    def test_grande_config_carries_the_flag(self):
        import yaml

        from neuroscan.config import PROJECT_ROOT

        payload = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "grande_clinical.yaml").read_text(encoding="utf-8")
        )
        assert payload["dataset"]["requires_ethics_approval"] is True


class TestReportRetention:
    def test_old_reports_are_purged_with_uploads(self, cfg):
        from neuroscan.web.services import InferenceService

        service = InferenceService.__new__(InferenceService)
        service.cfg = cfg

        reports = cfg.paths.reports_dir
        reports.mkdir(parents=True, exist_ok=True)
        old_pdf = reports / "report_old.pdf"
        old_pdf.write_bytes(b"%PDF-old")
        stale = time.time() - (cfg.web.retain_uploads_hours + 1) * 3600
        os.utime(old_pdf, (stale, stale))

        fresh_pdf = reports / "report_fresh.pdf"
        fresh_pdf.write_bytes(b"%PDF-fresh")

        removed = service.purge_old_uploads()
        assert removed >= 1
        assert not old_pdf.exists(), "a stale report must purge with its upload"
        assert fresh_pdf.exists(), "a report inside the retention window must survive"

    def test_non_pdf_files_in_reports_dir_are_left_alone(self, cfg):
        from neuroscan.web.services import InferenceService

        service = InferenceService.__new__(InferenceService)
        service.cfg = cfg

        reports = cfg.paths.reports_dir
        reports.mkdir(parents=True, exist_ok=True)
        keep = reports / "README.txt"
        keep.write_text("not a report", encoding="utf-8")
        stale = time.time() - (cfg.web.retain_uploads_hours + 1) * 3600
        os.utime(keep, (stale, stale))

        service.purge_old_uploads()
        assert keep.exists()
