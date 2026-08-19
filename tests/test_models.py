"""Tests for architectures, checkpointing, metrics and Grad-CAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neuroscan.evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_metrics,
    tune_decision_threshold,
)
from neuroscan.evaluation.usability import (
    ParticipantResponse,
    UsabilityAnalysis,
    UsabilityError,
    adjective_rating,
    letter_grade,
)
from neuroscan.models.factory import (
    ARCHITECTURES,
    ModelError,
    build_model,
    load_checkpoint,
    save_checkpoint,
)

ALL_ARCHITECTURES = sorted(ARCHITECTURES)


class TestArchitectures:
    @pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
    def test_forward_pass_shape(self, cfg, architecture):
        cfg.training.architecture = architecture
        model = build_model(cfg).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 224, 224))
        assert output.shape == (2, cfg.dataset.num_classes)

    @pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
    def test_declares_a_gradcam_target(self, cfg, architecture):
        """Part of the model's own definition rather than something the
        explainability code guesses at - a wrong guess silently produces a
        meaningless heatmap."""
        cfg.training.architecture = architecture
        assert isinstance(build_model(cfg).gradcam_target_layer(), torch.nn.Module)

    @pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
    def test_freeze_then_partial_unfreeze(self, cfg, architecture):
        cfg.training.architecture = architecture
        model = build_model(cfg)
        total = sum(p.numel() for p in model.parameters())

        model.freeze_backbone()
        head_only = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert head_only < total

        model.unfreeze_last(5)
        after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert after >= head_only

    def test_unknown_architecture_raises(self, cfg):
        with pytest.raises(ModelError, match="Unknown architecture"):
            build_model(cfg, architecture="not_a_real_model")

    def test_parameter_groups_separate_backbone_and_head(self, cfg):
        model = build_model(cfg, architecture="efficientnet_b0")
        groups = model.parameter_groups(lr_backbone=1e-5, lr_head=1e-3)
        names = {g["name"] for g in groups}
        assert names == {"backbone", "head"}
        assert {g["lr"] for g in groups} == {1e-5, 1e-3}


class TestCheckpoints:
    def test_round_trip_preserves_outputs(self, cfg, tmp_path):
        model = build_model(cfg, architecture="efficientnet_b0").eval()
        path = save_checkpoint(tmp_path / "best.pt", model, cfg, metrics={"accuracy": 0.95})

        restored, metadata = load_checkpoint(path, device="cpu")
        restored.eval()

        probe = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            assert torch.allclose(model(probe), restored(probe), atol=1e-6)
        assert metadata["metrics"]["accuracy"] == 0.95

    def test_checkpoint_carries_preprocessing(self, cfg, tmp_path):
        """Serving a model under preprocessing it was not trained with
        degrades accuracy silently rather than raising."""
        model = build_model(cfg, architecture="baseline_cnn")
        path = save_checkpoint(tmp_path / "b.pt", model, cfg)
        _, metadata = load_checkpoint(path, device="cpu")
        assert metadata["preprocessing"]["image_size"] == cfg.preprocessing.image_size
        assert "clahe_clip_limit" in metadata["preprocessing"]

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(ModelError, match="not found"):
            load_checkpoint(tmp_path / "absent.pt")

    def test_non_checkpoint_file_raises(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        with pytest.raises(ModelError):
            load_checkpoint(bad)


class TestServingExport:
    def test_cv_folds_do_not_overwrite_the_served_model(self, cfg, tmp_path):
        """Regression: cross-validation exported every fold to models_dir, so
        the last fold to finish silently became the model answering
        clinicians - and which one that was depended on scheduling."""
        import inspect

        from neuroscan.training.trainer import Trainer

        signature = inspect.signature(Trainer.__init__)
        assert "export_for_serving" in signature.parameters
        assert signature.parameters["export_for_serving"].default is True

    def test_cross_validation_passes_the_flag(self):
        """The guarantee only holds if the CV path actually sets it."""
        source = (
            Path(__file__).resolve().parents[1] / "scripts" / "train.py"
        ).read_text(encoding="utf-8")
        assert "export_for_serving=False" in source, (
            "run_cross_validation must disable serving export"
        )


class TestMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        metrics = compute_metrics(y_true, y_score, ["normal", "abnormal"], threshold=0.5)
        assert metrics.accuracy == 1.0
        assert metrics.recall == 1.0
        assert metrics.auc_roc == 1.0

    def test_single_class_split_does_not_crash(self):
        """Routine in a small validation fold; must not kill a training run."""
        y_true = np.array([1, 1, 1])
        y_score = np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
        metrics = compute_metrics(y_true, y_score, ["normal", "abnormal"])
        assert metrics.auc_roc == 0.5

    def test_threshold_changes_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([[0.7, 0.3], [0.6, 0.4], [0.45, 0.55], [0.3, 0.7]])
        low = compute_metrics(y_true, y_score, ["normal", "abnormal"], threshold=0.35)
        high = compute_metrics(y_true, y_score, ["normal", "abnormal"], threshold=0.65)
        assert low.recall >= high.recall

    def test_clinical_summary_counts_misses(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.9, 0.1], [0.2, 0.8]])
        summary = compute_metrics(y_true, y_score, ["normal", "abnormal"]).clinical_summary()
        assert "MISSED" in summary

    def test_specificity_and_npv_are_computed(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 3)
        metrics = compute_metrics(y_true, y_score, ["normal", "abnormal"])
        assert metrics.specificity == 1.0
        assert metrics.npv == 1.0


class TestThresholdTuning:
    def test_respects_the_recall_floor(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 50 + [1] * 50)
        positive = np.concatenate([rng.uniform(0.0, 0.6, 50), rng.uniform(0.4, 1.0, 50)])
        y_score = np.column_stack([1 - positive, positive])

        threshold, diagnostics = tune_decision_threshold(y_true, y_score, min_recall=0.9)
        assert 0.0 < threshold < 1.0
        if diagnostics.get("constraint_met"):
            assert diagnostics["recall"] >= 0.9

    def test_chooses_the_middle_of_a_tied_plateau(self):
        """When the classes separate perfectly, every threshold in the gap
        scores identically on validation. Picking an end of that plateau
        leaves zero margin and fails on test; the midpoint maximises it."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        positive = np.array([0.10, 0.15, 0.20, 0.80, 0.85, 0.90])
        y_score = np.column_stack([1 - positive, positive])

        threshold, diagnostics = tune_decision_threshold(y_true, y_score, min_recall=0.9)
        assert 0.20 < threshold < 0.80
        assert diagnostics.get("plateau_size", 0) >= 1

    def test_single_class_returns_default(self):
        y_true = np.array([1, 1, 1])
        y_score = np.column_stack([[0.2] * 3, [0.8] * 3])
        threshold, diagnostics = tune_decision_threshold(y_true, y_score)
        assert threshold == 0.5
        assert diagnostics["reason"] == "single_class"


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 120)
        positive = np.where(y_true == 1, rng.uniform(0.5, 1.0, 120), rng.uniform(0.0, 0.5, 120))
        y_score = np.column_stack([1 - positive, positive])

        result = bootstrap_confidence_interval(y_true, y_score, metric="accuracy", n_samples=200)
        assert result["lower"] <= result["point"] <= result["upper"]

    def test_is_reproducible_for_a_seed(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, 60)
        y_score = np.column_stack([rng.random(60), rng.random(60)])
        a = bootstrap_confidence_interval(y_true, y_score, n_samples=100, seed=7)
        b = bootstrap_confidence_interval(y_true, y_score, n_samples=100, seed=7)
        assert a == b


class TestGradCAM:
    @pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
    def test_produces_a_normalised_heatmap(self, cfg, architecture, scan_image):
        from neuroscan.data.preprocessing import build_eval_transform, standardise
        from neuroscan.explain.gradcam import explain_prediction

        cfg.training.architecture = architecture
        model = build_model(cfg).eval()

        display = standardise(scan_image, image_size=224)
        tensor = build_eval_transform(cfg.preprocessing)(display).unsqueeze(0)

        result = explain_prediction(model, tensor, display)
        assert result.heatmap.shape == display.shape[:2]
        assert 0.0 <= result.heatmap.min() <= result.heatmap.max() <= 1.0
        assert result.overlay.shape == display.shape
        assert 0.0 <= result.confidence <= 1.0

    def test_hooks_are_removed_after_use(self, cfg, scan_image):
        """A model accumulating forward hooks leaks memory across requests and
        retains the computation graph."""
        from neuroscan.data.preprocessing import build_eval_transform, standardise
        from neuroscan.explain.gradcam import GradCAM

        model = build_model(cfg, architecture="baseline_cnn").eval()
        display = standardise(scan_image, image_size=224)
        tensor = build_eval_transform(cfg.preprocessing)(display).unsqueeze(0)

        target = model.gradcam_target_layer()
        before = len(target._forward_hooks)
        with GradCAM(model) as cam:
            cam.generate(tensor)
        assert len(target._forward_hooks) == before

    def test_rejects_a_batch(self, cfg):
        from neuroscan.explain.gradcam import GradCAM

        model = build_model(cfg, architecture="baseline_cnn").eval()
        with GradCAM(model) as cam, pytest.raises(ValueError, match="single image"):
            cam.generate(torch.randn(4, 3, 224, 224))


class TestUsabilityScoring:
    def test_all_best_answers_score_100(self):
        """Odd items positive, even items negative - the alternation is the
        instrument's design."""
        best = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
        assert ParticipantResponse("P1", best).sus_score == 100.0

    def test_all_worst_answers_score_zero(self):
        worst = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
        assert ParticipantResponse("P1", worst).sus_score == 0.0

    def test_all_neutral_scores_fifty(self):
        assert ParticipantResponse("P1", [3] * 10).sus_score == 50.0

    def test_rejects_wrong_item_count(self):
        with pytest.raises(UsabilityError, match="expected 10"):
            ParticipantResponse("P1", [3] * 8)

    def test_rejects_out_of_range_response(self):
        with pytest.raises(UsabilityError, match="outside the valid range"):
            ParticipantResponse("P1", [3] * 9 + [7])

    def test_flags_straight_lining(self):
        """An identical answer to all ten items is near self-contradictory on
        an alternating-polarity scale."""
        assert ParticipantResponse("P1", [4] * 10).is_straight_lined

    def test_grades_and_adjectives_match_the_literature(self):
        assert letter_grade(85) == "A"
        assert letter_grade(60) == "D"
        assert adjective_rating(85) == "Best imaginable"
        assert adjective_rating(30) == "Poor"

    def test_analysis_aggregates(self):
        participants = [
            ParticipantResponse(f"P{i}", [4, 2, 4, 2, 4, 2, 4, 2, 4, 2], role="nurse")
            for i in range(5)
        ]
        analysis = UsabilityAnalysis(participants, min_participants=5)
        assert analysis.n == 5
        assert analysis.mean == 75.0
        assert analysis.meets_participant_target
        assert "nurse" in analysis.by_group("role")

    def test_confidence_interval_uses_t_distribution(self):
        rng = np.random.default_rng(3)
        participants = [
            ParticipantResponse(
                f"P{i}",
                [int(np.clip(round(rng.normal(4, 0.7)), 1, 5)) if j % 2 == 0
                 else int(np.clip(round(rng.normal(2, 0.7)), 1, 5)) for j in range(10)],
            )
            for i in range(20)
        ]
        analysis = UsabilityAnalysis(participants)
        low, high = analysis.confidence_interval_95
        assert low < analysis.mean < high
