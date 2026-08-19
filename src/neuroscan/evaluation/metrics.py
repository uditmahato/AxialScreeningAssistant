"""Classification metrics and model evaluation.

Three decisions here are clinical rather than statistical, and each is stated
where it is made:

**Recall on the abnormal class is the headline number, not accuracy.** A
missed tumour and a false alarm are not equal errors. The false alarm costs a
radiologist's review; the miss costs the delay the whole project exists to
prevent. Accuracy averages the two and hides that asymmetry, so it is reported
but never optimised against.

**The decision threshold is tuned, not assumed.** ``argmax`` on a softmax is a
0.5 threshold, which is only optimal when errors are symmetric. The threshold
is selected on the validation split against a recall floor and then frozen -
never touched again on test.

**Every headline metric carries a bootstrap confidence interval.** On a test
split of maybe 75 images, a single accuracy figure has a spread of several
percentage points. Reporting 0.94 without an interval overstates what the
experiment can support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import DataLoader

    from neuroscan.config import Config
    from neuroscan.models.base import BaseClassifier

log = get_logger("evaluation.metrics")


@dataclass
class ClassificationMetrics:
    """Every metric computed for one split, plus the raw arrays behind them."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    specificity: float
    npv: float
    mcc: float
    kappa: float
    loss: float = 0.0
    threshold: float = 0.5

    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    n_samples: int = 0

    # Retained for curve plotting and error analysis; excluded from to_dict().
    y_true: np.ndarray | None = field(default=None, repr=False)
    y_pred: np.ndarray | None = field(default=None, repr=False)
    y_score: np.ndarray | None = field(default=None, repr=False)
    metadata: list[dict] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable summary for the run manifest."""
        payload = asdict(self)
        for key in ("y_true", "y_pred", "y_score", "metadata"):
            payload.pop(key, None)
        return payload

    def summary_line(self) -> str:
        return (
            f"acc={self.accuracy:.4f} bal_acc={self.balanced_accuracy:.4f} "
            f"prec={self.precision:.4f} rec={self.recall:.4f} f1={self.f1:.4f} "
            f"auc={self.auc_roc:.4f} spec={self.specificity:.4f}"
        )

    def clinical_summary(self) -> str:
        """Plain-language reading of the confusion matrix.

        Written for the final report and the supervisor review, where "12
        abnormal scans were missed" communicates the failure mode far better
        than "recall = 0.84".
        """
        if not self.confusion or len(self.confusion) != 2:
            return "Clinical summary is only defined for the binary task."
        tn, fp = self.confusion[0]
        fn, tp = self.confusion[1]
        return (
            f"Of {tp + fn} abnormal scans, {tp} were correctly flagged and "
            f"{fn} were MISSED. Of {tn + fp} normal scans, {tn} were correctly "
            f"cleared and {fp} were falsely flagged for review."
        )


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    *,
    threshold: float = 0.5,
    loss: float = 0.0,
) -> ClassificationMetrics:
    """Compute the full metric set from labels and predicted scores.

    Args:
        y_true: Integer labels, shape ``(N,)``.
        y_score: Class probabilities, shape ``(N, C)``.
        class_names: Ordered class names; index 1 is the positive
            (``abnormal``) class in the binary task.
        threshold: Applied to the positive-class probability in the binary
            case. Ignored for multiclass, which uses ``argmax``.
        loss: Mean loss over the split, carried through for reporting.

    Returns:
        A populated :class:`ClassificationMetrics`.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score)
    n_classes = len(class_names)
    binary = n_classes == 2

    if y_score.ndim == 1:
        y_score = np.column_stack([1.0 - y_score, y_score])

    if binary:
        positive = y_score[:, 1]
        y_pred = (positive >= threshold).astype(int)
    else:
        y_pred = y_score.argmax(axis=1)

    average = "binary" if binary else "macro"
    # zero_division=0 rather than raising: an early-training split can legally
    # contain no predictions for a class, and that should not abort a run.
    common = {"average": average, "zero_division": 0}

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, **common))
    recall = float(recall_score(y_true, y_pred, **common))
    f1 = float(f1_score(y_true, y_pred, **common))

    auc_roc, auc_pr = _safe_auc(y_true, y_score, binary, n_classes)

    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    specificity, npv = _specificity_npv(cm, binary)

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = 0.0
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    except ValueError:
        kappa = 0.0

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        support = int((y_true == i).sum())
        per_class[name] = {
            "precision": float(precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
            "support": support,
        }

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        specificity=specificity,
        npv=npv,
        mcc=mcc,
        kappa=kappa,
        loss=float(loss),
        threshold=float(threshold),
        per_class=per_class,
        confusion=cm.tolist(),
        class_names=list(class_names),
        n_samples=len(y_true),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )


def _safe_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    binary: bool,
    n_classes: int,
) -> tuple[float, float]:
    """AUC-ROC and average precision, degrading gracefully.

    ``roc_auc_score`` raises when a split contains only one class. That happens
    routinely in a small validation fold, and it must not kill a training run,
    so 0.5 (chance) is returned with a warning instead.
    """
    try:
        if binary:
            if len(np.unique(y_true)) < 2:
                log.warning("Only one class present in this split; AUC is undefined, using 0.5")
                return 0.5, 0.0
            return (
                float(roc_auc_score(y_true, y_score[:, 1])),
                float(average_precision_score(y_true, y_score[:, 1])),
            )

        present = np.unique(y_true)
        if len(present) < 2:
            return 0.5, 0.0
        onehot = np.zeros((len(y_true), n_classes), dtype=int)
        onehot[np.arange(len(y_true)), y_true] = 1
        return (
            float(roc_auc_score(onehot[:, present], y_score[:, present], average="macro", multi_class="ovr")),
            float(average_precision_score(onehot[:, present], y_score[:, present], average="macro")),
        )
    except ValueError as exc:
        log.warning("AUC computation failed (%s); reporting chance level", exc)
        return 0.5, 0.0


def _specificity_npv(cm: np.ndarray, binary: bool) -> tuple[float, float]:
    """Specificity and negative predictive value.

    NPV is the number the referring clinician actually needs: given a 'normal'
    result, how likely is it that the patient truly has no abnormality.
    """
    if not binary or cm.shape != (2, 2):
        return 0.0, 0.0
    tn, fp, fn, _tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    return specificity, npv


def tune_decision_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    min_recall: float = 0.90,
    metric: str = "f2",
) -> tuple[float, dict[str, float]]:
    """Select an operating threshold on the validation split.

    The search maximises ``metric`` subject to a hard floor on recall for the
    abnormal class.

    **Why F2 rather than F1 by default.** F1 weights precision and recall
    equally, which asserts that a missed tumour and a false alarm cost the
    same. In triage they do not: a false alarm costs a radiologist's review,
    while a miss costs the diagnostic delay this project exists to prevent.
    F-beta with beta=2 weights recall twice as heavily as precision, which
    states that trade explicitly rather than leaving it implicit in a
    constraint.

    The recall floor alone was not sufficient. It is a *constraint*, so it
    stops binding as soon as it is satisfied: measured on Br35H, EfficientNetB0
    cleared a 0.95 floor at threshold 0.783 and the tuner then had no reason to
    look lower, despite an AUC of 0.9994 showing that better sensitivity was
    available. The objective has to carry the preference, not just the
    constraint.

    If no threshold reaches ``min_recall``, the constraint is relaxed and the
    highest-recall threshold is returned with a warning rather than silently
    falling back to 0.5.

    Args:
        y_true: Integer labels.
        y_score: Class probabilities, shape ``(N, 2)``.
        min_recall: Required sensitivity for the positive class.
        metric: ``'f1'``, ``'balanced_accuracy'``, or ``'youden'``.

    Returns:
        ``(threshold, diagnostics)``.
    """
    y_true = np.asarray(y_true).ravel()
    positive = np.asarray(y_score)[:, 1] if np.ndim(y_score) > 1 else np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        log.warning("Cannot tune a threshold on a single-class split; keeping 0.5")
        return 0.5, {"reason": "single_class"}

    candidates = np.unique(np.round(np.concatenate([positive, [0.05, 0.5, 0.95]]), 4))
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]
    if candidates.size == 0:
        return 0.5, {"reason": "no_candidates"}

    rows = []
    for t in candidates:
        pred = (positive >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # F-beta with beta=2: recall weighted twice as heavily as precision.
        beta_sq = 4.0
        f2 = (
            (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
            if (precision + recall)
            else 0.0
        )

        if metric == "youden":
            score = recall + specificity - 1.0
        elif metric == "balanced_accuracy":
            score = (recall + specificity) / 2.0
        elif metric == "f1":
            score = f1
        else:
            score = f2

        rows.append(
            {"threshold": float(t), "score": score, "recall": recall,
             "precision": precision, "specificity": specificity, "f1": f1, "f2": f2}
        )

    eligible = [r for r in rows if r["recall"] >= min_recall]
    if not eligible:
        best = max(rows, key=lambda r: (r["recall"], r["score"]))
        log.warning(
            "No threshold reaches recall >= %.2f on validation (best achievable %.3f). "
            "Using threshold %.3f. The model is not yet sensitive enough for triage use.",
            min_recall,
            best["recall"],
            best["threshold"],
        )
        return best["threshold"], {**best, "constraint_met": False}

    # Among thresholds meeting the recall floor, take the best score - but when
    # several thresholds tie, choose the MIDDLE of the tied plateau, not an end
    # of it.
    #
    # This matters more than it looks. On a small validation split the model
    # often separates the classes perfectly, so every threshold in the gap
    # between the highest-scoring normal and the lowest-scoring abnormal gives
    # identical, perfect validation numbers. Picking either end of that plateau
    # places the operating point flush against a validation score, leaving zero
    # margin: any test image landing marginally the wrong side is misclassified.
    # Choosing the midpoint maximises the distance to the nearest validation
    # score in both directions, which is the standard max-margin argument and
    # is what makes the threshold transfer to unseen data.
    best_score = max(r["score"] for r in eligible)
    plateau = [r for r in eligible if r["score"] >= best_score - 1e-9]
    thresholds = sorted(r["threshold"] for r in plateau)
    midpoint = float(np.median(thresholds))

    # Report the plateau member closest to the midpoint so the diagnostics
    # describe an operating point that was actually measured.
    best = min(plateau, key=lambda r: abs(r["threshold"] - midpoint))

    log.info(
        "Tuned threshold %.3f on validation (plateau of %d equivalent threshold(s) "
        "spanning %.3f-%.3f) | recall=%.3f precision=%.3f specificity=%.3f %s=%.3f",
        midpoint, len(plateau), thresholds[0], thresholds[-1],
        best["recall"], best["precision"], best["specificity"], metric, best["score"],
    )
    return midpoint, {
        **best,
        "threshold": midpoint,
        "constraint_met": True,
        "plateau_size": len(plateau),
        "plateau_low": thresholds[0],
        "plateau_high": thresholds[-1],
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "accuracy",
    threshold: float = 0.5,
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap interval for a single metric.

    Resamples the test set with replacement to estimate how much of the
    reported figure is the model and how much is the particular 75 images that
    landed in the test split.

    Returns:
        ``{'point', 'lower', 'upper', 'std'}``. Degenerate resamples - those
        containing only one class, which make AUC undefined - are skipped.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        y_score = np.column_stack([1.0 - y_score, y_score])

    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _score(idx: np.ndarray) -> float | None:
        yt, ys = y_true[idx], y_score[idx]
        if metric == "auc_roc":
            if len(np.unique(yt)) < 2:
                return None
            return float(roc_auc_score(yt, ys[:, 1]))
        yp = (ys[:, 1] >= threshold).astype(int)
        if metric == "accuracy":
            return float(accuracy_score(yt, yp))
        if metric == "recall":
            return float(recall_score(yt, yp, zero_division=0))
        if metric == "precision":
            return float(precision_score(yt, yp, zero_division=0))
        if metric == "f1":
            return float(f1_score(yt, yp, zero_division=0))
        if metric == "balanced_accuracy":
            if len(np.unique(yt)) < 2:
                return None
            return float(balanced_accuracy_score(yt, yp))
        raise ValueError(f"Unsupported metric for bootstrap: {metric!r}")

    point = _score(np.arange(n))
    values = [v for _ in range(n_samples) if (v := _score(rng.integers(0, n, n))) is not None]

    if not values:
        return {"point": point or 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    arr = np.asarray(values)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(point if point is not None else arr.mean()),
        "lower": float(np.percentile(arr, 100 * alpha)),
        "upper": float(np.percentile(arr, 100 * (1 - alpha))),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


@torch.no_grad()
def predict(
    model: BaseClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    criterion: torch.nn.Module | None = None,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, list[dict]]:
    """Run inference over a loader.

    Handles both two-tuple ``(x, y)`` and three-tuple ``(x, y, meta)`` batches,
    so the same function serves validation and the metadata-carrying test set.

    Returns:
        ``(y_true, y_score, mean_loss, metadata)``.
    """
    model.eval()
    all_true: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    all_meta: list[dict] = []
    total_loss = 0.0
    total_count = 0

    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        if len(batch) == 3:
            x, y, meta = batch
            all_meta.extend(meta)
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.detach()) * x.size(0)

        # Softmax in float32: at half precision the exponentials of confident
        # logits saturate, which distorts the probabilities the threshold and
        # the AUC are computed from.
        probs = torch.softmax(logits.float(), dim=1)

        all_true.append(y.detach().cpu().numpy())
        all_score.append(probs.detach().cpu().numpy())
        total_count += x.size(0)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_score = np.concatenate(all_score) if all_score else np.array([])
    mean_loss = total_loss / total_count if total_count and criterion is not None else 0.0
    return y_true, y_score, mean_loss, all_meta


def evaluate_model(
    model: BaseClassifier,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    *,
    criterion: torch.nn.Module | None = None,
    threshold: float | None = None,
    with_confidence_intervals: bool = False,
) -> ClassificationMetrics:
    """Evaluate a model on one split and return the full metric set."""
    y_true, y_score, loss, metadata = predict(
        model, loader, device, criterion=criterion, use_amp=cfg.training.mixed_precision
    )

    effective_threshold = (
        threshold if threshold is not None else cfg.evaluation.decision_threshold
    )
    metrics = compute_metrics(
        y_true, y_score, cfg.dataset.class_names, threshold=effective_threshold, loss=loss
    )
    metrics.metadata = metadata or None

    if with_confidence_intervals and len(y_true) > 0:
        intervals = {
            name: bootstrap_confidence_interval(
                y_true, y_score, metric=name, threshold=effective_threshold,
                n_samples=cfg.evaluation.bootstrap_ci_samples, seed=cfg.training.seed,
            )
            for name in ("accuracy", "recall", "precision", "f1", "auc_roc")
        }
        metrics.per_class["_confidence_intervals"] = intervals  # type: ignore[assignment]

    return metrics


def find_misclassified(metrics: ClassificationMetrics, limit: int = 20) -> list[dict]:
    """List the worst misclassifications for qualitative error analysis.

    Sorted by confidence in the wrong answer, so the most instructive failures
    - the ones the model was certain about - come first. These are the cases
    worth reproducing in the final report alongside their Grad-CAM maps.
    """
    if metrics.y_true is None or metrics.y_pred is None or metrics.metadata is None:
        return []

    rows = []
    for i, (true, pred) in enumerate(zip(metrics.y_true, metrics.y_pred, strict=False)):
        if true == pred or i >= len(metrics.metadata):
            continue
        confidence = float(metrics.y_score[i, pred]) if metrics.y_score is not None else 0.0
        rows.append(
            {
                "path": metrics.metadata[i].get("path", ""),
                "patient_id": metrics.metadata[i].get("patient_id", ""),
                "true_class": metrics.class_names[int(true)],
                "predicted_class": metrics.class_names[int(pred)],
                "confidence": round(confidence, 4),
                "error_type": "false_negative" if true == 1 else "false_positive",
            }
        )

    rows.sort(key=lambda r: r["confidence"], reverse=True)
    return rows[:limit]


__all__ = [
    "ClassificationMetrics",
    "bootstrap_confidence_interval",
    "compute_metrics",
    "evaluate_model",
    "find_misclassified",
    "predict",
    "tune_decision_threshold",
]
