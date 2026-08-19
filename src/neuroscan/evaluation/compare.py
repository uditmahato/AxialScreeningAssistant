"""Cross-architecture comparison (Project Design 4.1).

Produces the table the final report needs to justify the deployed model: three
architectures trained under an identical data pipeline and split, scored on the
same held-out test set, with the efficiency figures that matter for a
low-resource deployment alongside the accuracy figures.

Inference latency is measured on CPU deliberately. A district hospital in Nepal
is unlikely to have a GPU, so GPU throughput is not the number that decides
whether the system is deployable there.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch

from neuroscan.utils import get_logger, write_json

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config
    from neuroscan.models.base import BaseClassifier
    from neuroscan.training.trainer import TrainingResult

log = get_logger("evaluation.compare")


def measure_cpu_latency(
    model: BaseClassifier,
    image_size: int = 224,
    *,
    n_warmup: int = 3,
    n_runs: int = 20,
) -> dict[str, float]:
    """Median single-image CPU inference latency, in milliseconds.

    Median rather than mean: a background process on the host produces
    occasional multi-hundred-millisecond outliers that would dominate a mean
    and misrepresent typical performance.

    .. warning::
       This is wall-clock time on a shared machine and is only comparable
       between models measured under the same load. EfficientNetB0 measured
       47 ms on an idle host and 191 ms while a training job was running - a 4x
       swing from unrelated contention. Treat it as an order-of-magnitude
       indicator, and do not let it outrank a stable statistic when selecting a
       model (see :func:`select_best_architecture`).
    """
    device = torch.device("cpu")
    model = model.to(device).eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample)

        timings: list[float] = []
        for _ in range(n_runs):
            started = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    return {
        "median_ms": round(timings[len(timings) // 2], 2),
        "min_ms": round(timings[0], 2),
        "max_ms": round(timings[-1], 2),
    }


def build_comparison_table(results: list[TrainingResult], cfg: Config) -> pd.DataFrame:
    """Assemble the comparison table from completed training runs."""
    from neuroscan.models.factory import load_checkpoint

    rows = []
    for result in results:
        metrics = result.test_metrics
        summary = result.model_summary

        latency: dict[str, float] = {}
        try:
            model, _ = load_checkpoint(result.best_checkpoint, device="cpu")
            latency = measure_cpu_latency(model, cfg.preprocessing.image_size)
        except Exception as exc:
            log.warning("Could not measure CPU latency for %s: %s", result.architecture, exc)

        intervals = metrics.per_class.get("_confidence_intervals", {})
        accuracy_ci = intervals.get("accuracy", {}) if isinstance(intervals, dict) else {}

        # Absolute error counts. Rates in this comparison all sit between 0.95
        # and 1.00, where they no longer discriminate; the counts do, and they
        # are what a clinician can reason about.
        confusion = metrics.confusion
        missed = int(confusion[1][0]) if len(confusion) == 2 else 0
        false_alarms = int(confusion[0][1]) if len(confusion) == 2 else 0

        rows.append(
            {
                "architecture": result.architecture,
                "missed": missed,
                "false_alarms": false_alarms,
                "accuracy": round(metrics.accuracy, 4),
                "accuracy_ci_low": round(accuracy_ci.get("lower", 0.0), 4),
                "accuracy_ci_high": round(accuracy_ci.get("upper", 0.0), 4),
                "balanced_accuracy": round(metrics.balanced_accuracy, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "specificity": round(metrics.specificity, 4),
                "f1": round(metrics.f1, 4),
                "auc_roc": round(metrics.auc_roc, 4),
                "threshold": round(result.decision_threshold, 3),
                "params_millions": round(int(summary.get("total_parameters", 0)) / 1e6, 2),
                "size_mb": summary.get("size_mb", 0.0),
                "cpu_latency_ms": latency.get("median_ms", float("nan")),
                "train_minutes": round(result.total_seconds / 60.0, 1),
                "meets_target": metrics.accuracy >= cfg.evaluation.target_accuracy,
            }
        )

    table = pd.DataFrame(rows)
    # Rank by recall first: on this task, sensitivity to abnormality is the
    # property that decides clinical usefulness, and AUC breaks ties in a
    # threshold-independent way.
    return table.sort_values(["recall", "auc_roc"], ascending=False).reset_index(drop=True)


#: Minimum test-set recall for an architecture to be considered deployable.
SELECTION_RECALL_FLOOR = 0.90

#: How much recall may be given up to gain a cheaper model, in absolute terms.
#: Deliberately small. An earlier version ranked eligible models by latency
#: alone and selected the *worst*-recall architecture - 17 missed abnormalities
#: against the best model's 6 - to save 360 ms per scan. In a district hospital
#: reading a handful of scans a day that latency is free; eleven missed tumours
#: are not.
RECALL_TOLERANCE = 0.02


def select_best_architecture(table: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Pick the model to deploy, and record why.

    Sensitivity leads. Among architectures that clear the accuracy target and
    the recall floor, the highest recall wins; efficiency only breaks ties
    between models whose recall is within :data:`RECALL_TOLERANCE` of the best.
    Efficiency matters for a CPU-only district hospital, but not enough to
    trade away detections for it.
    """
    if table.empty:
        return {"selected": None, "reason": "no results"}

    eligible = table[
        (table["accuracy"] >= cfg.evaluation.target_accuracy)
        & (table["recall"] >= SELECTION_RECALL_FLOOR)
    ]

    if eligible.empty:
        best = table.sort_values("recall", ascending=False).iloc[0]
        return {
            "selected": best["architecture"],
            "reason": (
                f"No architecture met the {cfg.evaluation.target_accuracy:.0%} accuracy "
                f"target with recall >= {SELECTION_RECALL_FLOOR:.0%}. Selected the "
                f"highest-recall model as provisional best."
            ),
            "constraint_met": False,
            "accuracy": float(best["accuracy"]),
            "recall": float(best["recall"]),
        }

    best_recall = float(eligible["recall"].max())
    contenders = eligible[eligible["recall"] >= best_recall - RECALL_TOLERANCE]

    # Within the recall band, rank on AUC before efficiency.
    #
    # AUC is threshold-independent, so it measures how separable the classes
    # are across every operating point rather than at the one we happened to
    # pick. Between two models that tie at the chosen threshold, the higher-AUC
    # model has more margin and degrades more gracefully under the distribution
    # shift this system will certainly meet - Br35H to Nepali clinical scanners.
    #
    # Latency is the last tiebreak, and deliberately so: it is a wall-clock
    # measurement of a shared machine. Measured idle, EfficientNetB0 takes
    # 47 ms; measured while a training job was running, 191 ms. Ranking on a
    # figure that moves 4x with unrelated system load would be ranking on noise.
    ranked = contenders.sort_values(
        ["auc_roc", "cpu_latency_ms", "params_millions"],
        ascending=[False, True, True],
        na_position="last",
    )
    best = ranked.iloc[0]

    traded = best_recall - float(best["recall"])
    efficiency_note = (
        f" It was preferred over the single highest-recall model on threshold-independent "
        f"separability (AUC {best['auc_roc']:.4f}), giving up {traded:.2%} recall - within "
        f"the {RECALL_TOLERANCE:.0%} tolerance - for a model that should degrade more "
        f"gracefully on unseen scanners."
        if traded > 1e-9
        else f" It has the highest recall of the {len(eligible)} eligible architecture(s), "
             f"with AUC {best['auc_roc']:.4f}."
    )

    return {
        "selected": best["architecture"],
        "reason": (
            f"Met the accuracy target ({best['accuracy']:.2%}) and the recall floor "
            f"(recall {best['recall']:.2%}, i.e. {best['recall']:.1%} of abnormal scans "
            f"flagged).{efficiency_note}"
        ),
        "constraint_met": True,
        "accuracy": float(best["accuracy"]),
        "recall": float(best["recall"]),
        "recall_traded": round(traded, 4),
        "auc_roc": float(best["auc_roc"]),
        "cpu_latency_ms": float(best["cpu_latency_ms"]),
        "n_eligible": len(eligible),
    }


def save_comparison(
    table: pd.DataFrame,
    results: list[TrainingResult],
    out_dir: Path,
    cfg: Config,
) -> dict[str, Path]:
    """Write the comparison table, selection rationale and plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    csv_path = out_dir / "architecture_comparison.csv"
    table.to_csv(csv_path, index=False)
    written["csv"] = csv_path

    # Formatting must never discard a completed training run. An earlier
    # version let a missing optional dependency (tabulate) propagate and lose
    # the results of three trained models.
    try:
        markdown_path = out_dir / "architecture_comparison.md"
        markdown_path.write_text(_to_markdown(table, cfg), encoding="utf-8")
        written["markdown"] = markdown_path
    except Exception as exc:
        log.warning("Could not write the markdown comparison (%s); CSV and JSON are intact", exc)

    selection = select_best_architecture(table, cfg)
    json_path = out_dir / "comparison.json"
    write_json(
        json_path,
        {
            "table": table.to_dict(orient="records"),
            "selection": selection,
            "target_accuracy": cfg.evaluation.target_accuracy,
            "dataset": cfg.dataset.name,
            "runs": {r.architecture: str(r.run_dir) for r in results},
        },
    )
    written["json"] = json_path

    try:
        from neuroscan.evaluation.plots import plot_comparison_bars, plot_roc_comparison

        written["roc"] = plot_roc_comparison(results, out_dir / "roc_comparison.png", cfg)
        written["bars"] = plot_comparison_bars(table, out_dir / "metric_comparison.png")
    except Exception as exc:
        log.warning("Comparison plots could not be generated: %s", exc)

    log.info("Comparison written to %s", out_dir)
    log.info("Selected architecture: %s - %s", selection.get("selected"), selection.get("reason"))
    return written


def _to_markdown(table: pd.DataFrame, cfg: Config) -> str:
    """Render the table as Markdown for direct inclusion in the final report."""
    selection = select_best_architecture(table, cfg)
    columns = [
        "architecture", "accuracy", "precision", "recall", "specificity",
        "f1", "auc_roc", "params_millions", "cpu_latency_ms", "train_minutes",
    ]
    display = table[[c for c in columns if c in table.columns]]

    try:
        rendered = display.to_markdown(index=False)
    except ImportError:
        # tabulate is optional; a fixed-width table is still readable.
        rendered = "```\n" + display.to_string(index=False) + "\n```"

    lines = [
        "# Architecture Comparison",
        "",
        f"Dataset: `{cfg.dataset.name}` | Target accuracy: {cfg.evaluation.target_accuracy:.0%} "
        f"| Image size: {cfg.preprocessing.image_size}px",
        "",
        "All three architectures were trained on an identical data pipeline, the same",
        "patient-grouped split, and the same seed. Latency is measured on CPU, single",
        "image, because the deployment target may have no GPU.",
        "",
        rendered,
        "",
        "## Selected model",
        "",
        f"**{selection.get('selected')}** - {selection.get('reason')}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "build_comparison_table",
    "measure_cpu_latency",
    "save_comparison",
    "select_best_architecture",
]
