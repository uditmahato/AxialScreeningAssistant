#!/usr/bin/env python
"""Evaluate the split-ablation checkpoints on decontaminated external data.

The manuscript's central claim is that split protocol barely matters on the
home benchmark because Br35H is close to saturated, and that the real damage
shows externally. The direct test, which the original ablation never ran, is
to take every ablation checkpoint (three protocols, five folds each) and
evaluate it zero-shot on the external sets after removing Br35H twins. If the
leak-free-trained models transfer no better, the saturation reading stands;
if they transfer better, contamination harmed the models themselves and the
paper's framing must change.

Also reports the removed-versus-retained contrast for the served checkpoint:
performance on the external images that have a Br35H twin against those that
do not, which is the direct evidence for or against the memorisation account
of the external specificity collapse.

Usage:
    python scripts/evaluate_ablation_external.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from neuroscan.config import load_config
from neuroscan.utils import get_logger, setup_logging, write_json

log = get_logger("scripts.evaluate_ablation_external")

EXTERNAL = ("brain_tumor_mri", "sartaj")
PROTOCOLS = ("random", "grouped", "grouped_dedup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score

    y_pred = (y_score[:, 1] >= threshold).astype(int)
    out: dict[str, object] = {"n": len(y_true)}
    if len(y_true) == 0:
        return out
    out["accuracy"] = round(float((y_pred == y_true).mean()), 4)
    positives = y_true == 1
    negatives = y_true == 0
    out["n_abnormal"] = int(positives.sum())
    out["n_normal"] = int(negatives.sum())
    out["recall"] = round(float((y_pred[positives] == 1).mean()), 4) if positives.any() else None
    out["specificity"] = (
        round(float((y_pred[negatives] == 0).mean()), 4) if negatives.any() else None
    )
    if out["recall"] is not None and out["specificity"] is not None:
        out["balanced_accuracy"] = round((out["recall"] + out["specificity"]) / 2, 4)
    if positives.any() and negatives.any():
        out["auc_roc"] = round(float(roc_auc_score(y_true, y_score[:, 1])), 4)
    return out


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    import torch
    from torch.utils.data import DataLoader

    from neuroscan.data.adapters import ImageFolderAdapter
    from neuroscan.data.datamodule import MRIDataset
    from neuroscan.data.dedup import compute_hashes
    from neuroscan.data.preprocessing import build_eval_transform
    from neuroscan.evaluation.integrity import min_hamming_to_reference
    from neuroscan.evaluation.metrics import predict
    from neuroscan.models.factory import find_best_checkpoint, load_checkpoint
    from neuroscan.utils import resolve_device

    cfg = load_config()
    out_dir = Path(args.out) if args.out else cfg.paths.artifacts_dir / "audit"
    device = resolve_device(cfg.training.device)
    transform = build_eval_transform(cfg.preprocessing)

    ablation = json.loads((out_dir / "split_ablation.json").read_text(encoding="utf-8"))

    dataset_cfg = cfg.dataset.model_copy(
        update={"adapter": "imagefolder", "patient_id_pattern": None}
    )

    br35h_records = ImageFolderAdapter(
        cfg.paths.raw_dir / "br35h", dataset_cfg, source="br35h"
    ).discover()
    br35h_hashes, _ = compute_hashes(br35h_records)
    log.info("Reference hashes: %d Br35H images", len(br35h_hashes))

    # Preprocess each external subset once; the in-memory cache then makes
    # the fifteen checkpoint evaluations GPU-bound rather than CLAHE-bound.
    subsets: dict[tuple[str, str], MRIDataset] = {}
    for key in EXTERNAL:
        records = ImageFolderAdapter(
            cfg.paths.raw_dir / key, dataset_cfg, source=key
        ).discover()
        hashes, valid = compute_hashes(records)
        minima = min_hamming_to_reference(hashes, br35h_hashes)
        twin_indices = {valid[i] for i in np.nonzero(minima <= 1)[0]}
        retained = [r for i, r in enumerate(records) if i not in twin_indices]
        removed = [r for i, r in enumerate(records) if i in twin_indices]
        subsets[(key, "retained")] = MRIDataset(retained, cfg, transform, cache=True)
        subsets[(key, "removed")] = MRIDataset(removed, cfg, transform, cache=True)
        log.info("%s: %d retained (decontaminated), %d removed (Br35H twins)",
                 key, len(retained), len(removed))

    def loader(dataset: MRIDataset) -> DataLoader:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def evaluate(model, threshold: float, dataset: MRIDataset) -> dict:
        y_true, y_score, _, _ = predict(model, loader(dataset), device, use_amp=False)
        return _metrics(y_true, y_score, threshold)

    report: dict[str, object] = {"decontamination_threshold": 1, "protocols": {}, "served": {}}

    # -- ablation checkpoints on the decontaminated (retained) subsets -----
    for protocol in PROTOCOLS:
        run_id = ablation["protocols"][protocol]["run_id"]
        per_fold: dict[str, list[dict]] = {k: [] for k in EXTERNAL}
        for fold in range(ablation["folds"]):
            ckpt = cfg.paths.runs_dir / run_id / f"fold_{fold}" / "best_efficientnet_b0.pt"
            if not ckpt.exists():
                log.warning("Missing checkpoint %s - skipping fold", ckpt)
                continue
            model, metadata = load_checkpoint(ckpt, device=device)
            extra = metadata.get("extra", {}) or {}
            threshold = float(extra.get("tuned_threshold")
                              or metadata.get("decision_threshold")
                              or cfg.evaluation.decision_threshold)
            for key in EXTERNAL:
                m = evaluate(model, threshold, subsets[(key, "retained")])
                m["fold"] = fold
                m["threshold"] = round(threshold, 4)
                per_fold[key].append(m)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            log.info("%s fold %d evaluated externally", protocol, fold)

        aggregate: dict[str, dict] = {}
        for key in EXTERNAL:
            rows = per_fold[key]
            aggregate[key] = {
                metric: {
                    "mean": round(float(np.mean([r[metric] for r in rows])), 4),
                    "std": round(float(np.std([r[metric] for r in rows], ddof=1)), 4)
                    if len(rows) > 1 else 0.0,
                }
                for metric in ("accuracy", "recall", "specificity", "balanced_accuracy", "auc_roc")
                if all(r.get(metric) is not None for r in rows)
            }
        report["protocols"][protocol] = {"aggregate": aggregate, "per_fold": per_fold}

    # -- served checkpoint: removed versus retained ------------------------
    served_path = find_best_checkpoint(cfg.paths.models_dir)
    model, metadata = load_checkpoint(served_path, device=device)
    extra = metadata.get("extra", {}) or {}
    served_threshold = float(extra.get("tuned_threshold")
                             or metadata.get("decision_threshold")
                             or cfg.evaluation.decision_threshold)
    for key in EXTERNAL:
        report["served"][key] = {
            "checkpoint": served_path.name,
            "threshold": round(served_threshold, 4),
            "removed_twins": evaluate(model, served_threshold, subsets[(key, "removed")]),
            "retained": evaluate(model, served_threshold, subsets[(key, "retained")]),
        }

    out_path = out_dir / "ablation_external.json"
    write_json(out_path, report)

    print("\n" + "=" * 88)
    print("ABLATION CHECKPOINTS, ZERO-SHOT ON DECONTAMINATED EXTERNAL DATA (mean over folds)")
    print("=" * 88)
    print(f"{'protocol':<16}{'dataset':<18}{'acc':>12}{'recall':>12}{'spec':>12}{'bal acc':>12}{'auc':>12}")
    for protocol in PROTOCOLS:
        for key in EXTERNAL:
            a = report["protocols"][protocol]["aggregate"].get(key, {})

            def cell(name, stats=a):
                s = stats.get(name)
                return f"{s['mean']*100:6.2f}{chr(177)}{s['std']*100:<4.2f}" if s else "   n/a  "
            print(f"{protocol:<16}{key:<18}{cell('accuracy'):>12}{cell('recall'):>12}"
                  f"{cell('specificity'):>12}{cell('balanced_accuracy'):>12}{cell('auc_roc'):>12}")

    print("\n" + "=" * 88)
    print("SERVED CHECKPOINT: REMOVED (Br35H twins) VERSUS RETAINED")
    print("=" * 88)
    print(f"{'dataset':<18}{'subset':<10}{'n':>7}{'n_norm':>8}{'acc':>8}{'recall':>8}{'spec':>8}{'auc':>8}")
    for key in EXTERNAL:
        for subset in ("removed_twins", "retained"):
            m = report["served"][key][subset]

            def fmt(v):
                return f"{v*100:7.2f}" if isinstance(v, float) else f"{'n/a':>7}"
            print(f"{key:<18}{subset:<10}{m['n']:>7}{m.get('n_normal',0):>8}"
                  f"{fmt(m.get('accuracy'))}{fmt(m.get('recall'))}"
                  f"{fmt(m.get('specificity'))}{fmt(m.get('auc_roc'))}")
    print(f"\nWritten to: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
