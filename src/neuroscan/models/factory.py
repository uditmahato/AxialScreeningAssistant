"""Model construction, checkpointing and restoration.

Checkpoints are self-describing: each one carries the architecture name, class
names, preprocessing parameters and the metrics it achieved. The web
application can therefore load a ``.pt`` file and configure its whole inference
path from it, with no risk of serving a model under preprocessing settings it
was never trained with - a mismatch that degrades accuracy silently rather than
raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from neuroscan.models.base import BaseClassifier
from neuroscan.models.baseline_cnn import BaselineCNN
from neuroscan.models.transfer import EfficientNetB0Classifier, VGG16Classifier
from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config

log = get_logger("models.factory")

CHECKPOINT_VERSION = 1

ARCHITECTURES: dict[str, type[BaseClassifier]] = {
    "baseline_cnn": BaselineCNN,
    "vgg16": VGG16Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
}


class ModelError(RuntimeError):
    """Raised when a model cannot be built or a checkpoint cannot be restored."""


def build_model(cfg: Config, *, architecture: str | None = None) -> BaseClassifier:
    """Instantiate the configured architecture.

    Args:
        cfg: Supplies class names, dropout and the pretrained flag.
        architecture: Overrides ``cfg.training.architecture``. Used by the
            comparison script to build all three from one config.

    Raises:
        ModelError: If the architecture name is not registered.
    """
    name = architecture or cfg.training.architecture
    model_cls = ARCHITECTURES.get(name)
    if model_cls is None:
        raise ModelError(f"Unknown architecture {name!r}. Available: {sorted(ARCHITECTURES)}")

    kwargs: dict[str, Any] = {
        "num_classes": cfg.dataset.num_classes,
        "class_names": list(cfg.dataset.class_names),
        "dropout": cfg.training.dropout,
    }
    # The scratch baseline has no ImageNet weights to load.
    if name != "baseline_cnn":
        kwargs["pretrained"] = cfg.training.pretrained

    model = model_cls(**kwargs)
    summary = model.describe()
    log.info(
        "Built %s | %s parameters (%s trainable) | %.1f MB",
        summary["architecture"],
        f"{summary['total_parameters']:,}",
        f"{summary['trainable_parameters']:,}",
        summary["size_mb"],
    )
    return model


def save_checkpoint(
    path: Path,
    model: BaseClassifier,
    cfg: Config,
    *,
    metrics: dict[str, float] | None = None,
    epoch: int | None = None,
    optimizer_state: dict | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a self-describing checkpoint.

    ``optimizer_state`` is stored only for resumable mid-training checkpoints;
    the exported best model omits it, which typically halves the file size.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": model.architecture_name,
        "class_names": list(model.class_names),
        "num_classes": model.num_classes,
        "state_dict": model.state_dict(),
        "model_summary": model.describe(),
        # Inference must reproduce training-time preprocessing exactly.
        "preprocessing": cfg.preprocessing.model_dump(mode="json"),
        "training_config": cfg.training.model_dump(mode="json"),
        "dataset_config": cfg.dataset.model_dump(mode="json"),
        "decision_threshold": cfg.evaluation.decision_threshold,
        "metrics": metrics or {},
        "epoch": epoch,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)
    log.info("Checkpoint saved: %s (%.1f MB)", path, path.stat().st_size / 1024**2)
    return path


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> tuple[BaseClassifier, dict[str, Any]]:
    """Restore a model and its metadata from a checkpoint.

    Args:
        path: Checkpoint file.
        device: Where to map the weights.
        strict: Require an exact state-dict match.

    Returns:
        ``(model, metadata)``. ``metadata`` holds everything except the raw
        tensors - notably ``preprocessing``, which the caller must use to
        rebuild the inference transform.

    Raises:
        ModelError: If the file is missing, malformed, or names an unknown
            architecture.
    """
    path = Path(path)
    if not path.exists():
        raise ModelError(f"Checkpoint not found: {path}")

    try:
        # weights_only=False is required because the payload carries config
        # dicts alongside tensors. Only load checkpoints this project produced.
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ModelError(f"Failed to read checkpoint {path}: {exc}") from exc

    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ModelError(f"{path} is not an Axial Screening Assistant checkpoint (no 'state_dict' key)")

    version = payload.get("checkpoint_version", 0)
    if version > CHECKPOINT_VERSION:
        log.warning(
            "Checkpoint version %s is newer than this build supports (%s); "
            "loading anyway, but fields may be missing.",
            version,
            CHECKPOINT_VERSION,
        )

    architecture = payload.get("architecture")
    model_cls = ARCHITECTURES.get(architecture or "")
    if model_cls is None:
        raise ModelError(
            f"Checkpoint names unknown architecture {architecture!r}. "
            f"Available: {sorted(ARCHITECTURES)}"
        )

    class_names = payload.get("class_names") or []
    num_classes = payload.get("num_classes") or len(class_names) or 2

    kwargs: dict[str, Any] = {"num_classes": num_classes, "class_names": class_names}
    if architecture != "baseline_cnn":
        # Skip the ImageNet download; the checkpoint overwrites these weights.
        kwargs["pretrained"] = False

    model = model_cls(**kwargs)

    try:
        model.load_state_dict(payload["state_dict"], strict=strict)
    except RuntimeError as exc:
        raise ModelError(
            f"State dict does not match the {architecture} architecture: {exc}\n"
            f"The checkpoint may have been written by a different model version."
        ) from exc

    model.to(device)
    model.eval()

    metadata = {k: v for k, v in payload.items() if k not in {"state_dict", "optimizer_state"}}
    log.info(
        "Loaded %s checkpoint from %s (classes=%s, epoch=%s)",
        architecture,
        path.name,
        class_names,
        payload.get("epoch"),
    )
    return model, metadata


def find_best_checkpoint(models_dir: Path, architecture: str | None = None) -> Path | None:
    """Locate the most recent ``best_*.pt`` checkpoint.

    Used by the Flask application at start-up so a demo does not require a
    hard-coded path.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None

    pattern = f"best_{architecture}*.pt" if architecture else "best_*.pt"
    candidates = sorted(models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


__all__ = [
    "ARCHITECTURES",
    "CHECKPOINT_VERSION",
    "ModelError",
    "build_model",
    "find_best_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
]
