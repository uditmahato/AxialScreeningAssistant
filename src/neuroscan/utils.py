"""Cross-cutting helpers: logging, seeding, device selection, run identifiers.

Kept dependency-light on purpose - ``torch`` is imported lazily inside the
functions that need it so that the web layer, the RAG stack and the test suite
can import this module without paying for a CUDA context.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | None = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure root logging once, optionally teeing to a file.

    Args:
        level: Log level for the system's own loggers.
        log_file: When given, logs are also written here at DEBUG level so a
            failed training run leaves a full trace on disk.
        quiet_libraries: Suppress the noisy INFO chatter from transformers,
            sentence-transformers, urllib3 and matplotlib's font manager.

    Returns:
        The ``neuroscan`` package logger.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if quiet_libraries:
        for noisy in (
            "matplotlib",
            "matplotlib.font_manager",
            "PIL",
            "urllib3",
            "transformers",
            "sentence_transformers",
            "httpx",
            "faiss",
            "filelock",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("neuroscan")


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``neuroscan``."""
    if name.startswith("neuroscan"):
        return logging.getLogger(name)
    return logging.getLogger(f"neuroscan.{name}")


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG that affects a training run.

    ``deterministic`` trades throughput for exact reproducibility. It is on by
    default because a clinical prototype has to be able to defend its numbers;
    turn it off only when benchmarking speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # cuBLAS needs this to make matmuls reproducible on CUDA >= 10.2.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:  # pragma: no cover
        pass


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device string to a concrete :class:`torch.device`.

    Falls back to CPU with a warning rather than raising, so that a machine
    without CUDA can still run inference and the web application.
    """
    import torch

    log = get_logger("utils")

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "The installed torch build is probably CPU-only - reinstall with:\n"
                "  pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu126"
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    log.warning("CUDA unavailable - falling back to CPU. Training will be substantially slower.")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device summary for run logs and provenance records."""
    import torch

    if device.type != "cuda":
        return "CPU"
    index = device.index or 0
    name = torch.cuda.get_device_name(index)
    total_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
    return f"{name} ({total_gb:.1f} GB VRAM, CUDA {torch.version.cuda})"


def count_parameters(model: Any) -> tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """UTC timestamp used to name run directories."""
    return datetime.now(UTC).strftime(fmt)


def make_run_id(architecture: str, dataset: str) -> str:
    """Build a collision-resistant, self-describing run identifier."""
    return f"{timestamp()}_{architecture}_{dataset}"


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON with UTF-8 encoding, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def format_duration(seconds: float) -> str:
    """Format a duration as ``1h 04m 09s`` / ``4m 09s`` / ``9.4s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


__all__ = [
    "count_parameters",
    "describe_device",
    "format_duration",
    "get_logger",
    "make_run_id",
    "read_json",
    "resolve_device",
    "set_seed",
    "setup_logging",
    "timestamp",
    "write_json",
]
