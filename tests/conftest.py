"""Shared pytest fixtures.

Fixtures build everything in a temporary directory and avoid loading real
models, so the suite runs in seconds and needs no downloaded dataset. Tests
that genuinely require data or a model server are marked and deselected by
default.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from neuroscan.config import Config, load_config
from neuroscan.data.adapters import DatasetRecord


@pytest.fixture
def tmp_paths(tmp_path: Path) -> dict[str, Path]:
    """A complete temporary path layout."""
    layout = {
        name: tmp_path / name
        for name in [
            "data", "raw", "interim", "processed", "artifacts", "runs",
            "models", "index", "knowledge_base", "uploads", "reports",
        ]
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


@pytest.fixture
def cfg(tmp_paths: dict[str, Path]) -> Config:
    """A config pointed entirely at temporary directories."""
    return load_config(
        use_env=False,
        overrides={
            "paths": {
                "data_root": str(tmp_paths["data"]),
                "raw_dir": str(tmp_paths["raw"]),
                "interim_dir": str(tmp_paths["interim"]),
                "processed_dir": str(tmp_paths["processed"]),
                "artifacts_dir": str(tmp_paths["artifacts"]),
                "runs_dir": str(tmp_paths["runs"]),
                "models_dir": str(tmp_paths["models"]),
                "index_dir": str(tmp_paths["index"]),
                "knowledge_base_dir": str(tmp_paths["knowledge_base"]),
                "uploads_dir": str(tmp_paths["uploads"]),
                "reports_dir": str(tmp_paths["reports"]),
            },
            "training": {"num_workers": 0, "batch_size": 4, "pretrained": False},
            "split": {"detect_near_duplicates": False},
        },
    )


def synthetic_scan(
    *,
    lesion: bool = False,
    size: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic axial brain MRI.

    Dark background, bright elliptical brain, darker central ventricles, and
    optionally a bright focal lesion. Enough structure to exercise the
    preprocessing pipeline and the brain-plausibility heuristic without
    shipping real patient data in the repository.
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    centre = size // 2

    brain = ((yy - centre) ** 2 / (size * 0.37) ** 2
             + (xx - centre) ** 2 / (size * 0.31) ** 2) < 1
    image[brain] = 110

    ventricle = ((yy - centre * 0.94) ** 2 / (size * 0.085) ** 2
                 + (xx - centre) ** 2 / (size * 0.047) ** 2) < 1
    image[ventricle & brain] = 45

    if lesion:
        cy = int(rng.integers(int(size * 0.36), int(size * 0.64)))
        cx = int(rng.integers(int(size * 0.34), int(size * 0.68)))
        radius = int(size * 0.075)
        mass = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius**2
        image[mass & brain] = 225

    noisy = image.astype(np.int16) + rng.normal(0, 8, image.shape).astype(np.int16)
    gray = np.clip(noisy, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


@pytest.fixture
def scan_image() -> np.ndarray:
    return synthetic_scan(lesion=True, seed=1)


@pytest.fixture
def normal_scan_image() -> np.ndarray:
    return synthetic_scan(lesion=False, seed=2)


@pytest.fixture
def scan_file(tmp_path: Path) -> Path:
    """A single synthetic scan written to disk."""
    path = tmp_path / "scan.png"
    cv2.imwrite(str(path), cv2.cvtColor(synthetic_scan(lesion=True, seed=3), cv2.COLOR_RGB2BGR))
    return path


@pytest.fixture
def image_folder_dataset(tmp_paths: dict[str, Path]) -> Path:
    """An ImageFolder-layout dataset of synthetic scans."""
    root = tmp_paths["raw"] / "synthetic"
    for class_name, lesion in [("no", False), ("yes", True)]:
        folder = root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            image = synthetic_scan(lesion=lesion, seed=hash((class_name, i)) % 10000)
            cv2.imwrite(str(folder / f"P{i:03d}_s0.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return root


def make_records(
    n_patients: int = 20,
    slices_per_patient: int = 2,
    classes: tuple[str, str] = ("normal", "abnormal"),
) -> list[DatasetRecord]:
    """Build synthetic records without touching the filesystem."""
    records: list[DatasetRecord] = []
    for patient in range(n_patients):
        class_name = classes[patient % len(classes)]
        for slice_index in range(slices_per_patient):
            records.append(
                DatasetRecord(
                    path=Path(f"/synthetic/P{patient:04d}_s{slice_index}.png"),
                    class_name=class_name,
                    label=classes.index(class_name),
                    patient_id=f"P{patient:04d}",
                )
            )
    return records


@pytest.fixture
def records() -> list[DatasetRecord]:
    return make_records()
