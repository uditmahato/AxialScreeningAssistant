"""Import every public module.

Cheap, and it catches the class of error that only appears at import time -
a dataclass field ordering mistake, a circular import, a typo in an
``__all__``. One such bug reached a training run because no test in the suite
imported ``neuroscan.training`` at all.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "neuroscan",
    "neuroscan.config",
    "neuroscan.safety",
    "neuroscan.utils",
    "neuroscan.data",
    "neuroscan.data.adapters",
    "neuroscan.data.augment",
    "neuroscan.data.datamodule",
    "neuroscan.data.dedup",
    "neuroscan.data.download",
    "neuroscan.data.preprocessing",
    "neuroscan.data.splits",
    "neuroscan.models",
    "neuroscan.models.base",
    "neuroscan.models.baseline_cnn",
    "neuroscan.models.factory",
    "neuroscan.models.transfer",
    "neuroscan.training",
    "neuroscan.training.callbacks",
    "neuroscan.training.trainer",
    "neuroscan.evaluation",
    "neuroscan.evaluation.compare",
    "neuroscan.evaluation.metrics",
    "neuroscan.evaluation.plots",
    "neuroscan.evaluation.usability",
    "neuroscan.explain",
    "neuroscan.explain.gradcam",
    "neuroscan.rag",
    "neuroscan.rag.advisory",
    "neuroscan.rag.corpus",
    "neuroscan.rag.llm_provider",
    "neuroscan.rag.prompts",
    "neuroscan.rag.vectorstore",
    "neuroscan.chatbot",
    "neuroscan.chatbot.engine",
    "neuroscan.chatbot.language",
    "neuroscan.reporting",
    "neuroscan.reporting.devanagari",
    "neuroscan.reporting.pdf_report",
    "neuroscan.web",
    "neuroscan.web.app",
    "neuroscan.web.services",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", MODULES)
def test_all_exports_resolve(module_name):
    """Every name in ``__all__`` must actually exist."""
    module = importlib.import_module(module_name)
    for name in getattr(module, "__all__", []):
        assert hasattr(module, name), f"{module_name}.__all__ names missing {name!r}"


def test_epoch_record_constructs():
    """Regression: a defaulted field was inserted before non-default ones,
    which fails at class-definition time."""
    from neuroscan.training.callbacks import EpochRecord

    record = EpochRecord(
        epoch=1, stage="head", train_loss=0.5, val_loss=0.4, val_accuracy=0.9,
        val_recall=0.9, val_f1=0.9, val_auc=0.95, learning_rate=1e-3,
        duration_seconds=1.0,
    )
    assert record.stage_epoch == 0
    assert record.is_best is False


def test_scripts_are_importable():
    """The CLI entry points must at least parse and import cleanly."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for script in ["train.py", "build_index.py", "run_app.py",
                   "download_data.py", "analyse_usability.py"]:
        result = subprocess.run(
            [sys.executable, "-c",
             f"import ast,pathlib; ast.parse(pathlib.Path(r'{root / 'scripts' / script}').read_text('utf-8'))"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{script} failed to parse: {result.stderr}"
