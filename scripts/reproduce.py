#!/usr/bin/env python
"""Reproduce the full pipeline end to end, and record the environment.

Runs every stage in order and writes a provenance record, so a reported number
can be traced to the exact code, data and hardware that produced it.

Usage:
    python scripts/reproduce.py                # full pipeline
    python scripts/reproduce.py --skip-train   # everything except training
    python scripts/reproduce.py --check        # verify prerequisites only
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroscan.config import load_config  # noqa: E402
from neuroscan.utils import format_duration, setup_logging, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the Axial Screening Assistant pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify prerequisites and exit")
    parser.add_argument("--config", default="efficientnet_b0")
    return parser.parse_args()


def environment_record() -> dict:
    """Capture everything needed to interpret a result later."""
    record: dict = {
        "captured_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    try:
        record["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
        record["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip())
    except Exception:
        record["git_commit"] = "unavailable"

    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
            record["cuda_version"] = torch.version.cuda
            record["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except ImportError:
        record["torch"] = "not installed"

    # Distribution name differs from import name for some packages, and
    # importlib.metadata is the supported route - Flask deprecated
    # __version__ entirely.
    distributions = {
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "opencv-python-headless": "cv2",
        "faiss-cpu": "faiss",
        "langchain": "langchain",
        "flask": "flask",
        "reportlab": "reportlab",
        "sentence-transformers": "sentence_transformers",
    }
    from importlib.metadata import PackageNotFoundError, version

    for distribution, import_name in distributions.items():
        try:
            record[import_name] = version(distribution)
        except PackageNotFoundError:
            record[import_name] = "not installed"

    return record


def check_prerequisites(cfg) -> tuple[bool, list[str]]:
    """Report what is present and what is missing."""
    problems: list[str] = []

    print("\n" + "=" * 74)
    print("PREREQUISITES")
    print("=" * 74)

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  torch          : {torch.__version__} (CUDA {'yes' if cuda else 'NO - CPU only'})")
        if not cuda:
            print("                   Training will be slow. See docs/SETUP.md step 2.")
    except ImportError:
        problems.append("PyTorch is not installed")
        print("  torch          : MISSING")

    dataset_dir = cfg.paths.raw_dir / cfg.dataset.name
    if dataset_dir.exists() and any(dataset_dir.rglob("*.jpg")):
        count = sum(1 for _ in dataset_dir.rglob("*.jpg"))
        print(f"  dataset        : {count:,} images at {dataset_dir}")
    else:
        problems.append(f"dataset missing at {dataset_dir}")
        print(f"  dataset        : MISSING ({dataset_dir})")

    kb = cfg.paths.knowledge_base_dir / "medical"
    if kb.exists():
        docs = sum(1 for _ in kb.rglob("*.md"))
        print(f"  knowledge base : {docs} medical documents")
        if docs < cfg.rag.min_corpus_documents:
            problems.append(f"corpus has {docs} documents, the corpus requirement is 50+")
    else:
        problems.append("knowledge base missing")
        print("  knowledge base : MISSING")

    if cfg.paths.index_dir.exists() and any(cfg.paths.index_dir.iterdir()):
        print(f"  FAISS index    : present at {cfg.paths.index_dir}")
    else:
        print("  FAISS index    : not built")

    try:
        import httpx

        response = httpx.get(f"{cfg.llm.base_url}/api/tags", timeout=2.0)
        models = [m.get("name") for m in response.json().get("models", [])]
        print(f"  Ollama         : running, {len(models)} model(s)")
    except Exception:
        print("  Ollama         : not reachable (advisory will show source text directly)")

    return not problems, problems


def run(label: str, command: list[str]) -> tuple[bool, float]:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    started = time.perf_counter()
    result = subprocess.run([sys.executable, *command], cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started
    ok = result.returncode == 0
    print(f"\n  -> {'OK' if ok else f'FAILED (exit {result.returncode})'} "
          f"in {format_duration(elapsed)}")
    return ok, elapsed


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    cfg = load_config(args.config)
    cfg.paths.ensure_all()

    print("\n" + "#" * 74)
    print("#  Axial Screening Assistant - full pipeline reproduction")
    print("#" * 74)

    env = environment_record()
    print(f"\n  commit  : {env.get('git_commit', '?')[:12]}"
          f"{' (uncommitted changes present)' if env.get('git_dirty') else ''}")
    print(f"  python  : {env['python']}   torch: {env.get('torch')}")
    print(f"  gpu     : {env.get('gpu', 'none')}")

    ok, problems = check_prerequisites(cfg)
    if args.check:
        if problems:
            print("\n  Problems:")
            for problem in problems:
                print(f"    - {problem}")
        print()
        return 0 if ok else 1

    stages: list[tuple[str, list[str]]] = []
    if not args.skip_download:
        stages.append(("1. Download public dataset",
                       ["scripts/download_data.py", "--dataset", "br35h"]))
    if not args.skip_index:
        stages.append(("2. Build FAISS index", ["scripts/build_index.py", "--rebuild"]))
    if not args.skip_tests:
        stages.append(("3. Test suite", ["-m", "pytest", "tests/", "-q", "--no-header"]))
    if not args.skip_train:
        stages.append(("4. Train and compare architectures",
                       ["scripts/train.py", "--compare-all"]))

    results = {}
    total = 0.0
    for label, command in stages:
        succeeded, elapsed = run(label, command)
        results[label] = {"ok": succeeded, "seconds": round(elapsed, 1)}
        total += elapsed
        if not succeeded and "Test suite" not in label:
            print(f"\nStopping: {label} failed.")
            break

    provenance = {
        "environment": env,
        "config": args.config,
        "stages": results,
        "total_seconds": round(total, 1),
    }
    out_path = cfg.paths.artifacts_dir / "reproduction_record.json"
    write_json(out_path, provenance)

    print("\n" + "#" * 74)
    print(f"#  Complete in {format_duration(total)}")
    print("#" * 74)
    for label, outcome in results.items():
        print(f"  {'OK  ' if outcome['ok'] else 'FAIL'}  {label:<44} "
              f"{format_duration(outcome['seconds'])}")
    print(f"\n  Provenance record: {out_path}")
    print(f"  Comparison table : {cfg.paths.artifacts_dir / 'comparison'}")
    print("\n  Next: python scripts/run_app.py\n")

    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
