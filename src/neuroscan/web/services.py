"""Inference orchestration for the web layer.

Everything expensive - the CNN, the embedding model, the FAISS index - is
loaded once at application start-up and shared. Loading per request would add
seconds of latency and, on an 8GB GPU, would eventually exhaust VRAM.

The service is built to **start successfully even when parts are missing**. A
health worker whose Ollama server is down should still get a classification and
a Grad-CAM heatmap, with the advisory degraded to verbatim source text, rather
than a page that will not load. Each component reports its own readiness.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config
    from neuroscan.models.base import BaseClassifier
    from neuroscan.safety import Language

log = get_logger("web.services")


@dataclass
class AnalysisResult:
    """The outcome of analysing one uploaded scan."""

    analysis_id: str
    prediction: str
    prediction_index: int
    confidence: float
    probabilities: dict[str, float]
    threshold: float
    architecture: str
    scan_image_url: str = ""
    heatmap_image_url: str = ""
    scan_image_path: Path | None = None
    heatmap_image_path: Path | None = None
    heatmap_focus_ratio: float = 0.0
    heatmap_is_diffuse: bool = False
    #: The Grad-CAM came back with no activation at all - the known failure
    #: mode, not a finding. The interface shows "no map could be generated"
    #: instead of a black square the reader cannot interpret.
    heatmap_failed: bool = False
    heatmap_peak: tuple[float, float] = (0.5, 0.5)
    inference_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    advisory_text: str = ""
    #: Structured advisory when the model returned valid, safe JSON:
    #: {"summary", "possible_causes": [{"name","note"}], "next_steps": [...]}.
    #: None when generation was degraded; the interface then falls back to
    #: rendering advisory_text.
    advisory_structured: dict | None = None
    advisory_citations: list[str] = field(default_factory=list)
    advisory_degraded: bool = False
    red_flag_instruction: str = ""
    red_flags: list[str] = field(default_factory=list)

    @property
    def is_abnormal(self) -> bool:
        return self.prediction.lower() != "normal"

    @property
    def scan_ref(self) -> str:
        """Human-facing scan identifier, e.g. ``NS-4F2A1C``.

        Derived from the analysis id rather than anything patient-linked, so
        it is safe to print on reports and read out over the phone.
        """
        return f"NS-{self.analysis_id[:6].upper()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "threshold": round(self.threshold, 4),
            "architecture": self.architecture,
            "scan_image_url": self.scan_image_url,
            "heatmap_image_url": self.heatmap_image_url,
            "heatmap_focus_ratio": round(self.heatmap_focus_ratio, 4),
            "heatmap_is_diffuse": self.heatmap_is_diffuse,
            "heatmap_failed": self.heatmap_failed,
            "inference_ms": round(self.inference_ms, 1),
            "advisory_text": self.advisory_text,
            "advisory_structured": self.advisory_structured,
            "scan_ref": self.scan_ref,
            "advisory_citations": self.advisory_citations,
            "advisory_degraded": self.advisory_degraded,
            "red_flags": self.red_flags,
            "created_at": self.created_at.isoformat(),
        }


class ServiceError(RuntimeError):
    """Raised when an operation cannot be completed."""


class ModelUnavailableError(ServiceError):
    """Raised when no trained classifier is present."""


class NotABrainScanError(ServiceError):
    """Raised when an upload does not resemble a brain MRI."""


class InferenceService:
    """Holds the loaded model and RAG components for the lifetime of the app."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model: BaseClassifier | None = None
        self.model_metadata: dict[str, Any] = {}
        self.advisory_engine = None
        self.chatbot_engine = None
        self.device = None
        self.threshold: float = cfg.evaluation.decision_threshold

        self._status: dict[str, Any] = {
            "model": False, "rag": False, "llm": False, "llm_provider": None, "errors": [],
        }
        # Grad-CAM registers backward hooks and runs a backward pass, so the
        # model cannot safely serve two requests at once.
        self._inference_lock = threading.Lock()

        self._load_model()
        self._load_rag()
        self.cv_stats: dict[str, Any] | None = self._load_cv_stats()

    # -- start-up ----------------------------------------------------------

    def _load_model(self) -> None:
        from neuroscan.models.factory import find_best_checkpoint, load_checkpoint
        from neuroscan.utils import resolve_device

        checkpoint = find_best_checkpoint(self.cfg.paths.models_dir)
        if checkpoint is None:
            # Training writes into runs/<run_id>/; fall back to the newest.
            candidates = sorted(
                self.cfg.paths.runs_dir.glob("**/best_*.pt"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            checkpoint = candidates[0] if candidates else None

        if checkpoint is None:
            message = (
                f"No trained model found in {self.cfg.paths.models_dir} or "
                f"{self.cfg.paths.runs_dir}. Train one with:\n"
                f"  python scripts/train.py --config efficientnet_b0"
            )
            log.error(message)
            self._status["errors"].append(message)
            return

        try:
            self.device = resolve_device(self.cfg.training.device)
            self.model, self.model_metadata = load_checkpoint(checkpoint, device=self.device)
            # Prefer the threshold the model was actually evaluated at over the
            # config default, so the served operating point matches the
            # reported metrics.
            extra = self.model_metadata.get("extra", {}) or {}
            self.threshold = float(
                extra.get("tuned_threshold")
                or self.model_metadata.get("decision_threshold")
                or self.cfg.evaluation.decision_threshold
            )
            self._status["model"] = True
            log.info(
                "Model ready: %s from %s (threshold %.3f)",
                self.model.architecture_name, checkpoint.name, self.threshold,
            )
        except Exception as exc:
            message = f"Failed to load model from {checkpoint}: {exc}"
            log.error(message)
            self._status["errors"].append(message)

    def _load_rag(self) -> None:
        from neuroscan.chatbot.engine import ChatbotEngine
        from neuroscan.rag.advisory import AdvisoryEngine
        from neuroscan.rag.llm_provider import build_llm_provider
        from neuroscan.rag.vectorstore import load_index

        try:
            store = load_index(self.cfg)
        except Exception as exc:
            message = (
                f"FAISS index unavailable: {exc}\n"
                f"Build it with: python scripts/build_index.py --rebuild"
            )
            log.error(message)
            self._status["errors"].append(message)
            return

        llm = build_llm_provider(self.cfg, allow_fallback=True)
        self.advisory_engine = AdvisoryEngine(store, llm, self.cfg)
        self.chatbot_engine = ChatbotEngine(store, llm, self.cfg)

        self._status["rag"] = True
        self._status["llm"] = llm.is_generative
        self._status["llm_provider"] = llm.name
        log.info("RAG ready: %d vectors, LLM provider %s", store.size, llm.name)

    def _load_cv_stats(self) -> dict[str, Any] | None:
        """Load the newest cross-validation aggregate for the About page.

        The cross-validated mean with a standard deviation is the number the
        project quotes; showing it in the interface keeps the About page
        honest without hand-copying figures that would go stale.
        """
        from neuroscan.utils import read_json

        candidates = sorted(
            self.cfg.paths.runs_dir.glob("*cv_*/cross_validation.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                payload = read_json(path)
                if payload.get("aggregate"):
                    log.info("Loaded cross-validation stats from %s", path)
                    return payload
            except Exception as exc:
                log.debug("Unreadable cross-validation file %s: %s", path, exc)
        return None

    @property
    def status(self) -> dict[str, Any]:
        return dict(self._status)

    @property
    def is_ready(self) -> bool:
        """Whether the core classification path works. RAG is optional."""
        return self._status["model"]

    # -- inference ---------------------------------------------------------

    def analyse(
        self,
        image_path: Path,
        *,
        analysis_id: str | None = None,
        check_is_brain: bool = True,
    ) -> AnalysisResult:
        """Classify an uploaded scan and generate its Grad-CAM explanation.

        Raises:
            ModelUnavailableError: If no classifier is loaded.
            NotABrainScanError: If the image does not resemble a brain MRI.
        """
        if self.model is None:
            raise ModelUnavailableError("No trained model is loaded")

        import torch

        from neuroscan.data.preprocessing import (
            is_plausible_brain_scan,
            load_image,
            preprocess_for_inference,
        )
        from neuroscan.explain.gradcam import explain_prediction

        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()

        # Screen the RAW upload, not the preprocessed version. The question is
        # "did the user give us a brain MRI", which is a property of what they
        # uploaded - and crop_black_border removes the dark margin the check
        # partly relies on, so screening the cropped image tested for a feature
        # the pipeline had already deleted. The extra decode costs a few
        # milliseconds against a multi-hundred-millisecond inference.
        if check_is_brain and not is_plausible_brain_scan(load_image(image_path)):
            raise NotABrainScanError(
                "The uploaded image does not appear to be a brain MRI scan."
            )

        tensor, display = preprocess_for_inference(image_path, self.cfg.preprocessing)

        class_names = list(self.model.class_names)

        with self._inference_lock:
            device = self.device or torch.device("cpu")
            with torch.no_grad():
                probabilities = self.model.predict_proba(tensor.to(device))[0].cpu().numpy()

            # Threshold the positive class rather than taking argmax, so the
            # served operating point matches the one selected on validation.
            if len(class_names) == 2:
                index = 1 if float(probabilities[1]) >= self.threshold else 0
            else:
                index = int(np.argmax(probabilities))

            explanation = explain_prediction(
                self.model, tensor, display, target_class=index
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        assets = self.cfg.paths.uploads_dir / analysis_id
        assets.mkdir(parents=True, exist_ok=True)

        scan_path = assets / "scan.png"
        heatmap_path = assets / "heatmap.png"
        cv2.imwrite(str(scan_path), cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(explanation.overlay, cv2.COLOR_RGB2BGR))

        result = AnalysisResult(
            analysis_id=analysis_id,
            prediction=class_names[index],
            prediction_index=index,
            confidence=float(probabilities[index]),
            probabilities={n: float(p) for n, p in zip(class_names, probabilities, strict=False)},
            threshold=self.threshold,
            architecture=self.model.architecture_name,
            scan_image_url=f"/media/{analysis_id}/scan.png",
            heatmap_image_url=f"/media/{analysis_id}/heatmap.png",
            scan_image_path=scan_path,
            heatmap_image_path=heatmap_path,
            heatmap_focus_ratio=explanation.focus_ratio,
            heatmap_is_diffuse=explanation.is_diffuse(),
            heatmap_failed=explanation.is_blank,
            heatmap_peak=explanation.peak_location,
            inference_ms=elapsed_ms,
        )

        log.info(
            "Analysis %s: %s (%.1f%%) in %.0f ms, focus=%.2f",
            analysis_id, result.prediction, result.confidence * 100,
            elapsed_ms, explanation.focus_ratio,
        )
        return result

    def attach_advisory(self, result: AnalysisResult, language: Language = "en") -> AnalysisResult:
        """Generate and attach the RAG advisory for a completed analysis."""
        if self.advisory_engine is None:
            log.warning("Advisory engine unavailable - returning result without advice")
            return result

        if result.heatmap_failed:
            note = "could not be generated for this scan"
        elif result.heatmap_is_diffuse:
            note = "attention was diffuse, not localised to one region"
        else:
            note = (
                f"focused region at approximately x={result.heatmap_peak[0]:.2f}, "
                f"y={result.heatmap_peak[1]:.2f} of the image"
            )

        advisory = self.advisory_engine.generate(
            prediction=result.prediction,
            confidence=result.confidence,
            architecture=result.architecture,
            language=language,
            heatmap_note=note,
        )

        result.advisory_text = advisory.text
        result.advisory_structured = advisory.structured
        result.advisory_citations = advisory.citations
        result.advisory_degraded = advisory.degraded
        result.red_flag_instruction = advisory.red_flag_instruction
        result.red_flags = advisory.red_flags
        return result

    def ask(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        language: Language | None = None,
        scan_prediction: str | None = None,
        scan_confidence: float | None = None,
    ):
        """Answer a chatbot question."""
        if self.chatbot_engine is None:
            raise ServiceError(
                "The chatbot is unavailable because the knowledge base index could not "
                "be loaded. Run: python scripts/build_index.py --rebuild"
            )

        from neuroscan.chatbot.engine import ChatTurn

        turns = [
            ChatTurn(question=h.get("question", ""), answer=h.get("answer", ""))
            for h in (history or [])
            if h.get("question")
        ]

        return self.chatbot_engine.ask(
            question,
            history=turns,
            language=language,
            scan_prediction=scan_prediction,
            scan_confidence=scan_confidence,
        )

    # -- housekeeping ------------------------------------------------------

    def purge_old_uploads(self) -> int:
        """Delete upload directories older than the configured retention.

        Required by the ethics commitment that scans are not retained
        (docs/ETHICS.md). Called opportunistically on upload rather than from a
        scheduler, which keeps the deployment a single process with no cron
        dependency - appropriate for an offline clinic machine.
        """
        import shutil

        cutoff = time.time() - self.cfg.web.retain_uploads_hours * 3600
        removed = 0

        uploads = self.cfg.paths.uploads_dir
        if not uploads.exists():
            return 0

        for entry in uploads.iterdir():
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError as exc:
                log.debug("Could not purge %s: %s", entry, exc)

        if removed:
            log.info("Purged %d upload director(ies) older than %d hours",
                     removed, self.cfg.web.retain_uploads_hours)
        return removed


__all__ = [
    "AnalysisResult",
    "InferenceService",
    "ModelUnavailableError",
    "NotABrainScanError",
    "ServiceError",
]
