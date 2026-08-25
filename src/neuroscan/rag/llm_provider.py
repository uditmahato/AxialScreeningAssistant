"""Pluggable LLM backends, local-first.

Ollama is the default so the system runs fully offline in a clinic with no
internet and no patient data leaving the building - the privacy and
availability argument made in Project Design 4.2 ("running a local LLM to serve
offline, save money, and ensure privacy of medical information").

The ``echo`` provider is not a toy. It lets the entire retrieval pipeline, the
web application and the test suite run deterministically with no model server
present, which is what makes the RAG layer testable in CI.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config, LLMConfig

log = get_logger("rag.llm")


class LLMError(RuntimeError):
    """Raised when a provider cannot be reached or fails to generate."""


@dataclass
class LLMResponse:
    """A generated response with the provenance needed for the audit trail."""

    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float = 0.0
    degraded: bool = False  # True when a fallback produced this


class LLMProvider(ABC):
    """Interface every backend implements."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def is_generative(self) -> bool:
        """Whether this backend actually synthesises new text.

        Callers use this to decide two things. First, whether to run the
        output through :func:`validate_generated_text` - the prohibited-pattern
        screen is designed for model output, and running it over verbatim
        corpus text produces false positives on legitimate clinical phrasing
        such as "the diagnosis is usually built from". Second, whether to
        bother constructing a generation prompt at all.
        """
        return True

    @abstractmethod
    def generate(
        self, system_prompt: str, user_prompt: str, *, json_mode: bool = False
    ) -> LLMResponse:
        """Generate a reply.

        ``json_mode`` asks the backend to constrain output to valid JSON where
        it supports that natively (Ollama does). Backends without native
        support ignore it; callers must still parse defensively.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the backend can currently serve a request.

        Checked at start-up so the web application can warn about a missing
        model server rather than failing on the first user request.
        """


class OllamaProvider(LLMProvider):
    """Local Ollama server - the intended production backend."""

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        try:
            import httpx

            response = httpx.get(f"{self.cfg.base_url}/api/tags", timeout=3.0)
            if response.status_code != 200:
                return False
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:
            log.debug("Ollama availability check failed: %s", exc)
            return False

        if not models:
            log.warning("Ollama is running but has no models. Pull one:\n  ollama pull %s",
                        self.cfg.model)
            return False

        # Ollama tags carry a version suffix ("llama3.1:8b"); match on the stem
        # so a configured "llama3.1" still matches "llama3.1:latest".
        stem = self.cfg.model.split(":")[0]
        if not any(m.split(":")[0] == stem for m in models):
            log.warning(
                "Model %r is not present in Ollama. Available: %s. Pull it with:\n"
                "  ollama pull %s",
                self.cfg.model, models, self.cfg.model,
            )
            return False
        return True

    def generate(
        self, system_prompt: str, user_prompt: str, *, json_mode: bool = False
    ) -> LLMResponse:
        import time

        started = time.perf_counter()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=self.cfg.model,
                base_url=self.cfg.base_url,
                temperature=self.cfg.temperature,
                num_predict=self.cfg.max_tokens,
                num_ctx=self.cfg.num_ctx,
                timeout=self.cfg.timeout_seconds,
                # Native JSON constraint: far more reliable than asking nicely
                # in the prompt, though callers still parse defensively.
                format="json" if json_mode else None,
            )
            result = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            text = result.content if isinstance(result.content, str) else str(result.content)
        except Exception as exc:
            raise LLMError(
                f"Ollama generation failed: {exc}\n"
                f"Check the server is running (ollama serve) and the model is pulled "
                f"(ollama pull {self.cfg.model})."
            ) from exc

        return LLMResponse(
            text=text.strip(),
            provider=self.name,
            model=self.cfg.model,
            latency_seconds=time.perf_counter() - started,
        )


class HuggingFaceProvider(LLMProvider):
    """In-process transformers backend.

    Heavier than Ollama and competes with the CNN for VRAM, so it is a fallback
    rather than the default. The pipeline is cached across calls because model
    loading dominates the cost.
    """

    _pipeline = None  # class-level cache

    @property
    def name(self) -> str:
        return "huggingface"

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_pipeline(self):
        if HuggingFaceProvider._pipeline is None:
            from transformers import pipeline

            log.info("Loading HuggingFace model %s (first call is slow)", self.cfg.model)
            HuggingFaceProvider._pipeline = pipeline(
                "text-generation", model=self.cfg.model, device_map="auto"
            )
        return HuggingFaceProvider._pipeline

    def generate(
        self, system_prompt: str, user_prompt: str, *, json_mode: bool = False
    ) -> LLMResponse:
        import time

        started = time.perf_counter()
        try:
            generator = self._get_pipeline()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            output = generator(
                messages,
                max_new_tokens=self.cfg.max_tokens,
                temperature=max(self.cfg.temperature, 0.01),
                do_sample=self.cfg.temperature > 0,
                return_full_text=False,
            )
            text = output[0]["generated_text"]
            if isinstance(text, list):  # chat-formatted output
                text = text[-1].get("content", "")
        except Exception as exc:
            raise LLMError(f"HuggingFace generation failed: {exc}") from exc

        return LLMResponse(
            text=str(text).strip(),
            provider=self.name,
            model=self.cfg.model,
            latency_seconds=time.perf_counter() - started,
        )


class EchoProvider(LLMProvider):
    """Deterministic offline provider used for testing and graceful degradation.

    It performs no generation. It returns the retrieved context, lightly
    formatted, with an explicit notice that no language model was involved.

    This is the correct degraded behaviour for a clinical tool: showing the
    user the source material verbatim is honest and still useful, whereas
    failing outright leaves them with nothing and inventing text would be
    unsafe.
    """

    @property
    def name(self) -> str:
        return "echo"

    @property
    def is_generative(self) -> bool:
        return False

    def is_available(self) -> bool:
        return True

    def generate(
        self, system_prompt: str, user_prompt: str, *, json_mode: bool = False
    ) -> LLMResponse:
        # Callers that hold the retrieved chunks should check ``is_generative``
        # and render the source text themselves - that path is language-aware
        # and does not depend on parsing the prompt back apart. This method is
        # the last-resort path for callers that do not.
        return LLMResponse(
            text=(
                "No language model is currently available, so this result could not be "
                "summarised. Please review the retrieved source material and consult a "
                "qualified clinician."
            ),
            provider=self.name,
            model="none",
            degraded=True,
        )


PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "huggingface": HuggingFaceProvider,
    "echo": EchoProvider,
}


def build_llm_provider(cfg: Config, *, allow_fallback: bool = True) -> LLMProvider:
    """Construct the configured provider, degrading to ``echo`` if unavailable.

    Args:
        cfg: Resolved configuration.
        allow_fallback: When True, an unavailable provider degrades to
            ``echo`` with a warning instead of raising. The web application
            wants this - it should start and remain usable even with no model
            server. Batch evaluation scripts pass False so a missing backend
            fails loudly rather than silently producing echo output.

    Raises:
        LLMError: If the provider is unknown, or unavailable with
            ``allow_fallback=False``.
    """
    name = cfg.llm.provider
    provider_cls = PROVIDERS.get(name)

    if provider_cls is None:
        raise LLMError(f"Unknown LLM provider {name!r}. Available: {sorted(PROVIDERS)}")

    if name in {"anthropic", "openai"} and not os.environ.get(cfg.llm.api_key_env_var):
        message = (
            f"Provider {name!r} needs an API key in ${cfg.llm.api_key_env_var}, which is not set."
        )
        if not allow_fallback:
            raise LLMError(message)
        log.warning("%s Falling back to the echo provider.", message)
        return EchoProvider(cfg.llm)

    provider = provider_cls(cfg.llm)

    if not provider.is_available():
        message = (
            f"LLM provider {name!r} is not available. "
            f"For Ollama, start the server and pull the model:\n"
            f"  ollama serve\n  ollama pull {cfg.llm.model}"
        )
        if not allow_fallback:
            raise LLMError(message)
        log.warning(
            "%s\nDegrading to the echo provider: retrieved source text will be shown "
            "verbatim without summarisation.", message,
        )
        return EchoProvider(cfg.llm)

    log.info("LLM provider ready: %s (%s)", provider.name, cfg.llm.model)
    return provider


__all__ = [
    "PROVIDERS",
    "EchoProvider",
    "HuggingFaceProvider",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "OllamaProvider",
    "build_llm_provider",
]
