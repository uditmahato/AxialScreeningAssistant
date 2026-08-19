"""The bilingual conversational engine.

A chat turn passes through four stages, and can be stopped at any of them:

    screen -> retrieve -> generate -> validate

Screening happens *before* retrieval so a prohibited request (a dosage, a
prognosis) never reaches the model at all. Validation happens after generation
because instructions alone are not a guarantee.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from neuroscan.chatbot.language import detect_language
from neuroscan.rag.advisory import validate_generated_text
from neuroscan.rag.prompts import build_chatbot_prompt, format_context
from neuroscan.safety import (
    Language,
    get_no_context_fallback,
    screen_question,
)
from neuroscan.textfmt import (
    drop_leading_title,
    humanise_dashes,
    strip_passage_refs,
    to_plain_text,
)
from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config
    from neuroscan.rag.llm_provider import LLMProvider
    from neuroscan.rag.vectorstore import VectorStore

log = get_logger("chatbot.engine")

#: Words that signal a question depends on something not stated in it.
#: Matched as whole words against the lowercased question.
_ANAPHORIC_MARKERS = frozenset({
    "it", "this", "that", "these", "those", "my", "mine", "the result",
    "यो", "त्यो", "यसको", "मेरो", "यसले",
})

#: Questions that are short and have no clinical noun of their own are treated
#: as context-dependent regardless of whether they contain a pronoun -
#: "What next?", "Where do I go?", "How much?".
_SELF_CONTAINED_HINTS = frozenset({
    "meningioma", "glioma", "glioblastoma", "tumour", "tumor", "tuberculoma",
    "neurocysticercosis", "abscess", "stroke", "hydrocephalus", "seizure",
    "epilepsy", "mri", "ct", "headache", "aneurysm", "metastases", "lymphoma",
})


def _is_anaphoric(question: str) -> bool:
    """Whether a question depends on context not contained in it.

    Used to decide whether query expansion is worth the noise it introduces.
    A question naming its own subject ("Is a meningioma dangerous?") retrieves
    perfectly well on its own; one that says "it" does not.
    """
    lowered = question.casefold()
    words = set(re.findall(r"[\wऀ-ॿ]+", lowered))

    if words & _SELF_CONTAINED_HINTS:
        return False
    if words & _ANAPHORIC_MARKERS:
        return True
    # Very short questions with no subject of their own are context-dependent.
    return len(words) <= 6


@dataclass
class ChatTurn:
    """One exchange in a conversation."""

    question: str
    answer: str
    language: Language = "en"


@dataclass
class ChatResponse:
    """A chatbot reply with its provenance and safety record."""

    text: str
    language: Language
    citations: list[str] = field(default_factory=list)
    retrieved_count: int = 0
    provider: str = ""
    refused: bool = False
    refusal_category: str | None = None
    safety_flags: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    degraded: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "language": self.language,
            "citations": self.citations,
            "retrieved_count": self.retrieved_count,
            "provider": self.provider,
            "refused": self.refused,
            "refusal_category": self.refusal_category,
            "safety_flags": self.safety_flags,
            "latency_seconds": round(self.latency_seconds, 2),
            "degraded": self.degraded,
        }


class ChatbotEngine:
    """Retrieval-grounded bilingual chatbot.

    Args:
        vector_store: Retrieval index over the knowledge base.
        llm: Generation backend.
        cfg: Resolved configuration.
    """

    def __init__(self, vector_store: VectorStore, llm: LLMProvider, cfg: Config) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.cfg = cfg

    def ask(
        self,
        question: str,
        *,
        history: list[ChatTurn] | None = None,
        language: Language | None = None,
        scan_prediction: str | None = None,
        scan_confidence: float | None = None,
    ) -> ChatResponse:
        """Answer a user question.

        Args:
            question: Raw user input.
            history: Prior turns, for conversational context.
            language: Force a reply language. When None it is detected.
            scan_prediction: Current scan result, so questions like "what does
                this mean?" can be answered in context.
            scan_confidence: Confidence for the above.
        """
        started = time.perf_counter()
        history = history or []

        question = (question or "").strip()
        if not question:
            resolved = language or self.cfg.chatbot.default_language
            return ChatResponse(
                text=get_no_context_fallback(resolved),
                language=resolved,
                latency_seconds=time.perf_counter() - started,
            )

        # Truncate rather than reject: an over-long question is usually someone
        # pasting a report, and the useful part is at the start.
        limit = self.cfg.chatbot.max_question_chars
        if len(question) > limit:
            log.info("Question truncated from %d to %d characters", len(question), limit)
            question = question[:limit]

        resolved_language: Language = language or detect_language(
            question, default=self.cfg.chatbot.default_language
        )

        # --- Stage 1: screen before the model ever sees it -------------------
        check = screen_question(question, resolved_language)
        if not check.allowed:
            log.info("Refused question in category %r", check.category)
            return ChatResponse(
                text=check.response or "",
                language=resolved_language,
                refused=True,
                refusal_category=check.category,
                safety_flags=[f"refused_{check.category}"],
                latency_seconds=time.perf_counter() - started,
            )

        # --- Stage 2: retrieve ----------------------------------------------
        # Plain similarity, not MMR: the user asked about one specific thing,
        # and MMR's diversity penalty would trade away directly relevant
        # passages for adjacent topics they did not ask about.
        #
        # Multi-query because conversational questions are short and anaphoric
        # ("could it be an infection instead?"), which embeds badly on its own.
        chunks = self.vector_store.search_multi(
            self._build_queries(question, history, scan_prediction),
            search_type="similarity",
        )

        if not chunks:
            log.info("No context above threshold for question: %.60s", question)
            return ChatResponse(
                text=get_no_context_fallback(resolved_language),
                language=resolved_language,
                retrieved_count=0,
                provider=self.llm.name,
                safety_flags=["no_context"],
                degraded=True,
                latency_seconds=time.perf_counter() - started,
            )

        # No generative backend: return the retrieved source text, localised
        # and attributed. Skipping generation also skips output validation,
        # which is correct - the corpus is curated and reviewed, and screening
        # it with patterns designed for model output only produces false
        # positives on ordinary clinical prose.
        if not self.llm.is_generative:
            return ChatResponse(
                text=self._source_text_fallback(chunks, resolved_language),
                language=resolved_language,
                citations=self._citations(chunks),
                retrieved_count=len(chunks),
                provider=self.llm.name,
                safety_flags=["generation_unavailable"],
                degraded=True,
                latency_seconds=time.perf_counter() - started,
            )

        # --- Stage 3: generate ----------------------------------------------
        scan_context = ""
        if scan_prediction:
            confidence_note = (
                f" with {scan_confidence:.1%} confidence" if scan_confidence is not None else ""
            )
            scan_context = (
                "CURRENT SCAN RESULT\n-------------------\n"
                f"The user's uploaded scan was classified as: {scan_prediction}{confidence_note}.\n"
                "Bear this in mind if the question refers to 'this result' or 'my scan'."
            )

        system_prompt, user_prompt = build_chatbot_prompt(
            question=question,
            context=format_context(chunks),
            history=self._format_history(history),
            scan_context=scan_context,
            language=resolved_language,
        )

        try:
            response = self.llm.generate(system_prompt, user_prompt)
        except Exception as exc:
            log.error("Chat generation failed: %s", exc)
            return ChatResponse(
                text=self._source_text_fallback(chunks, resolved_language),
                language=resolved_language,
                citations=self._citations(chunks),
                retrieved_count=len(chunks),
                provider=self.llm.name,
                safety_flags=["generation_unavailable"],
                degraded=True,
                latency_seconds=time.perf_counter() - started,
            )

        # --- Stage 4: validate ----------------------------------------------
        flags = validate_generated_text(response.text)
        if flags:
            log.error("Chat reply violated safety rules %s - replacing with source text", flags)
            return ChatResponse(
                text=self._source_text_fallback(chunks, resolved_language),
                language=resolved_language,
                citations=self._citations(chunks),
                retrieved_count=len(chunks),
                provider=response.provider,
                safety_flags=[*flags, "replaced_unsafe_output"],
                degraded=True,
                latency_seconds=time.perf_counter() - started,
            )

        return ChatResponse(
            # Plain text, not markdown: chat answers are shown via textContent
            # (XSS-safe, but markdown-literal) and printed in the PDF
            # transcript. No per-answer disclaimer: the page carries a
            # persistent one, and repeating the identical warning under every
            # reply teaches users to skip all of them. The /chat route returns
            # a separate disclaimer field for programmatic consumers.
            text=to_plain_text(strip_passage_refs(humanise_dashes(response.text))),
            language=resolved_language,
            citations=self._citations(chunks),
            retrieved_count=len(chunks),
            provider=response.provider,
            degraded=response.degraded,
            latency_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _build_queries(
        question: str,
        history: list[ChatTurn],
        scan_prediction: str | None,
    ) -> list[str]:
        """Build alternative phrasings of the user's information need.

        The user's own wording always comes first and is never displaced.
        Expansions are kept **short and specific**: a long keyword-stuffed
        rewrite embeds as generic clinical text and matches every document
        equally, which is worse than not expanding at all.

        Expansion is also conditional. A self-contained question
        ("Is a meningioma dangerous?") needs no help, and adding context to it
        only introduces noise. Only questions that actually depend on an
        unstated referent are rewritten.
        """
        queries = [question]

        if scan_prediction and _is_anaphoric(question):
            # Minimal pronoun resolution: name the referent, add nothing else.
            queries.append(f"{scan_prediction} brain MRI scan result: {question}")

        if history and _is_anaphoric(question):
            previous = history[-1].question
            if previous and previous != question:
                queries.append(f"{previous} {question}")

        return queries

    def _format_history(self, history: list[ChatTurn]) -> str:
        """Render recent turns, oldest first, bounded by the configured window.

        Only the most recent turns are kept. An unbounded history eventually
        crowds the retrieved context out of the model's window, at which point
        answers stop being grounded in the corpus - the exact failure this
        architecture exists to prevent.
        """
        if not history:
            return ""
        recent = history[-self.cfg.chatbot.max_history_turns :]
        return "\n\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in recent)

    @staticmethod
    def _source_text_fallback(chunks: list, language: Language) -> str:
        header = (
            "ज्ञान-भण्डारबाट लिइएको सम्बन्धित जानकारी:"
            if language == "ne"
            else "Relevant information from the knowledge base:"
        )
        # Corpus markdown and [[wiki-links]] must not reach the reader.
        body = "\n\n".join(
            f"{c.title}\n\n{drop_leading_title(to_plain_text(c.content), c.title)}"
            for c in chunks[:2]
        )
        return f"{header}\n\n{body}"

    @staticmethod
    def _citations(chunks: list) -> list[str]:
        seen: set[str] = set()
        citations: list[str] = []
        for chunk in chunks:
            citation = chunk.citation()
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations


def build_chatbot_engine(cfg: Config, *, allow_fallback: bool = True) -> ChatbotEngine:
    """Construct a :class:`ChatbotEngine` from configuration."""
    from neuroscan.rag.llm_provider import build_llm_provider
    from neuroscan.rag.vectorstore import load_index

    return ChatbotEngine(load_index(cfg), build_llm_provider(cfg, allow_fallback=allow_fallback), cfg)


__all__ = ["ChatResponse", "ChatTurn", "ChatbotEngine", "build_chatbot_engine"]
