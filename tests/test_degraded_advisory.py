"""Tests for the degraded advisory path, where no model text is available.

This path is reached four ways: a non-generative backend, a generation
exception, an unparseable response, and a safety violation in generated
text. It serves verbatim corpus text, and the corpus is English only, so
the surrounding notice is the only thing a Nepali reader can rely on. These
tests pin that down, because the Nepali reader is the one this path is
worst for.
"""

from __future__ import annotations

import re

import pytest

from neuroscan.config import load_config
from neuroscan.rag.advisory import AdvisoryEngine
from neuroscan.rag.llm_provider import LLMProvider
from neuroscan.rag.vectorstore import RetrievedChunk
from neuroscan.safety import DEGRADED_NOTICE_EN, DEGRADED_NOTICE_NE

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


class NonGenerativeProvider(LLMProvider):
    """A backend that retrieves but cannot synthesise, like the offline path."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "test-nongenerative"

    @property
    def is_generative(self) -> bool:
        return False

    def is_available(self) -> bool:
        return True

    def generate(self, system_prompt, user_prompt, *, json_mode=False):
        raise AssertionError("generate must not be called on this path")


class ExplodingProvider(NonGenerativeProvider):
    """Generative in principle, but every call fails."""

    @property
    def name(self) -> str:
        return "test-exploding"

    @property
    def is_generative(self) -> bool:
        return True

    def generate(self, system_prompt, user_prompt, *, json_mode=False):
        raise RuntimeError("backend unreachable")


class StubStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def search(self, query: str) -> list[RetrievedChunk]:
        return self._chunks


def make_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content="# Headache - Red Flags\n\nA **new** headache with fever needs "
                    "assessment. See [[emergency-recognition]].",
            score=0.9,
            title="Headache - Red Flags",
            doc_id="headache-red-flags",
            category="symptom",
            severity="emergency",
            sources="NICE CG150",
            last_reviewed="2026-08-13",
        ),
    ]


@pytest.fixture
def engine_factory():
    cfg = load_config()

    def build(provider: LLMProvider) -> AdvisoryEngine:
        return AdvisoryEngine(StubStore(make_chunks()), provider, cfg)

    return build


class TestDegradedNotice:
    def test_nepali_reader_gets_a_nepali_notice(self, engine_factory):
        result = engine_factory(NonGenerativeProvider()).generate(
            prediction="normal", confidence=0.96, language="ne"
        )
        assert DEGRADED_NOTICE_NE in result.text
        assert result.degraded is True

    def test_nepali_notice_says_the_material_is_english_only(self, engine_factory):
        """The defect this path had: English text with no explanation of why."""
        result = engine_factory(NonGenerativeProvider()).generate(
            prediction="normal", confidence=0.96, language="ne"
        )
        # The second paragraph of the Nepali notice is the English-only caveat.
        caveat = DEGRADED_NOTICE_NE.split("\n\n")[1]
        assert caveat in result.text

    def test_english_reader_gets_no_language_caveat(self, engine_factory):
        result = engine_factory(NonGenerativeProvider()).generate(
            prediction="normal", confidence=0.96, language="en"
        )
        assert DEGRADED_NOTICE_EN in result.text
        assert DEVANAGARI.search(result.text.split("MEDICAL DISCLAIMER")[0]) is None

    def test_notice_precedes_the_source_text(self, engine_factory):
        """A caveat after the English body would be read too late, or not at all."""
        result = engine_factory(NonGenerativeProvider()).generate(
            prediction="normal", confidence=0.96, language="ne"
        )
        assert result.text.index(DEGRADED_NOTICE_NE) < result.text.index("Headache")

    def test_source_text_is_not_translated(self, engine_factory):
        """Corpus text is served verbatim; nothing invents Nepali clinical prose."""
        result = engine_factory(NonGenerativeProvider()).generate(
            prediction="normal", confidence=0.96, language="ne"
        )
        assert "A new headache with fever needs assessment." in result.text
        # Markdown and wiki-links are still stripped for the reader.
        assert "**" not in result.text
        assert "[[" not in result.text


class TestDegradedPathsAgree:
    @pytest.mark.parametrize("language", ["en", "ne"])
    def test_generation_failure_uses_the_same_notice(self, engine_factory, language):
        result = engine_factory(ExplodingProvider()).generate(
            prediction="abnormal", confidence=0.88, language=language
        )
        expected = DEGRADED_NOTICE_NE if language == "ne" else DEGRADED_NOTICE_EN
        assert expected in result.text
        assert result.degraded is True

    @pytest.mark.parametrize("language", ["en", "ne"])
    def test_disclaimer_and_red_flags_survive(self, engine_factory, language):
        """Degradation must not cost the reader the safety furniture."""
        result = engine_factory(NonGenerativeProvider()).generate(
            prediction="abnormal", confidence=0.88, language=language
        )
        marker = "अस्वीकरण" if language == "ne" else "MEDICAL DISCLAIMER"
        assert marker in result.text
        assert result.red_flags
        assert result.red_flag_instruction

    @pytest.mark.parametrize("language", ["en", "ne"])
    def test_red_flags_are_in_the_readers_language(self, engine_factory, language):
        result = engine_factory(NonGenerativeProvider()).generate(
            prediction="abnormal", confidence=0.88, language=language
        )
        joined = " ".join(result.red_flags)
        if language == "ne":
            assert DEVANAGARI.search(joined)
        else:
            assert DEVANAGARI.search(joined) is None
