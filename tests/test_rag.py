"""Tests for the corpus, retrieval configuration and language detection.

Tests requiring the built FAISS index are marked ``requires_data`` and are
skipped when it is absent, so the suite runs on a fresh clone.
"""

from __future__ import annotations

import pytest

from neuroscan.chatbot.engine import _is_anaphoric
from neuroscan.chatbot.language import detect_language, devanagari_ratio, get_ui_strings, t
from neuroscan.config import PROJECT_ROOT
from neuroscan.rag.advisory import build_retrieval_query
from neuroscan.rag.corpus import corpus_statistics, load_corpus, parse_frontmatter

KB_DIR = PROJECT_ROOT / "knowledge_base"
requires_corpus = pytest.mark.skipif(
    not (KB_DIR / "medical").exists(), reason="knowledge base not present"
)


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "What does an abnormal result mean?",
            "Is a meningioma dangerous?",
            "The patient has a headache and vomiting",
            "How much does an MRI cost?",
        ],
    )
    def test_detects_english(self, text):
        assert detect_language(text) == "en"

    @pytest.mark.parametrize(
        "text",
        [
            "मेरो रिपोर्टमा असामान्य लेखेको छ, यसको अर्थ के हो?",
            "टाउको दुख्ने कहिले खतरनाक हुन्छ?",
            "क्षयरोगको उपचार निःशुल्क छ?",
        ],
    )
    def test_detects_devanagari_nepali(self, text):
        assert detect_language(text) == "ne"

    @pytest.mark.parametrize(
        "text",
        [
            "tauko dukhyo ra banta aayo, ke garne?",
            "malai kasto aspatal jane ho?",
            "MRI ko kharcha kati parcha?",
        ],
    )
    def test_detects_romanised_nepali(self, text):
        """Very common in practice - many users have no Devanagari keyboard -
        and completely invisible to script detection."""
        assert detect_language(text) == "ne"

    def test_empty_input_returns_default(self):
        assert detect_language("") == "en"
        assert detect_language("   ", default="ne") == "ne"

    def test_devanagari_ratio(self):
        assert devanagari_ratio("hello") == 0.0
        assert devanagari_ratio("नमस्ते") == 1.0
        assert 0.0 < devanagari_ratio("MRI को नतिजा") < 1.0

    def test_mixed_script_favours_nepali(self):
        """A question that is mostly an English medical term with a few Nepali
        words is still a Nepali question."""
        assert detect_language("MRI मा के देखियो?") == "ne"


class TestUIStrings:
    def test_every_key_exists_in_both_languages(self):
        english = get_ui_strings("en")
        nepali = get_ui_strings("ne")
        assert set(english) == set(nepali)
        assert all(v for v in english.values())
        assert all(v for v in nepali.values())

    def test_nepali_strings_are_actually_nepali(self):
        """Guards against an untranslated key silently falling back to
        English."""
        nepali = get_ui_strings("ne")
        # 'language' is the toggle label and is deliberately the other language.
        checked = {k: v for k, v in nepali.items() if k != "language"}
        devanagari = [v for v in checked.values() if any(ord(c) > 0x900 for c in v)]
        assert len(devanagari) / len(checked) > 0.9

    def test_unknown_key_returns_the_key(self):
        assert t("nonexistent_key_xyz", "en") == "nonexistent_key_xyz"


class TestFrontmatter:
    def test_parses_metadata_and_body(self):
        text = "---\nid: test-doc\ntitle: Test\n---\n\nBody content here."
        metadata, body = parse_frontmatter(text)
        assert metadata["id"] == "test-doc"
        assert body.strip() == "Body content here."

    def test_handles_absent_frontmatter(self):
        metadata, body = parse_frontmatter("Just body text.")
        assert metadata == {}
        assert body == "Just body text."

    def test_malformed_yaml_does_not_raise(self):
        """One bad document must not prevent the whole index from building."""
        metadata, body = parse_frontmatter("---\n: : invalid: [\n---\n\nBody.")
        assert isinstance(metadata, dict)
        assert "Body" in body


class TestRetrievalQuery:
    def test_abnormal_query_names_the_nepal_infections(self):
        """Embedding the bare word 'abnormal' retrieves almost nothing useful.
        Naming the conditions is what guarantees the Nepal-relevant infective
        causes reach the context rather than being left to chance."""
        query = build_retrieval_query("abnormal").lower()
        assert "neurocysticercosis" in query
        assert "tuberculoma" in query

    def test_normal_query_covers_the_limitations(self):
        query = build_retrieval_query("normal").lower()
        assert "normal" in query
        assert "limitation" in query or "not exclude" in query


class TestAnaphoraDetection:
    @pytest.mark.parametrize(
        "question",
        ["Could it be an infection?", "What does this mean?", "Is it serious?",
         "What should I do next?", "Where do I go?"],
    )
    def test_identifies_context_dependent_questions(self, question):
        assert _is_anaphoric(question)

    @pytest.mark.parametrize(
        "question",
        ["Is a meningioma dangerous?",
         "What is tuberculoma and how is it treated in Nepal?",
         "Which hospitals perform neurosurgery in eastern Nepal today?"],
    )
    def test_leaves_self_contained_questions_alone(self, question):
        """Expanding a self-contained question only adds noise."""
        assert not _is_anaphoric(question)


@requires_corpus
class TestCorpus:
    def test_loads_the_documents(self):
        assert len(load_corpus(KB_DIR)) >= 50

    def test_meets_corpus_minimum(self):
        stats = corpus_statistics(load_corpus(KB_DIR))
        assert stats["medical_documents"] >= 50, (
            f"The corpus requirement is 50+ medical documents, found "
            f"{stats['medical_documents']}"
        )

    def test_covers_the_nepal_critical_conditions(self):
        """A corpus assembled from Western sources alone would misdirect a
        Nepali health worker on the single most common lesion causes."""
        ids = {d.doc_id for d in load_corpus(KB_DIR)}
        for required in ("neurocysticercosis", "tuberculoma",
                         "ring-enhancing-lesion-differential"):
            assert required in ids, f"corpus is missing {required!r}"

    def test_nepal_database_meets_facility_minimum(self):
        import json

        payload = json.loads((KB_DIR / "nepal" / "hospitals.json").read_text("utf-8"))
        assert payload["count"] >= 10, "The facility database requires 10+ neurology hospitals"
        for hospital in payload["hospitals"]:
            assert hospital["name"] and hospital["city"] and hospital["province"]

    def test_every_document_declares_sources(self):
        undocumented = [d.doc_id for d in load_corpus(KB_DIR) if not d.sources]
        assert not undocumented, f"documents without sources: {undocumented}"

    def test_every_document_has_a_review_date(self):
        stale = [d.doc_id for d in load_corpus(KB_DIR) if d.last_reviewed == "unknown"]
        assert not stale, f"documents without last_reviewed: {stale}"

    def test_document_ids_are_unique(self):
        ids = [d.doc_id for d in load_corpus(KB_DIR)]
        assert len(ids) == len(set(ids))

    def test_corpus_contains_no_dosage_guidance(self):
        """Editorial rule 1: the corpus must not contain material that would
        tempt the model to answer a dosage question."""
        import re

        pattern = re.compile(r"\b\d+\s*(?:mg|milligrams?)\b", re.IGNORECASE)
        offenders = [d.doc_id for d in load_corpus(KB_DIR) if pattern.search(d.content)]
        assert not offenders, f"documents containing dosages: {offenders}"
