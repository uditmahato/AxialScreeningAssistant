"""Tests for the clinical safety guardrails.

These are the highest-value tests in the suite. A regression in retrieval
quality degrades the product; a regression here can produce medically unsafe
output.
"""

from __future__ import annotations

import pytest

from neuroscan.rag.advisory import validate_generated_text
from neuroscan.safety import (
    append_disclaimer,
    get_disclaimer,
    get_no_context_fallback,
    get_red_flags,
    screen_question,
)


class TestDisclaimers:
    def test_present_in_both_languages(self):
        assert "MEDICAL DISCLAIMER" in get_disclaimer("en")
        assert "अस्वीकरण" in get_disclaimer("ne")

    def test_states_it_is_not_a_diagnosis(self):
        english = get_disclaimer("en").lower()
        assert "not" in english and "diagnosis" in english

    def test_short_variants_exist(self):
        assert get_disclaimer("en", short=True)
        assert get_disclaimer("ne", short=True)

    def test_append_is_idempotent(self):
        """The LLM is also told to add a disclaimer; without this check the
        warning appears twice in roughly a third of responses."""
        once = append_disclaimer("Findings suggest a mass.", "en")
        twice = append_disclaimer(once, "en")
        assert once == twice
        assert once.count("MEDICAL DISCLAIMER") == 1

    def test_append_idempotent_in_nepali(self):
        once = append_disclaimer("नतिजा", "ne")
        assert append_disclaimer(once, "ne") == once


class TestQuestionScreening:
    @pytest.mark.parametrize(
        "question",
        [
            "How much steroid should I take for brain swelling?",
            "What dose of dexamethasone is right?",
            "how many mg should i take",
            "What is the dosage?",
            "How many tablets should I give?",
        ],
    )
    def test_refuses_dosage_questions(self, question):
        """Regression test. An earlier substring-matching implementation missed
        every phrasing that put a word between 'how much' and 'should I take',
        which is how people actually ask."""
        check = screen_question(question)
        assert not check.allowed
        assert check.category == "dosage"
        assert check.response

    @pytest.mark.parametrize(
        "question",
        [
            "How long will I live with a glioma?",
            "What is the survival rate?",
            "how long do i have",
            "What is my life expectancy?",
            "Am I going to die?",
        ],
    )
    def test_refuses_prognosis_questions(self, question):
        check = screen_question(question)
        assert not check.allowed
        assert check.category == "prognosis"

    @pytest.mark.parametrize(
        "question",
        [
            "Can I cure this at home without a doctor?",
            "How do I treat it myself?",
            "Can I avoid surgery?",
        ],
    )
    def test_refuses_self_treatment_questions(self, question):
        check = screen_question(question)
        assert not check.allowed
        assert check.category == "self_treatment"

    @pytest.mark.parametrize(
        "question",
        [
            "What is a meningioma?",
            "Could this be an infection rather than cancer?",
            "Which hospitals in Nepal do neurosurgery?",
            "What does an abnormal result mean?",
            "How much does an MRI cost in Nepal?",
            "What are the warning signs I should watch for?",
        ],
    )
    def test_allows_legitimate_questions(self, question):
        """Over-refusal is also a failure. A tool that refuses ordinary
        questions will not be used."""
        assert screen_question(question).allowed

    def test_refusal_localised_to_nepali(self):
        check = screen_question("What is the survival rate?", "ne")
        assert not check.allowed
        assert any(ord(c) > 0x900 for c in check.response)


class TestGeneratedTextValidation:
    @pytest.mark.parametrize(
        ("text", "category"),
        [
            ("You have a brain tumour and need surgery.", "definitive_diagnosis"),
            ("You have a malignant glioma.", "definitive_diagnosis"),
            ("Take 500 mg of the medication daily.", "dosage"),
            ("Take 2 tablets twice a day.", "dosage"),
            ("The survival rate is 40% at five years.", "prognosis"),
            ("Your life expectancy is around 18 months.", "prognosis"),
        ],
    )
    def test_flags_prohibited_content(self, text, category):
        assert category in validate_generated_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The scan shows features that may be consistent with an abnormality.",
            "Possible causes include neurocysticercosis, tuberculoma and tumour.",
            "Please seek emergency care if you develop a severe headache.",
            "A radiologist should review this scan before any decision is made.",
            # Ordinary clinical prose that an earlier over-broad pattern rejected.
            "The diagnosis is usually built from chest imaging and clinical findings.",
            "Referral to neurosurgery may be appropriate depending on the findings.",
        ],
    )
    def test_permits_safe_clinical_language(self, text):
        """False positives here are not harmless: a flagged response is
        discarded, so an over-broad pattern silently removes good advice."""
        assert validate_generated_text(text) == []


class TestNepaliScreening:
    """Devanagari input screening.

    Regression suite for the worst defect an adversarial audit found: every
    refusal trigger was English (plus four romanised fragments), so the safety
    posture inverted depending on which language the user typed in. An English
    dosage question was refused outright while the identical Nepali question
    reached the model. The system advertises Nepali as a first-class language.
    """

    @pytest.mark.parametrize(
        ("question", "category"),
        [
            ("कति मात्रा औषधि खाने?", "dosage"),                # how much medicine to take
            ("मैले कति मात्रा औषधि खानुपर्छ?", "dosage"),
            ("औषधि कति मिलिग्राम खाने?", "dosage"),
            ("म कति दिन बाँच्छु?", "prognosis"),                 # how many days will I live
            ("यो कति दिन बाँच्छ?", "prognosis"),
            ("घरमै उपचार गर्न सकिन्छ?", "self_treatment"),        # can I treat at home
            ("के म घरमै उपचार गर्न सक्छु?", "self_treatment"),
            ("डाक्टर बिना उपचार गर्न मिल्छ?", "self_treatment"),
        ],
    )
    def test_devanagari_questions_are_screened(self, question, category):
        check = screen_question(question, "ne")
        assert not check.allowed, f"Nepali {category} question reached the model: {question}"
        assert check.category == category

    @pytest.mark.parametrize(
        "question",
        [
            "न्यूरोसिस्टिसर्कोसिस के हो?",          # what is neurocysticercosis
            "नेपालमा MRI को खर्च कति हो?",          # how much does an MRI cost
            "यो नतिजाको अर्थ के हो?",               # what does this result mean
            "कुन अस्पतालमा जाने?",                   # which hospital should I go to
        ],
    )
    def test_legitimate_nepali_questions_are_allowed(self, question):
        """Over-refusal is also a failure: a tool that refuses ordinary Nepali
        questions is worse than one that never offered Nepali."""
        assert screen_question(question, "ne").allowed


class TestNaturalPhrasingScreening:
    """The phrasings people actually type, rather than the textbook ones.

    The original patterns had no slot for an intervening word, so "what is the
    usual dose of DEXAMETHASONE" passed while a bare "what dose" was caught.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "What is the usual dose of dexamethasone for brain swelling?",
            "What is the standard dose of steroids?",
            "What is the paediatric dose of albendazole?",
            "Tell me the dose of albendazole",
            "Recommended dose for dexamethasone?",
            "How much albendazole?",
            "Kati mg khanu parcha?",
        ],
    )
    def test_dosage_phrasings(self, question):
        assert screen_question(question).category == "dosage"

    @pytest.mark.parametrize(
        "question",
        [
            "How long does someone with glioblastoma usually live?",
            "What are the chances of survival with this?",
            "Is this fatal?",
            "What is the outlook for my father?",
        ],
    )
    def test_prognosis_phrasings(self, question):
        """Third-person matters as much as first: people ask about a relative
        far more often than about themselves."""
        assert screen_question(question).category == "prognosis"

    @pytest.mark.parametrize(
        "question",
        [
            "Can I manage this at home with herbs?",
            "Is there a natural remedy I can try first?",
        ],
    )
    def test_self_treatment_phrasings(self, question):
        assert screen_question(question).category == "self_treatment"


class TestPlainLanguageOutputValidation:
    """Output validation must cover plain language, not textbook phrasing.

    The system prompts instruct the model to write plainly for a reader who
    "may be a community health volunteer", which pushes it toward spelled-out
    units - exactly what the original patterns missed. They caught
    "400 mg twice a day" and let "500 milligrams ... taken twice daily" pass.
    """

    @pytest.mark.parametrize(
        ("text", "category"),
        [
            ("Give albendazole 500 milligrams for 8 days.", "dosage"),
            ("Albendazole is taken twice daily for eight days.", "dosage"),
            ("Take two tablets three times daily.", "dosage"),
            ("Dexamethasone 4 milligrams every six hours is standard.", "dosage"),
            ("The dose is 15 mg/kg per day.", "dosage"),
            ("Median survival is 12 to 15 months for this condition.", "prognosis"),
            ("The five-year survival is around 15%.", "prognosis"),
            ("Life expectancy is about 18 months.", "prognosis"),
        ],
    )
    def test_plain_english_is_flagged(self, text, category):
        assert category in validate_generated_text(text)

    @pytest.mark.parametrize(
        ("text", "category"),
        [
            ("एल्बेन्डाजोल ४०० मिलिग्राम दिनको दुई पटक खानुहोस्।", "dosage"),
            ("तपाईंलाई मस्तिष्कको क्यान्सर छ।", "definitive_diagnosis"),
            ("बाँच्ने सम्भावना ४५ प्रतिशत छ।", "prognosis"),
            # Mixed script is the realistic Nepali case - drug names and units
            # usually stay Latin.
            ("एल्बेन्डाजोल 400 mg दिनको दुई पटक खानुहोस्।", "dosage"),
        ],
    )
    def test_devanagari_output_is_flagged(self, text, category):
        assert category in validate_generated_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            "यो नतिजा निदान होइन। कृपया चिकित्सकसँग परामर्श गर्नुहोस्।",
            "सम्भावित कारणहरूमा न्यूरोसिस्टिसर्कोसिस र क्षयरोग पर्छन्।",
            "Treatment is delivered free through the National Tuberculosis Programme.",
        ],
    )
    def test_safe_bilingual_text_passes(self, text):
        assert validate_generated_text(text) == []


class TestRedFlags:
    def test_available_in_both_languages(self):
        for language in ("en", "ne"):
            instruction, flags = get_red_flags(language)
            assert instruction
            assert len(flags) >= 5

    def test_cover_the_critical_presentations(self):
        _, flags = get_red_flags("en")
        joined = " ".join(flags).lower()
        for term in ("headache", "seizure", "weakness", "vision", "consciousness"):
            assert term in joined, f"red flags omit {term!r}"

    def test_nepali_flags_are_devanagari(self):
        _, flags = get_red_flags("ne")
        assert all(any(ord(c) > 0x900 for c in flag) for flag in flags)


class TestFallbacks:
    def test_no_context_fallback_declines_rather_than_inventing(self):
        english = get_no_context_fallback("en").lower()
        assert "do not have" in english or "not have reliable" in english
        assert "consult" in english

    def test_nepali_fallback_is_devanagari(self):
        assert any(ord(c) > 0x900 for c in get_no_context_fallback("ne"))
