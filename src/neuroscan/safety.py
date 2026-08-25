"""Clinical safety constants and guardrails.

Every user-facing surface - web page, chatbot reply, PDF report - must route
its disclaimer through this module. Centralising the text means the wording is
reviewed once and cannot silently drift between the screen and the printout,
which is exactly the drift a clinical audit would flag.

Report section 5.3: "The system will prominently display disclaimers ... and
will have required medical disclaimers on all outputs."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Language = Literal["en", "ne"]

# ---------------------------------------------------------------------------
# Mandatory disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER_EN = (
    "MEDICAL DISCLAIMER: This system is a research prototype and a "
    "decision-support tool. It does NOT provide a medical diagnosis and is NOT "
    "a substitute for professional medical judgement. Every result must be "
    "reviewed by a qualified radiologist or physician before any clinical "
    "action is taken. Do not make treatment decisions based on this output "
    "alone. If you have urgent symptoms, seek immediate medical care."
)

DISCLAIMER_NE = (
    "चिकित्सकीय अस्वीकरण: यो प्रणाली एक अनुसन्धानमूलक प्रणाली र "
    "निर्णय-सहायता उपकरण मात्र हो। यसले चिकित्सकीय निदान गर्दैन र योग्य "
    "चिकित्सकको व्यावसायिक निर्णयको विकल्प होइन। कुनै पनि उपचार सुरु गर्नुअघि "
    "सबै नतिजा योग्य रेडियोलोजिस्ट वा चिकित्सकबाट पुनरावलोकन गराउनुपर्छ। यस "
    "नतिजाको आधारमा मात्र उपचार सम्बन्धी निर्णय नगर्नुहोस्। यदि तपाईंलाई "
    "गम्भीर लक्षणहरू देखिएका छन् भने तुरुन्तै चिकित्सकीय सहायता लिनुहोस्।"
)

DISCLAIMER_SHORT_EN = "Decision support, not a diagnosis. A qualified clinician must review every result."
DISCLAIMER_SHORT_NE = (
    "निर्णय-सहायता मात्र - यो निदान होइन। योग्य चिकित्सकबाट पुनरावलोकन अनिवार्य छ।"
)

# ---------------------------------------------------------------------------
# Emergency red flags
#
# These are surfaced unconditionally, on every result, regardless of what the
# classifier predicted. A "normal" prediction on a patient who is actively
# deteriorating is the most dangerous output this system can produce, so the
# escalation advice must not be gated behind the model's confidence.
# ---------------------------------------------------------------------------

RED_FLAGS_EN = [
    "Sudden, severe headache unlike any previous headache",
    "Seizure, fitting, or loss of consciousness",
    "Sudden weakness or numbness affecting one side of the body",
    "Sudden loss of vision, double vision, or difficulty speaking",
    "Confusion, drowsiness, or unusual behaviour change",
    "Repeated vomiting together with headache, especially on waking",
    "Head injury followed by worsening symptoms",
]

RED_FLAGS_NE = [
    "अचानक, अत्यन्तै तीव्र टाउको दुख्ने (पहिले कहिल्यै नभएको खालको)",
    "शरीर काम्ने (दौरा) वा बेहोस हुने",
    "अचानक शरीरको एकापट्टि कमजोरी वा सुन्निने",
    "अचानक आँखा नदेख्ने, दोहोरो देखिने, वा बोल्न कठिनाइ हुने",
    "अलमल हुने, निद्रा लागिरहने, वा व्यवहारमा असामान्य परिवर्तन",
    "टाउको दुखाइसँगै बारम्बार बान्ता हुने, विशेषगरी बिहान उठ्दा",
    "टाउकोमा चोट लागेपछि लक्षणहरू बिग्रँदै जाने",
]

EMERGENCY_INSTRUCTION_EN = (
    "If any of the following are present, treat this as a medical emergency "
    "and go to the nearest hospital emergency department immediately. Do not "
    "wait for a scan review."
)

EMERGENCY_INSTRUCTION_NE = (
    "तलका मध्ये कुनै पनि लक्षण देखिएमा यसलाई आपतकालीन अवस्था मानी तुरुन्तै "
    "नजिकको अस्पतालको आपतकालीन कक्षमा जानुहोस्। स्क्यान पुनरावलोकनको "
    "पर्खाइ नगर्नुहोस्।"
)

# Nepal ambulance / emergency number.
EMERGENCY_CONTACT = "102 (Ambulance, Nepal) / 100 (Police)"

# ---------------------------------------------------------------------------
# LLM behavioural guardrails
# ---------------------------------------------------------------------------

#: Injected into every RAG and chatbot system prompt.
SAFETY_RULES = """\
You are a clinical decision-support assistant. You must obey these rules without exception:

1. NEVER state or imply a definitive diagnosis. Use hedged language such as
   "the scan shows features that may be consistent with ..." and always defer
   to a qualified clinician.
2. NEVER prescribe medication, dosages, or treatment regimens. You may name
   general categories of management that a clinician might consider, and you
   must attribute them to the retrieved source.
3. NEVER estimate prognosis, survival time, or life expectancy.
4. Answer ONLY from the retrieved context provided below. If the context does
   not support an answer, say plainly that you do not have reliable
   information on that point and recommend consulting a neurologist or
   neurosurgeon. Do not fill gaps from memory.
5. ALWAYS recommend review by a qualified radiologist or physician.
6. If the user describes any emergency red-flag symptom, your FIRST sentence
   must direct them to emergency care immediately, before any other content.
7. Keep answers concise, plain, and free of unexplained jargon. The reader may
   be a community health volunteer or a patient, not a specialist.
8. Reply in the same language the user wrote in (English or Nepali).
"""

#: Returned verbatim when retrieval yields nothing above the score threshold.
#: Preferable to letting the model improvise on a medical question.
NO_CONTEXT_FALLBACK_EN = (
    "I do not have reliable information in my knowledge base to answer that "
    "question safely. Please consult a qualified neurologist, neurosurgeon, or "
    "your nearest hospital for guidance on this."
)

NO_CONTEXT_FALLBACK_NE = (
    "यस प्रश्नको सुरक्षित उत्तर दिन मेरो ज्ञान-भण्डारमा भरपर्दो जानकारी छैन। "
    "कृपया योग्य न्यूरोलोजिस्ट, न्यूरोसर्जन, वा नजिकको अस्पतालमा सम्पर्क "
    "गर्नुहोस्।"
)

# Topics the assistant refuses outright, as regular expressions matched
# case-insensitively against the user's question.
#
# **Bias these toward over-refusal.** A refusal costs the user one redirect to a
# clinician; a miss lets a dosage or prognosis question reach the model. There
# is exactly one screening call site and no second line of defence, so anything
# that gets past here is answered.
#
# An earlier version was far too narrow and failed open on 16 of 24 realistic
# probes. Two failures are worth naming, because both are the kind that look
# fine in review:
#
# * Patterns had no slot for an intervening word, so "what is the usual dose of
#   DEXAMETHASONE" passed while the bare "what dose" was caught. Real users
#   write the former.
# * Every pattern was English. The system advertises Nepali support and routes
#   Devanagari queries to a multilingual retriever, so a Nepali dosage question
#   was screened by nothing at all. Devanagari and romanised patterns are now
#   first-class, not an afterthought.
REFUSAL_TRIGGERS: dict[str, tuple[str, ...]] = {
    "dosage": (
        # "dose"/"dosage"/"dosing" in a question is essentially always about
        # medication. Refusing a stray radiation-dose question is the safe miss.
        r"\bdos(?:e|es|age|ages|ing)\b",
        r"\bhow (?:much|many)\b[\w\s,'-]{0,40}\b(?:take|taken|taking|give|given|"
        r"swallow|inject|administer|prescrib\w*)\b",
        r"\bhow many\b[\w\s]{0,20}\b(?:mg|milligrams?|mcg|micrograms?|ml|"
        r"tablets?|pills?|capsules?|drops?|injections?|doses?)\b",
        r"\bhow much\b[\w\s]{0,20}\b(?:mg|milligrams?|ml|medicine|medication|drug|"
        r"steroids?|antibiotics?|albendazole|dexamethasone|paracetamol)\b",
        r"\d+\s*(?:mg|mcg|ml|milligrams?)\b",
        r"\bprescrib\w*\s+(?:me|for me|something)\b",
        r"\bwhat (?:medicine|medication|drug|tablets?|pills?)\s+(?:should|do|can|must)\b",
        # Quantity-and-frequency phrasings. Red-teaming found "half a tablet
        # ... twice daily" and "दिनको कति पटक" sailing through: the question
        # names an amount or a schedule without the word "dose".
        r"\b(?:half|quarter)\s+(?:a\s+|an?\s+)?(?:tablet|pill|capsule)\b",
        r"\b(?:give|take|giving|taking)\b[\w\s,'-]{0,30}\b(?:tablets?|pills?|capsules?)\b"
        r"[\w\s,'-]{0,25}\b(?:daily|twice|thrice|a day|per day|every)\b",
        # Romanised Nepali
        r"\bkati\s+(?:matra|khanu|khane|mg|tablet|golee|goli)",
        r"\bmatra\s+kati\b",
        r"\baushadhi\s+kati\b",
        r"\b(?:aushadhi|ausadhi|goli|golee)\b[\w\s]{0,25}\bkati\s+patak\b",
        r"\bkati\s+patak\b[\w\s]{0,25}\b(?:khane|khanu|aushadhi|ausadhi|goli|golee)\b",
        # Devanagari: मात्रा (dose), कति खाने (how much to take), औषधि कति,
        # कति पटक (how many times) paired with taking medicine
        r"मात्रा",
        r"कति\s*(?:वटा\s*)?(?:खाने|खानु|लिने|लिनु)",
        r"औषधि\s*कति",
        r"कति\s*पटक\s*(?:औषधि|खाने|खानु|दिने)",
        r"दिनको\s*कति\s*पटक",
        r"डोज",
    ),
    "prognosis": (
        # Third-person and impersonal phrasings matter as much as first-person:
        # people ask about a relative far more often than about themselves.
        r"how long\b[\w\s,'-]{0,40}\b(?:live|survive|got left|have left|last)\b",
        # First-person phrasings kept explicit. A bare "have" in the
        # alternation above would also catch "how long have I had this
        # headache?", which is a duration question, not a prognosis one.
        r"how long (?:do|have|has) (?:i|he|she|they|my \w+) (?:have|got)\b",
        r"\blife expectancy\b",
        r"\bsurviv(?:al|e|es|ing)\b",
        r"\bprognosis\b",
        r"\boutlook\b",
        r"\b(?:is|are)\s+(?:this|it|they|he|she)\s+(?:fatal|terminal|curable|incurable)\b",
        r"\b(?:am|is|are)\s+\w+\s+going to die\b",
        r"\bwill\s+(?:i|he|she|they|my \w+)\s+(?:die|survive|make it|recover)\b",
        r"\bchances? of (?:surviv\w+|recovery|living)\b",
        r"\bhow (?:bad|serious|advanced)\b[\w\s]{0,20}\b(?:stage|grade)\b",
        r"\bterminal\b",
        # "How many years do I have?" and "death sentence" carry the survival
        # question without any of the survival vocabulary above.
        r"\bdeath sentence\b",
        r"\bhow (?:many|much) (?:years?|months?|days?|time)\b[\w\s,'-]{0,20}"
        r"\b(?:do|does|have|has|left)\b",
        # Romanised Nepali
        r"\bkati\s+din\s+bach",
        r"\bmarcha\b|\bmarchha\b",
        # Devanagari: बाँच्नु (to live), आयु (lifespan), मर्नु (to die)
        r"कति\s*दिन\s*बाँ",
        r"बाँच्(?:छु|छ|ने|न)",
        r"आयु\s*कति",
        r"मर्(?:छु|छ|ने)",
        r"कति\s*समय\s*(?:बाँ|छ)",
    ),
    "self_treatment": (
        r"instead of (?:seeing|going to|consulting) (?:a |the )?doctor",
        r"without (?:seeing |going to |consulting )?(?:a |the )?doctor",
        r"\bavoid (?:surgery|the operation|going to hospital|the doctor)\b",
        r"\b(?:cure|treat|manage|fix|handle)\b[\w\s]{0,20}\bat home\b",
        r"\b(?:cure|treat|fix|heal)\b[\w\s]{0,20}\b(?:myself|ourselves|on (?:my|our) own)\b",
        r"\bwithout (?:going to |the )?hospital\b",
        r"\bhome remed(?:y|ies)\b",
        r"\bnatural remed(?:y|ies)\b",
        r"\b(?:herbs?|herbal|ayurvedic|jadibuti)\b",
        r"\btreat (?:it |this |myself|him|her)\b[\w\s]{0,15}\bmyself\b",
        r"\btreat myself\b",
        r"do i really need (?:to see |to go to )?(?:a |the )?(?:doctor|hospital)",
        # Romanised Nepali
        r"\bghar\w*\s+(?:upachar|ilaj)\b",
        r"\bdaktar\s+bina\b",
        # Devanagari: घरमै उपचार (treat at home), डाक्टर बिना (without a doctor)
        r"घर(?:मै|मा|ेलु)\s*(?:उपचार|इलाज)",
        r"(?:डाक्टर|चिकित्सक)\s*बिना",
        r"आफैं?\s*(?:उपचार|इलाज)",
        r"जडिबुटी",
    ),
}

REFUSAL_RESPONSES_EN: dict[str, str] = {
    "dosage": (
        "I cannot advise on medication or dosages - that decision belongs to a "
        "prescribing clinician who knows your full history. Please speak to a "
        "doctor or pharmacist."
    ),
    "prognosis": (
        "I cannot estimate prognosis or survival. Those depend on findings, "
        "staging and individual factors that only your treating specialist can "
        "assess. Please discuss this with a neurologist or oncologist."
    ),
    "self_treatment": (
        "I cannot support managing a suspected brain abnormality without "
        "medical supervision. Please arrange review by a qualified doctor. If "
        "cost or distance is the barrier, ask me about government support "
        "schemes and the nearest neurology centre."
    ),
}

REFUSAL_RESPONSES_NE: dict[str, str] = {
    "dosage": (
        "म औषधि वा यसको मात्रा सम्बन्धी सल्लाह दिन सक्दिनँ - यो निर्णय तपाईंको "
        "पूर्ण स्वास्थ्य इतिहास थाहा भएका चिकित्सकले मात्र गर्नुपर्छ। कृपया "
        "चिकित्सक वा फार्मासिस्टसँग परामर्श गर्नुहोस्।"
    ),
    "prognosis": (
        "म रोगको भविष्यवाणी वा आयु सम्बन्धी अनुमान गर्न सक्दिनँ। यो स्क्यानको "
        "नतिजा, रोगको चरण र व्यक्तिगत अवस्थामा भर पर्छ, जुन तपाईंको उपचार "
        "गर्ने विशेषज्ञले मात्र मूल्याङ्कन गर्न सक्नुहुन्छ।"
    ),
    "self_treatment": (
        "चिकित्सकीय निगरानी बिना मस्तिष्क सम्बन्धी सम्भावित समस्याको उपचार "
        "गर्न म सहयोग गर्न सक्दिनँ। कृपया योग्य चिकित्सकबाट जाँच गराउनुहोस्। "
        "यदि खर्च वा दूरी बाधा हो भने, सरकारी सहयोग कार्यक्रम र नजिकको "
        "न्यूरोलोजी केन्द्रबारे मलाई सोध्नुहोस्।"
    ),
}


_COMPILED_TRIGGERS: dict[str, tuple[re.Pattern[str], ...]] = {
    category: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for category, patterns in REFUSAL_TRIGGERS.items()
}


@dataclass(frozen=True)
class SafetyCheck:
    """Outcome of screening a user question before it reaches the LLM."""

    allowed: bool
    category: str | None = None
    response: str | None = None


def screen_question(question: str, language: Language = "en") -> SafetyCheck:
    """Screen a user question against the refusal triggers.

    Runs *before* retrieval so a prohibited request never reaches the model.

    Args:
        question: Raw user input.
        language: Language for the refusal message.

    Returns:
        A :class:`SafetyCheck`. When ``allowed`` is False, ``response`` holds
        the text to return to the user unchanged.
    """
    haystack = question.casefold()
    for category, patterns in _COMPILED_TRIGGERS.items():
        if any(pattern.search(haystack) for pattern in patterns):
            table = REFUSAL_RESPONSES_NE if language == "ne" else REFUSAL_RESPONSES_EN
            return SafetyCheck(allowed=False, category=category, response=table[category])
    return SafetyCheck(allowed=True)


def get_disclaimer(language: Language = "en", *, short: bool = False) -> str:
    """Return the mandatory disclaimer in the requested language."""
    if short:
        return DISCLAIMER_SHORT_NE if language == "ne" else DISCLAIMER_SHORT_EN
    return DISCLAIMER_NE if language == "ne" else DISCLAIMER_EN


def get_red_flags(language: Language = "en") -> tuple[str, list[str]]:
    """Return ``(instruction, symptoms)`` for the emergency escalation block."""
    if language == "ne":
        return EMERGENCY_INSTRUCTION_NE, list(RED_FLAGS_NE)
    return EMERGENCY_INSTRUCTION_EN, list(RED_FLAGS_EN)


def get_no_context_fallback(language: Language = "en") -> str:
    """Return the safe response used when retrieval finds nothing usable."""
    return NO_CONTEXT_FALLBACK_NE if language == "ne" else NO_CONTEXT_FALLBACK_EN


def append_disclaimer(text: str, language: Language = "en") -> str:
    """Append the disclaimer to generated text if it is not already present.

    The idempotency check matters because the LLM is *also* instructed to
    include a disclaimer; without it, roughly one reply in three ends up
    carrying the warning twice.
    """
    marker = "अस्वीकरण" if language == "ne" else "MEDICAL DISCLAIMER"
    if marker in text:
        return text
    # A blank line, not a "---" rule: the literal dashes leaked into chat
    # bubbles and plain-text surfaces as punctuation noise. Each surface draws
    # its own separator around the disclaimer block.
    return f"{text.rstrip()}\n\n{get_disclaimer(language)}"


__all__ = [
    "DISCLAIMER_EN",
    "DISCLAIMER_NE",
    "DISCLAIMER_SHORT_EN",
    "DISCLAIMER_SHORT_NE",
    "EMERGENCY_CONTACT",
    "EMERGENCY_INSTRUCTION_EN",
    "EMERGENCY_INSTRUCTION_NE",
    "RED_FLAGS_EN",
    "RED_FLAGS_NE",
    "SAFETY_RULES",
    "Language",
    "SafetyCheck",
    "append_disclaimer",
    "get_disclaimer",
    "get_no_context_fallback",
    "get_red_flags",
    "screen_question",
]
