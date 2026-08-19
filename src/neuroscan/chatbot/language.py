"""Language detection and the bilingual interface string table.

Detection handles three cases, in decreasing order of reliability:

1. **Devanagari script** - unambiguous, and the dominant case for typed Nepali.
2. **Romanised Nepali** - Nepali written in Latin script ("tauko dukhyo",
   "ke garne"). Very common in practice, since many users lack a Devanagari
   keyboard, and completely invisible to script detection.
3. **English** - the default.

Romanised Nepali is handled with a keyword list rather than a statistical
model. That is a deliberate trade: a general-purpose language identifier
performs poorly on short, code-mixed clinical questions, whereas a curated list
of high-frequency Nepali function words is transparent, debuggable and
correctable by the project's own users.
"""

from __future__ import annotations

import re
import unicodedata

from neuroscan.safety import Language
from neuroscan.utils import get_logger

log = get_logger("chatbot.language")

# Devanagari block, plus the extended block used for some Nepali characters.
DEVANAGARI_RANGES = ((0x0900, 0x097F), (0xA8E0, 0xA8FF))

#: High-frequency Nepali words as typically romanised. Chosen for being
#: distinctive - words that are also common English words are excluded, so
#: "ma" (I) and "cha" are present but are weighted by requiring more than one
#: match before a verdict is reached.
ROMANISED_NEPALI_MARKERS = frozenset({
    # question words and pronouns
    "ke", "kasari", "kina", "kahile", "kaha", "kun", "kati", "ko", "kasto",
    "malai", "mero", "hamro", "tapai", "tapain", "tapailai", "usko", "uslai",
    # very common verbs and particles
    "ho", "hoina", "cha", "chha", "chaina", "chhaina", "garne", "garnu",
    "garna", "bhayo", "bhaye", "huncha", "hunchha", "parcha", "parchha",
    "lagyo", "laagyo", "sakcha", "sakchha", "pani", "matra", "tara", "ani",
    "bhane", "hola", "thiyo", "gareko", "bhako", "bhaeko",
    # clinical and everyday vocabulary
    "aspatal", "aushadhi", "ausadhi", "rog", "bimar", "birami", "upachar",
    "tauko", "dukhyo", "dukheko", "dukhcha", "jaanch", "janch", "paisa",
    "kharcha", "doctor", "daktar", "swasthya", "mastishka", "gantho",
    "risk", "napar", "napare", "jane", "janu", "aaunu", "khanu",
})

#: Words that look Nepali-ish but are ordinary English, kept out of the marker
#: set to avoid false positives on English clinical text.
_AMBIGUOUS = frozenset({"ko", "ke", "ho", "cha", "pani", "ani", "risk", "doctor"})

_STRONG_MARKERS = ROMANISED_NEPALI_MARKERS - _AMBIGUOUS

_WORD_PATTERN = re.compile(r"[a-zऀ-ॿ]+")


def devanagari_ratio(text: str) -> float:
    """Fraction of letter characters that are Devanagari."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    devanagari = sum(
        1 for c in letters
        if any(lo <= ord(c) <= hi for lo, hi in DEVANAGARI_RANGES)
    )
    return devanagari / len(letters)


def detect_language(
    text: str,
    *,
    default: Language = "en",
    devanagari_threshold: float = 0.20,
) -> Language:
    """Detect whether text is Nepali or English.

    Args:
        text: User input.
        default: Returned for empty input or when no signal is found.
        devanagari_threshold: Fraction of Devanagari letters above which the
            text is treated as Nepali. Set low deliberately - a question that
            is mostly an English drug name with a few Nepali words is still a
            Nepali question and should be answered in Nepali.

    Returns:
        ``'ne'`` or ``'en'``.
    """
    if not text or not text.strip():
        return default

    ratio = devanagari_ratio(text)
    if ratio >= devanagari_threshold:
        return "ne"

    words = set(_WORD_PATTERN.findall(text.lower()))
    if not words:
        return default

    strong_hits = words & _STRONG_MARKERS
    all_hits = words & ROMANISED_NEPALI_MARKERS

    # Two independent signals must agree before declaring romanised Nepali:
    # at least one distinctive marker, and enough overall density that a stray
    # loanword in an English sentence does not flip the verdict.
    density = len(all_hits) / max(len(words), 1)
    if strong_hits and (len(strong_hits) >= 2 or density >= 0.25):
        log.debug("Detected romanised Nepali (markers=%s, density=%.2f)", strong_hits, density)
        return "ne"

    return default


# ---------------------------------------------------------------------------
# Interface strings
#
# Held here rather than in templates so that the web UI, the PDF report and the
# chatbot cannot drift apart in their wording.
# ---------------------------------------------------------------------------

UI_STRINGS: dict[str, dict[Language, str]] = {
    "app_title": {"en": "Axial Screening Assistant", "ne": "एक्सियल स्क्रिनिङ सहायक"},
    "tagline": {
        "en": "Brain MRI screening support",
        "ne": "मस्तिष्क MRI स्क्रिनिङ सहयोग",
    },
    "upload_title": {"en": "Upload a brain scan", "ne": "मस्तिष्क स्क्यान अपलोड गर्नुहोस्"},
    "upload_hint": {
        "en": "Axial brain MRI, JPG or PNG. Crop out the patient's name before you upload.",
        "ne": "अक्षीय (Axial) मस्तिष्क MRI, JPG वा PNG। अपलोड गर्नुअघि बिरामीको नाम काटेर हटाउनुहोस्।",
    },
    "drop_or_choose": {
        "en": "Drop an image here or choose a file",
        "ne": "तस्बिर यहाँ तान्नुहोस् वा फाइल छान्नुहोस्",
    },
    "choose_image": {"en": "Choose image", "ne": "तस्बिर छान्नुहोस्"},
    "ready_to_analyse": {"en": "Ready to analyse", "ne": "विश्लेषणका लागि तयार"},
    "analyse": {"en": "Analyse scan", "ne": "स्क्यान विश्लेषण गर्नुहोस्"},
    "analysing": {"en": "Analysing scan", "ne": "स्क्यान विश्लेषण हुँदैछ"},
    "analysing_hint": {
        "en": "This usually takes a few seconds.",
        "ne": "सामान्यतया केही सेकेन्ड लाग्छ।",
    },
    "stage_checking": {"en": "Checking image", "ne": "तस्बिर जाँच हुँदैछ"},
    "stage_classifier": {"en": "Running classifier", "ne": "वर्गीकरण चल्दैछ"},
    "stage_preparing": {"en": "Preparing result", "ne": "नतिजा तयार हुँदैछ"},
    "result": {"en": "Result", "ne": "नतिजा"},
    "normal": {"en": "Normal", "ne": "सामान्य"},
    "abnormal": {"en": "Abnormal", "ne": "असामान्य"},
    "confidence": {"en": "Model confidence", "ne": "मोडेलको विश्वसनीयता"},
    "heatmap_title": {"en": "Where the model looked", "ne": "मोडेलले हेरेको क्षेत्र"},
    "heatmap_hint": {
        "en": "Check where the highlight sits. If it is on the skull, the image edge or scanner text instead of brain tissue, do not trust this result.",
        "ne": "हाइलाइट कहाँ छ हेर्नुहोस्। मस्तिष्कको तन्तुमा नभई खोपडी, तस्बिरको किनारा वा स्क्यानरको अक्षरमा छ भने यो नतिजामा भर नपर्नुहोस्।",
    },
    "advisory_title": {"en": "Clinical advisory", "ne": "चिकित्सकीय सल्लाह"},
    "sources": {"en": "Sources", "ne": "स्रोतहरू"},
    "emergency_title": {"en": "Seek emergency care if", "ne": "तुरुन्तै अस्पताल जानुहोस् यदि"},
    "chat_title": {"en": "Questions about this result", "ne": "यस नतिजाबारे प्रश्नहरू"},
    "chat_placeholder": {
        "en": "Type your question...",
        "ne": "तपाईंको प्रश्न लेख्नुहोस्...",
    },
    "ask_label": {
        "en": "Ask a question about this result",
        "ne": "यस नतिजाबारे प्रश्न सोध्नुहोस्",
    },
    "send": {"en": "Send", "ne": "पठाउनुहोस्"},
    "download_report": {"en": "Download PDF report", "ne": "PDF रिपोर्ट डाउनलोड गर्नुहोस्"},
    "new_scan": {"en": "Analyse another scan", "ne": "अर्को स्क्यान विश्लेषण गर्नुहोस्"},
    "language": {"en": "नेपाली", "ne": "English"},
    "not_a_diagnosis": {
        "en": "This is not a diagnosis",
        "ne": "यो निदान होइन",
    },
    "error_no_file": {"en": "Please choose an image file.", "ne": "कृपया तस्बिर छान्नुहोस्।"},
    "error_bad_type": {
        "en": "Unsupported file type. Please upload a JPG or PNG image.",
        "ne": "यो फाइल प्रकार समर्थित छैन। कृपया JPG वा PNG तस्बिर अपलोड गर्नुहोस्।",
    },
    "error_too_large": {
        "en": "That file is too large.",
        "ne": "फाइल धेरै ठूलो छ।",
    },
    "error_not_brain": {
        "en": "This does not appear to be a brain MRI image. Please check and upload an axial brain MRI.",
        "ne": "यो मस्तिष्कको MRI जस्तो देखिँदैन। कृपया जाँच गरी अक्षीय मस्तिष्क MRI अपलोड गर्नुहोस्।",
    },
    "error_model_missing": {
        "en": "No trained model is available. The system administrator must train a model first.",
        "ne": "कुनै प्रशिक्षित मोडेल उपलब्ध छैन। प्रणाली प्रशासकले पहिले मोडेल तयार गर्नुपर्छ।",
    },
    "processing_time": {"en": "Processed in", "ne": "प्रशोधन समय"},
    "model_label": {"en": "Model", "ne": "मोडेल"},
    "degraded_notice": {
        "en": "The language model is unavailable, so reference material is shown directly.",
        "ne": "भाषा मोडेल उपलब्ध नभएकाले सन्दर्भ सामग्री सिधै देखाइएको छ।",
    },
    # --- result page, redesigned -----------------------------------------
    "verdict_abnormal": {"en": "Abnormal pattern detected", "ne": "असामान्य ढाँचा भेटियो"},
    "verdict_normal": {"en": "No abnormal pattern detected", "ne": "असामान्य ढाँचा भेटिएन"},
    "verdict_abnormal_note": {
        "en": "The classifier detected a pattern that should be reviewed by a qualified clinician.",
        "ne": "वर्गीकरणकर्ताले योग्य चिकित्सकबाट पुनरावलोकन गर्नुपर्ने ढाँचा भेट्यो।",
    },
    "verdict_normal_note": {
        "en": "The classifier did not detect an abnormal pattern above its decision threshold. Clinical review may still be required.",
        "ne": "वर्गीकरणकर्ताले निर्णय सीमाभन्दा माथि असामान्य ढाँचा भेटेन। चिकित्सकीय पुनरावलोकन भने अझै आवश्यक हुन सक्छ।",
    },
    "confidence_note": {
        "en": "Shows how strongly the classifier favoured this result. It is not the probability of a specific disease.",
        "ne": "वर्गीकरणकर्ता यो नतिजामा कति ढुक्क छ भन्ने मात्र देखाउँछ। यो कुनै खास रोग हुने सम्भावना होइन।",
    },
    "analysis_details": {"en": "Analysis details", "ne": "विश्लेषण विवरण"},
    "tab_original": {"en": "Original", "ne": "मूल तस्बिर"},
    "tab_highlighted": {"en": "Highlighted areas", "ne": "हाइलाइट गरिएका भाग"},
    "heatmap_what": {
        "en": "Regions that contributed most strongly to the model's classification.",
        "ne": "मोडेलको वर्गीकरणमा सबैभन्दा धेरै योगदान गरेका भागहरू।",
    },
    "heatmap_not_disease": {
        "en": "This does not identify the location of a specific disease.",
        "ne": "यसले कुनै रोगको ठाउँ पहिचान गर्दैन।",
    },
    "model_result_label": {"en": "Model result", "ne": "मोडेलको नतिजा"},
    "clinical_context_label": {"en": "Clinical context", "ne": "चिकित्सकीय सन्दर्भ"},
    "clinical_context_note": {
        "en": "Generated from the model result and referenced medical sources. The classifier itself only separates normal from abnormal.",
        "ne": "मोडेलको नतिजा र सन्दर्भ चिकित्सा स्रोतबाट तयार गरिएको। वर्गीकरणकर्ताले सामान्य र असामान्य मात्र छुट्याउँछ।",
    },
    "possible_causes": {"en": "Possible causes", "ne": "सम्भावित कारणहरू"},
    "next_steps": {"en": "Suggested next steps", "ne": "अब के गर्ने"},
    "urgent_review": {"en": "Urgent medical review", "ne": "तुरुन्त चिकित्सकीय जाँच"},
    "sources_used": {"en": "Sources used", "ne": "प्रयोग गरिएका स्रोतहरू"},
    "your_question": {"en": "Your question", "ne": "तपाईंको प्रश्न"},
    "clinical_information": {"en": "Clinical information", "ne": "चिकित्सकीय जानकारी"},
    "preparing_answer": {"en": "Preparing answer...", "ne": "उत्तर तयार हुँदैछ..."},
    "nav_scan": {"en": "Scan", "ne": "स्क्यान"},
    "nav_history": {"en": "Previous scans", "ne": "अघिल्ला स्क्यान"},
    "nav_about": {"en": "About the model", "ne": "मोडेलबारे"},
    # NOTE: disclaimer text deliberately does not live here. It is owned by
    # neuroscan.safety.get_disclaimer() and reaches the templates as
    # `disclaimer` / `disclaimer_short`. Duplicating it in this table means
    # two sources of truth for the one string that must never drift.
    "change_image": {"en": "Change image", "ne": "तस्बिर बदल्नुहोस्"},
    "what_model_checks": {"en": "What this model checks", "ne": "यो मोडेलले के जाँच्छ"},
    "no_scans_yet": {
        "en": "No previous scans",
        "ne": "अघिल्ला स्क्यान छैनन्",
    },
    # --- portal chrome ----------------------------------------------------
    "nav_help": {"en": "Help", "ne": "सहयोग"},
    "skip_to_content": {"en": "Skip to content", "ne": "मुख्य सामग्रीमा जानुहोस्"},
    "alt_scan": {
        "en": "Uploaded brain MRI, preprocessed",
        "ne": "अपलोड गरिएको मस्तिष्क MRI, पूर्वप्रशोधित",
    },
    "alt_heatmap": {
        "en": "Heat map showing which regions of the scan the model weighted most heavily",
        "ne": "मोडेलले स्क्यानका कुन भागलाई बढी महत्त्व दियो देखाउने हिट म्याप",
    },
    "no_matching_scans": {
        "en": "No scans match this filter.",
        "ne": "यो फिल्टरसँग मिल्ने स्क्यान छैन।",
    },
    "file_too_large": {
        "en": "That image is too large to upload. Choose a smaller file.",
        "ne": "यो तस्बिर अपलोड गर्न धेरै ठूलो छ। सानो फाइल छान्नुहोस्।",
    },
    "hero_lede": {
        "en": "Screen a brain MRI for patterns classified as Normal or Abnormal.",
        "ne": "मस्तिष्क MRI मा सामान्य वा असामान्य वर्गीकरण हुने ढाँचा जाँच्नुहोस्।",
    },
    "hero_tag": {
        "en": "Clinical decision support. Not a diagnosis.",
        "ne": "चिकित्सकीय निर्णय सहयोग। निदान होइन।",
    },
    "footer_tagline": {
        "en": "Clinical decision support for brain MRI screening.",
        "ne": "मस्तिष्क MRI स्क्रिनिङका लागि चिकित्सकीय निर्णय सहयोग।",
    },
    # --- landing ----------------------------------------------------------
    "two_classes": {"en": "2 trained classes", "ne": "२ प्रशिक्षित वर्ग"},
    "normal_class_note": {
        "en": "No abnormal pattern detected by the classifier.",
        "ne": "वर्गीकरणकर्ताले असामान्य ढाँचा भेटेन।",
    },
    "abnormal_class_note": {
        "en": "A pattern was flagged for clinical review.",
        "ne": "चिकित्सकीय पुनरावलोकनका लागि ढाँचा फ्ल्याग गरियो।",
    },
    "single_image": {"en": "Single axial MRI image", "ne": "एउटा अक्षीय MRI तस्बिर"},
    "no_specific_disease": {
        "en": "Does not identify a specific disease.",
        "ne": "कुनै खास रोग पहिचान गर्दैन।",
    },
    "learn_model": {"en": "Learn more about the model", "ne": "मोडेलबारे थप जान्नुहोस्"},
    "before_uploading": {"en": "Before uploading", "ne": "अपलोड गर्नुअघि"},
    "urgent_help_title": {
        "en": "Need urgent medical help?",
        "ne": "तुरुन्त चिकित्सा सहायता चाहिन्छ?",
    },
    "urgent_help_body": {
        "en": "A new seizure, loss of consciousness or rapidly worsening neurological symptoms need urgent clinical assessment.",
        "ne": "नयाँ कम्पन (दौरा), होस गुम्नु वा छिटो बिग्रँदै गएका स्नायु लक्षणहरूलाई तुरुन्त चिकित्सकीय मूल्याङ्कन चाहिन्छ।",
    },
    "view_warning_signs": {"en": "View warning signs", "ne": "खतराका संकेतहरू हेर्नुहोस्"},
    # --- result and history -----------------------------------------------
    "print": {"en": "Print", "ne": "प्रिन्ट गर्नुहोस्"},
    "advisory_limited_title": {
        "en": "Clinical advisory currently limited",
        "ne": "चिकित्सकीय सल्लाह अहिले सीमित छ",
    },
    "advisory_limited_body": {
        "en": "The scan result is available, but the expanded clinical explanation could not be generated.",
        "ne": "स्क्यानको नतिजा उपलब्ध छ, तर विस्तृत चिकित्सकीय व्याख्या तयार गर्न सकिएन।",
    },
    "search_scan": {"en": "Search scan ID", "ne": "स्क्यान ID खोज्नुहोस्"},
    "filter_all": {"en": "All", "ne": "सबै"},
    "view": {"en": "View", "ne": "हेर्नुहोस्"},
    "history_empty_hint": {
        "en": "Analysed scans will appear here during this session.",
        "ne": "विश्लेषण गरिएका स्क्यानहरू यही सत्रभर यहाँ देखिनेछन्।",
    },
    "chat_failed": {
        "en": "The question could not be answered just now. Please try again.",
        "ne": "अहिले प्रश्नको उत्तर दिन सकिएन। कृपया फेरि प्रयास गर्नुहोस्।",
    },
    "heatmap_unavailable": {
        "en": "No attention map could be generated for this scan.",
        "ne": "यस स्क्यानका लागि ध्यान नक्सा तयार गर्न सकिएन।",
    },
    "heatmap_unavailable_note": {
        "en": "This is a limitation of the explanation step. It does not change the classification result.",
        "ne": "यो व्याख्या चरणको सीमा हो। यसले वर्गीकरणको नतिजा बदल्दैन।",
    },
}


def get_ui_strings(language: Language = "en") -> dict[str, str]:
    """Return the full interface string table for a language."""
    return {key: values.get(language, values["en"]) for key, values in UI_STRINGS.items()}


def t(key: str, language: Language = "en") -> str:
    """Translate a single interface key, falling back to the key itself."""
    entry = UI_STRINGS.get(key)
    if entry is None:
        log.warning("Missing UI string for key %r", key)
        return key
    return entry.get(language, entry["en"])


__all__ = [
    "DEVANAGARI_RANGES",
    "ROMANISED_NEPALI_MARKERS",
    "UI_STRINGS",
    "detect_language",
    "devanagari_ratio",
    "get_ui_strings",
    "t",
]
