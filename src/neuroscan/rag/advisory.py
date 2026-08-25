"""Advisory generation - the RAG pipeline that turns a classification into guidance.

The pipeline is deliberately conservative at every branch:

    classification -> build query -> retrieve -> threshold -> generate -> validate

If retrieval returns nothing usable, generation is skipped entirely and a fixed
safe response is returned. If the generated text contains a prohibited pattern -
a definitive diagnosis, a drug dose, a prognosis - it is rejected and replaced.
The system is designed so that its failure mode is being unhelpful, never being
confidently wrong.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from neuroscan.rag.prompts import (
    build_structured_advisory_prompt,
    format_context,
    is_normal_prediction,
)
from neuroscan.safety import (
    Language,
    append_disclaimer,
    get_disclaimer,
    get_no_context_fallback,
    get_red_flags,
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
    from neuroscan.rag.vectorstore import RetrievedChunk, VectorStore

log = get_logger("rag.advisory")

#: Patterns that must never appear in generated clinical text. Each is a
#: documented LLM failure mode on medical prompts, not a hypothetical.
#:
#: **These must cover plain language, not textbook phrasing.** The system
#: prompts instruct the model to "write plainly" for a reader who "may be a
#: community health volunteer", which actively pushes it toward spelled-out
#: units and everyday wording. An earlier version matched "400 mg twice a day"
#: but let "500 milligrams ... taken twice daily" straight through - it caught
#: the phrasing the prompt discourages and missed the one the prompt asks for.
#:
#: **And they must cover Devanagari.** Nepali is a first-class generation
#: language; an English-only validator inverts the safety posture depending on
#: which language the user speaks.
#:
#: Over-matching is cheap here: a rejected response falls back to verbatim
#: source text, which is safe. These run only on generated output, never on the
#: curated corpus, so ordinary clinical prose is not at risk.
PROHIBITED_PATTERNS: list[tuple[str, str]] = [
    # -- Definitive diagnosis ---------------------------------------------
    (r"\byou (?:have|are suffering from)\s+(?:a\s+)?(?:cancer|tumou?r|glioma|glioblastoma)",
     "definitive_diagnosis"),
    # "This scan is definitely a tumour" slipped an earlier version because
    # the subject alternation did not allow a noun after "this".
    (r"\b(?:this(?:\s+(?:scan|image|result))?|the\s+(?:scan|image|result))\s+"
     r"(?:is|shows)\s+(?:definitely|certainly|clearly)\s+(?:a\s+)?(?:cancer|tumou?r)",
     "definitive_diagnosis"),
    # Deliberately narrow. An earlier version matched a bare "diagnosis is",
    # which fired on ordinary clinical prose in the corpus itself - "the
    # diagnosis is usually built from" - and caused safe, curated text to be
    # rejected. Only assertive constructions are prohibited.
    (r"\b(?:the |your |his |her )?diagnosis is (?:definitely|certainly|clearly|confirmed)",
     "definitive_diagnosis"),
    (r"\byou (?:definitely |certainly )?have (?:a |an )?(?:\w+\s+){0,3}?(?:cancer|tumou?r|glioma|glioblastoma|meningioma)\b",
     "definitive_diagnosis"),
    (r"\b(?:i can confirm|this confirms)\b.{0,40}\b(?:cancer|tumou?r)\b",
     "definitive_diagnosis"),
    # Devanagari: "तपाईंलाई ... क्यान्सर/ट्युमर छ" = "you have cancer/tumour"
    (r"तपाईं(?:लाई)?[^।]{0,40}(?:क्यान्सर|ट्युमर|अर्बुद)\s*(?:छ|हो)", "definitive_diagnosis"),
    (r"निश्चित\s*रूपमा[^।]{0,30}(?:क्यान्सर|ट्युमर)", "definitive_diagnosis"),

    # -- Dosage ------------------------------------------------------------
    # Plurals and spelled-out units included; the trailing \b on "milligram"
    # previously made "milligrams" pass.
    (r"\b\d+\s*(?:mg|mcg|ml|g)\b", "dosage"),
    (r"\b(?:\d+|one|two|three|four|five|six)\s*(?:milligram|microgram|millilitre|milliliter|gram)s?\b",
     "dosage"),
    (r"\b(?:take|give|administer|swallow)\s+(?:\d+|one|two|three|a|half)\b[\w\s]{0,15}\b(?:tablet|pill|capsule|dose|drop)s?\b",
     "dosage"),
    (r"\b(?:once|twice|thrice|two times|three times|four times)\s+(?:a |per )?(?:day|daily|week|weekly)\b",
     "dosage"),
    (r"\b(?:daily|nightly|hourly)\s+dose\b", "dosage"),
    (r"\bevery\s+\d+\s*(?:hours?|hrs?|days?)\b", "dosage"),
    (r"\b\d+\s*(?:mg|milligrams?)\s*(?:/|per)\s*kg\b", "dosage"),
    # Devanagari dose units and frequency
    (r"मिलिग्राम|मि\.?\s*ग्रा\.?|मिलीग्राम", "dosage"),
    (r"(?:गोली|चक्की|खुराक)\s*(?:खानु|लिनु|दिनु)", "dosage"),
    (r"दिनको\s*\d*\s*पटक", "dosage"),

    # -- Prognosis ---------------------------------------------------------
    (r"\bsurvival rate\b", "prognosis"),
    (r"\b(?:median|average|mean|overall)\s+survival\b", "prognosis"),
    (r"\b\d+[\s-]*(?:year|month)\s+survival\b", "prognosis"),
    # U+2013 is the en dash, written as an escape rather than as a literal so
    # it cannot be misread as a hyphen in source. Models produce en-dashed
    # ranges as readily as hyphenated ones.
    (r"\b\d+\s*(?:-|to|–)\s*\d+\s*(?:month|year)s?\s+to live\b", "prognosis"),
    (r"\blife expectancy\b", "prognosis"),
    (r"\b\d+\s*%\s+(?:of patients\s+)?(?:survive|survival|live)", "prognosis"),
    (r"\bsurvival\s+is\s+(?:around|about|approximately|roughly)?\s*\d", "prognosis"),
    (r"\bexpect(?:ed)?\s+to\s+live\s+(?:for\s+)?(?:about|around)?\s*\d", "prognosis"),
    # Devanagari: बाँच्ने सम्भावना (chance of surviving), आयु (lifespan)
    (r"बाँच्ने\s*(?:सम्भावना|दर|अवधि)", "prognosis"),
    (r"(?:आयु|जीवन)\s*(?:अवधि|काल)", "prognosis"),
    (r"\d+\s*(?:प्रतिशत|%)\s*बाँच", "prognosis"),
]

_COMPILED_PROHIBITED = [(re.compile(p, re.IGNORECASE), label) for p, label in PROHIBITED_PATTERNS]


@dataclass
class AdvisoryResult:
    """A generated advisory with everything needed to audit it."""

    text: str
    language: Language
    prediction: str
    confidence: float
    #: Structured form when the model returned valid, safe JSON:
    #: {"summary": str, "possible_causes": [{"name", "note"}], "next_steps": [str]}.
    #: None on any parse or safety failure, in which case ``text`` is the only
    #: representation. The interface renders this natively; ``text`` is built
    #: FROM it, so presentation never depends on model-invented formatting.
    structured: dict | None = None
    citations: list[str] = field(default_factory=list)
    retrieved_count: int = 0
    provider: str = ""
    model: str = ""
    latency_seconds: float = 0.0
    degraded: bool = False
    safety_flags: list[str] = field(default_factory=list)
    red_flag_instruction: str = ""
    red_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "structured": self.structured,
            "language": self.language,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "citations": self.citations,
            "retrieved_count": self.retrieved_count,
            "provider": self.provider,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 2),
            "degraded": self.degraded,
            "safety_flags": self.safety_flags,
        }


def validate_generated_text(text: str) -> list[str]:
    """Screen generated text for prohibited clinical content.

    Returns:
        A list of violated categories. Empty means the text passed.
    """
    flags: list[str] = []
    for pattern, label in _COMPILED_PROHIBITED:
        if pattern.search(text) and label not in flags:
            flags.append(label)
    return flags


def parse_structured_advisory(raw: str) -> dict | None:
    """Parse the model's JSON advisory, defensively.

    Native JSON mode makes malformed output rare, not impossible. Returns None
    on anything that does not match the expected shape; the caller degrades to
    source text rather than guessing.
    """
    import json

    if not raw or not raw.strip():
        return None

    text = raw.strip()
    # Tolerate a fenced block even though the prompt forbids it.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    summary = payload.get("summary")
    causes = payload.get("possible_causes")
    steps = payload.get("next_steps")

    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(causes, list) or not isinstance(steps, list):
        return None

    parsed_causes: list[dict[str, str]] = []
    for item in causes[:5]:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            note = str(item.get("note") or "").strip()
            if name:
                parsed_causes.append({"name": name, "note": note})
        elif isinstance(item, str) and item.strip():
            parsed_causes.append({"name": item.strip(), "note": ""})

    parsed_steps = [str(s).strip() for s in steps[:6] if isinstance(s, str) and str(s).strip()]

    if not parsed_steps:
        return None

    return {
        "summary": summary.strip(),
        "possible_causes": parsed_causes,
        "next_steps": parsed_steps,
    }


def _structured_joined_text(structured: dict) -> str:
    """Every model-authored string in one blob, for safety validation."""
    parts = [structured["summary"]]
    for cause in structured["possible_causes"]:
        parts.append(cause["name"])
        parts.append(cause["note"])
    parts.extend(structured["next_steps"])
    return "\n".join(parts)


def clean_structured_advisory(structured: dict) -> dict:
    """Sanitise every field: dashes humanised, wiki-links and markdown removed.

    Runs after safety validation, whose patterns match the raw dash forms.
    """
    def _clean(value: str) -> str:
        return strip_passage_refs(humanise_dashes(to_plain_text(value))).strip()

    return {
        "summary": _clean(structured["summary"]),
        "possible_causes": [
            {"name": _clean(c["name"]), "note": _clean(c["note"])}
            for c in structured["possible_causes"]
        ],
        "next_steps": [_clean(s) for s in structured["next_steps"]],
    }


def enforce_prediction_consistency(structured: dict, prediction: str) -> dict:
    """Drop disease possibilities from a normal result's advisory.

    The prompt asks for an empty causes list on a normal classification, but
    a prompt is a request, not a guarantee, and this one was observed being
    ignored: a reader who had just been told "no abnormal pattern detected"
    faced a numbered list of tumours and infections presented as possible
    causes of that result. Enforced in code so it cannot recur.
    """
    if is_normal_prediction(prediction) and structured.get("possible_causes"):
        log.warning(
            "Removed %d disease possibilities from a normal-result advisory",
            len(structured["possible_causes"]),
        )
        return {**structured, "possible_causes": []}
    return structured


#: Section headings for the canonical text rendering, per language.
_SECTION_HEADINGS = {
    "en": ("What this result means", "Possible causes", "Suggested next steps"),
    "ne": ("यो नतिजाको अर्थ", "सम्भावित कारणहरू", "अब के गर्ने"),
}


def structured_to_markdown(structured: dict, language: Language) -> str:
    """Render the canonical text form of a structured advisory.

    This is what the PDF and any text-only surface consume. It is built by
    this code from the model's content, so its formatting is deterministic:
    fixed heading levels, fixed list style, no model-invented layout.
    """
    meaning, causes_h, steps_h = _SECTION_HEADINGS.get(language, _SECTION_HEADINGS["en"])

    lines: list[str] = [f"## {meaning}", "", structured["summary"], ""]

    if structured["possible_causes"]:
        lines += [f"## {causes_h}", ""]
        for index, cause in enumerate(structured["possible_causes"], start=1):
            note = f": {cause['note']}" if cause["note"] else ""
            lines.append(f"{index}. **{cause['name']}**{note}")
        lines.append("")

    lines += [f"## {steps_h}", ""]
    lines += [f"- {step}" for step in structured["next_steps"]]

    return "\n".join(lines).strip()


def build_retrieval_query(
    prediction: str,
    *,
    language: Language = "en",
    extra_terms: str = "",
) -> str:
    """Compose the retrieval query for a classification result.

    The query is expanded well beyond the bare class label. Embedding a single
    word like "abnormal" retrieves almost nothing useful, because no document
    is *about* the word abnormal. Naming the specific conditions the advisory
    must cover is what pulls the right passages - and it is how the
    Nepal-relevant infective causes are guaranteed a place in the context,
    rather than being left to chance.
    """
    if prediction.lower() in {"abnormal", "tumor", "tumour", "abnormality"}:
        query = (
            "abnormal brain MRI finding: possible causes including "
            "neurocysticercosis, tuberculoma, brain abscess, glioma, meningioma, "
            "metastasis; ring-enhancing lesion differential diagnosis; "
            "recommended next steps, low-cost investigation, referral in Nepal; "
            "emergency warning signs"
        )
    else:
        query = (
            "normal brain MRI result interpretation; what a normal single axial "
            "slice does not exclude; limitations of single slice analysis; "
            "when to seek further assessment despite a normal scan; "
            "headache red flags"
        )

    if extra_terms:
        query = f"{query}; {extra_terms}"
    return query


class AdvisoryEngine:
    """Generates grounded clinical advisories from classification results.

    Args:
        vector_store: The retrieval index.
        llm: The generation backend.
        cfg: Resolved configuration.
    """

    def __init__(self, vector_store: VectorStore, llm: LLMProvider, cfg: Config) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.cfg = cfg

    def generate(
        self,
        *,
        prediction: str,
        confidence: float,
        architecture: str = "efficientnet_b0",
        language: Language = "en",
        heatmap_note: str = "not available",
        extra_terms: str = "",
    ) -> AdvisoryResult:
        """Produce an advisory for a classification result."""
        started = time.perf_counter()

        query = build_retrieval_query(prediction, language=language, extra_terms=extra_terms)
        chunks = self.vector_store.search(query)

        instruction, red_flags = get_red_flags(language)

        # No usable context: return the fixed safe response. Generating from an
        # empty context is exactly how a model is induced to fabricate.
        if not chunks:
            log.warning(
                "No context retrieved above threshold %.2f for prediction %r - "
                "returning the safe fallback rather than generating.",
                self.cfg.rag.score_threshold, prediction,
            )
            return AdvisoryResult(
                text=append_disclaimer(get_no_context_fallback(language), language),
                language=language,
                prediction=prediction,
                confidence=confidence,
                retrieved_count=0,
                provider=self.llm.name,
                degraded=True,
                safety_flags=["no_context"],
                latency_seconds=time.perf_counter() - started,
                red_flag_instruction=instruction,
                red_flags=red_flags,
            )

        # No generative backend: present the retrieved source text directly.
        # Done here rather than inside the provider because this path knows the
        # requested language and holds the chunks, so it produces a properly
        # localised, attributed result instead of an English-only echo.
        if not self.llm.is_generative:
            return self._degraded_from_chunks(
                chunks, prediction, confidence, language, started, instruction, red_flags
            )

        # The model supplies content as JSON; the interface supplies the
        # presentation. This removes the class of defects free-form markdown
        # caused: stray heading levels, model-invented layout, inconsistent
        # bullets.
        system_prompt, user_prompt = build_structured_advisory_prompt(
            prediction=prediction,
            confidence=confidence,
            context=format_context(chunks),
            heatmap_note=heatmap_note,
            language=language,
        )

        try:
            response = self.llm.generate(system_prompt, user_prompt, json_mode=True)
        except Exception as exc:
            log.error("Advisory generation failed: %s", exc)
            return self._degraded_from_chunks(
                chunks, prediction, confidence, language, started, instruction, red_flags
            )

        structured = parse_structured_advisory(response.text)
        if structured is None:
            # A parse failure is a degradation, not an error worth surfacing:
            # the safe response is the attributed source text.
            log.warning("Advisory JSON did not parse; falling back to source text")
            degraded = self._degraded_from_chunks(
                chunks, prediction, confidence, language, started, instruction, red_flags
            )
            degraded.safety_flags = ["structured_parse_failed"]
            return degraded

        # Validate the CONTENT, joined, exactly as the free-form path did.
        joined = _structured_joined_text(structured)
        safety_flags = validate_generated_text(joined)
        if safety_flags:
            # Replace rather than attempt repair. A model that produced a dose
            # or a definitive diagnosis has demonstrably ignored its
            # instructions, and there is no reason to trust the rest of that
            # output either.
            log.error(
                "Generated advisory violated safety rules %s - replacing with source text",
                safety_flags,
            )
            degraded = self._degraded_from_chunks(
                chunks, prediction, confidence, language, started, instruction, red_flags
            )
            degraded.safety_flags = [*safety_flags, "replaced_unsafe_output"]
            return degraded

        structured = enforce_prediction_consistency(
            clean_structured_advisory(structured), prediction
        )
        canonical = structured_to_markdown(structured, language)

        return AdvisoryResult(
            # ``text`` is built FROM the structure by this code, not by the
            # model, so the PDF and every fallback surface get consistent
            # formatting. The disclaimer is appended for document surfaces;
            # the web page shows its own component instead.
            text=append_disclaimer(canonical, language),
            structured=structured,
            language=language,
            prediction=prediction,
            confidence=confidence,
            citations=self._citations(chunks),
            retrieved_count=len(chunks),
            provider=response.provider,
            model=response.model,
            latency_seconds=time.perf_counter() - started,
            degraded=response.degraded,
            red_flag_instruction=instruction,
            red_flags=red_flags,
        )

    def _degraded_from_chunks(
        self,
        chunks: list[RetrievedChunk],
        prediction: str,
        confidence: float,
        language: Language,
        started: float,
        instruction: str,
        red_flags: list[str],
    ) -> AdvisoryResult:
        """Fall back to presenting retrieved source text verbatim.

        Honest and still useful: the health worker sees the same reference
        material the model would have summarised, attributed to its source.
        """
        header = (
            "यो जानकारी ज्ञान-भण्डारबाट सिधै लिइएको हो (सारांश उपलब्ध छैन):"
            if language == "ne"
            else "The following information is taken directly from the knowledge base "
                 "(automatic summarisation was not available):"
        )
        # Corpus text carries markdown and [[wiki-links]] that are meaningful
        # inside the knowledge base and meaningless to a clinician. Without
        # this the report opens on '**Neurocysticercosis**' and
        # 'See [[nepal-neurology-hospitals]]'.
        body = "\n\n".join(
            f"{c.title}\n\n{drop_leading_title(to_plain_text(c.content), c.title)}"
            for c in chunks[:3]
        )
        return AdvisoryResult(
            text=append_disclaimer(f"{header}\n\n{body}", language),
            language=language,
            prediction=prediction,
            confidence=confidence,
            citations=self._citations(chunks),
            retrieved_count=len(chunks),
            provider=self.llm.name,
            degraded=True,
            safety_flags=["generation_unavailable"],
            latency_seconds=time.perf_counter() - started,
            red_flag_instruction=instruction,
            red_flags=red_flags,
        )

    @staticmethod
    def _citations(chunks: list[RetrievedChunk]) -> list[str]:
        """De-duplicated citations, preserving retrieval order."""
        seen: set[str] = set()
        citations: list[str] = []
        for chunk in chunks:
            citation = chunk.citation()
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations


def build_advisory_engine(cfg: Config, *, allow_fallback: bool = True) -> AdvisoryEngine:
    """Construct an :class:`AdvisoryEngine` from configuration."""
    from neuroscan.rag.llm_provider import build_llm_provider
    from neuroscan.rag.vectorstore import load_index

    vector_store = load_index(cfg)
    llm = build_llm_provider(cfg, allow_fallback=allow_fallback)
    return AdvisoryEngine(vector_store, llm, cfg)


__all__ = [
    "PROHIBITED_PATTERNS",
    "AdvisoryEngine",
    "AdvisoryResult",
    "build_advisory_engine",
    "build_retrieval_query",
    "clean_structured_advisory",
    "enforce_prediction_consistency",
    "get_disclaimer",
    "parse_structured_advisory",
    "structured_to_markdown",
    "validate_generated_text",
]
