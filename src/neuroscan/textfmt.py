"""Convert corpus and model markdown into text safe to show a user.

The knowledge base is authored in markdown with ``[[wiki-links]]`` for internal
cross-references. That is right for the corpus - the links keep related
documents discoverable, and the markdown carries structure the retrieval layer
benefits from - but none of it should ever reach a clinician verbatim.

It did. When the language model is unavailable the system falls back to showing
retrieved source text directly, and that path handed raw markdown straight
through: a PDF report opened on ``**Neurocysticercosis**``, ``## Why this
matters``, and ``See [[nepal-neurology-hospitals]]``. The Nepali path was worse,
because it renders to an image and so could not even rely on a markdown-aware
PDF flowable to strip the syntax.

This module is the single sanitisation point for anything derived from corpus
text or model output before display.
"""

from __future__ import annotations

import re

# Ordered: bold/italic markers must go before the bullet rewrite, or a line
# starting "**Note**" briefly looks like a list item.
_WIKILINK = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{2,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BULLET = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HRULE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")


def _humanise_doc_id(doc_id: str) -> str:
    """Turn a corpus id into something readable in prose.

    ``nepal-neurology-hospitals`` -> ``Nepal neurology hospitals``. The link
    target is not shown, because a document id is not an address the reader can
    do anything with.
    """
    words = doc_id.replace("_", "-").split("-")
    if not words:
        return doc_id
    return " ".join([words[0].capitalize(), *words[1:]])


def strip_wikilinks(text: str) -> str:
    """Replace ``[[target]]`` / ``[[target|label]]`` with readable text."""

    def _replace(match: re.Match[str]) -> str:
        target, label = match.group(1), match.group(2)
        return (label or _humanise_doc_id(target)).strip()

    return _WIKILINK.sub(_replace, text)


def to_plain_text(text: str, *, bullet: str = "•") -> str:
    """Flatten markdown to plain prose.

    Structure is preserved where it survives as text - headings keep their line,
    list items keep a bullet - but every syntax marker is removed. Used for the
    Nepali PDF path, which renders to an image and therefore cannot rely on a
    markdown-aware flowable, and for any text shown without further processing.

    Args:
        text: Markdown-ish source.
        bullet: Character used for list items.
    """
    if not text:
        return ""

    out = strip_wikilinks(text)
    out = _TABLE_SEP.sub("", out)
    out = _TABLE_ROW.sub(lambda m: m.group(0).strip().strip("|").replace("|", " · ").strip(), out)
    out = _HRULE.sub("", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _BOLD_ITALIC.sub(r"\2", out)
    out = _HEADING.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _BULLET.sub(rf"\1{bullet} ", out)
    out = _BLANK_RUN.sub("\n\n", out)
    return out.strip()


def drop_leading_title(text: str, title: str) -> str:
    """Remove an opening heading that merely repeats ``title``.

    Every corpus chunk carries the document's own H1 - either because it is the
    first chunk, or because the chunker prepends the title so an isolated chunk
    still identifies itself. The LLM does the same thing: told to write a
    "clinical advisory", it opens with ``**Clinical Advisory**`` under a page
    that already carries that heading. Callers that print the title as a header
    therefore end up showing it twice.

    Matches the first line whether it is a markdown heading, bold-wrapped, or
    bare text.
    """
    if not text or not title:
        return text

    lines = text.lstrip().split("\n")
    if not lines:
        return text

    first = _HEADING.sub("", lines[0])
    first = _BOLD_ITALIC.sub(r"\2", first).strip()
    if first.casefold() == title.strip().casefold():
        return "\n".join(lines[1:]).lstrip()
    return text


_NUMBERED = re.compile(r"^\s*\d+[.)]\s+")

# U+2014 em dash and U+2013 en dash, written escaped so neither can be misread
# as a hyphen in source.
_DASH_ASIDE = re.compile(r"\s+[—–]\s+")
_DASH_RANGE = re.compile(r"(?<=\d)\s*[—–]\s*(?=\d)")


def humanise_dashes(text: str) -> str:
    """Replace em- and en-dash constructions with plainer punctuation.

    The parenthetical em dash ("the lesion — which is common in Nepal — can")
    is the single strongest tell of machine-written prose, and models produce
    it constantly. A comma reads the same and reads human. Numeric ranges keep
    a plain hyphen.

    Applied to generated text at the point it is stored, so every consumer
    (web, PDF, chat history) sees the same cleaned prose. Runs after safety
    validation, whose patterns match both dash forms deliberately.
    """
    if not text:
        return ""
    text = _DASH_RANGE.sub("-", text)
    text = _DASH_ASIDE.sub(", ", text)
    return text.replace("—", "-").replace("–", "-")

#: Retrieval passages are numbered when handed to the model ("[Passage 2]"),
#: and despite instruction it sometimes cites them back ("see Passage 2"). The
#: reader cannot see the passages, so the reference is scaffolding leaking
#: through. Parenthetical citations are removed; a passage named mid-sentence
#: becomes "the reference material", which stays grammatical.
_PASSAGE_PAREN = re.compile(
    r"\s*\((?:see\s+|as in\s+|per\s+)?passages?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*\)",
    re.IGNORECASE,
)
_PASSAGE_LEAD = re.compile(
    r"(?:according to|as (?:stated|described|noted|mentioned) in|as per)\s+"
    r"passages?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*\s*,?\s*",
    re.IGNORECASE,
)
_PASSAGE_BARE = re.compile(r"\bpassages?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*\b", re.IGNORECASE)
_SENTENCE_START_THE = re.compile(r"(^|[.!?]\s+)the reference material")


def strip_passage_refs(text: str) -> str:
    """Remove leaked references to numbered retrieval passages.

    Applied to generated text only, never to the corpus, which does not use
    passage numbering.
    """
    if not text:
        return ""
    text = _PASSAGE_PAREN.sub("", text)
    text = _PASSAGE_LEAD.sub("", text)
    text = _PASSAGE_BARE.sub("the reference material", text)
    return _SENTENCE_START_THE.sub(lambda m: m.group(1) + "The reference material", text)


#: HTML escapes, applied before any markup conversion. The input is model
#: output; a converter that inserted tags before escaping would let generated
#: text inject markup into the page.
_HTML_ESCAPES = [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;")]


def to_safe_html(text: str) -> str:
    """Convert advisory markdown to HTML that is safe to mark ``| safe``.

    Exists because the web result page rendered ``advisory_text`` as plain text
    inside a ``pre-wrap`` block, so every ``##`` and ``**`` the model emitted
    appeared literally on screen. The PDF path converts markdown to flowables;
    the web path needs the same treatment, but as HTML.

    Escape-first, then a deliberately small grammar: headings, bold, bullet and
    numbered lists, horizontal rules, paragraphs. Anything outside it renders
    as text, which is the safe failure.
    """
    if not text:
        return ""

    source = strip_wikilinks(text)
    for char, escaped in _HTML_ESCAPES:
        source = source.replace(char, escaped)

    # Bold after escaping: the asterisks survive escaping untouched.
    def _bold(segment: str) -> str:
        return _BOLD_ITALIC.sub(r"<strong>\2</strong>", segment)

    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_tag = ""

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append("<p>" + _bold(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if list_items:
            blocks.append(f"<{list_tag}>" + "".join(list_items) + f"</{list_tag}>")
            list_items.clear()
        list_tag = ""

    for raw_line in source.splitlines():
        line = raw_line.strip()

        if not line:
            flush_paragraph()
            flush_list()
            continue

        if _HRULE.match(line):
            flush_paragraph()
            flush_list()
            blocks.append("<hr>")
            continue

        heading = _HEADING.match(raw_line)
        if heading:
            flush_paragraph()
            flush_list()
            level = raw_line.lstrip().count("#", 0, 6)
            tag = "h3" if level <= 2 else "h4"
            blocks.append(f"<{tag}>" + _bold(_HEADING.sub("", line)) + f"</{tag}>")
            continue

        if _BULLET.match(raw_line):
            flush_paragraph()
            if list_tag and list_tag != "ul":
                flush_list()
            list_tag = "ul"
            list_items.append("<li>" + _bold(_BULLET.sub("", line)) + "</li>")
            continue

        if _NUMBERED.match(raw_line):
            flush_paragraph()
            if list_tag and list_tag != "ol":
                flush_list()
            list_tag = "ol"
            list_items.append("<li>" + _bold(_NUMBERED.sub("", line)) + "</li>")
            continue

        flush_list()
        paragraph.append(line)

    flush_paragraph()
    flush_list()
    return "\n".join(blocks)


def to_display_markdown(text: str) -> str:
    """Sanitise text that a markdown-aware renderer will handle.

    Keeps headings, bold and bullets - the PDF's markdown converter and the web
    template both understand those - but removes the internal link syntax that
    is meaningless outside the corpus.
    """
    if not text:
        return ""
    return _BLANK_RUN.sub("\n\n", strip_wikilinks(text)).strip()


__all__ = [
    "drop_leading_title",
    "humanise_dashes",
    "strip_passage_refs",
    "strip_wikilinks",
    "to_display_markdown",
    "to_plain_text",
    "to_safe_html",
]
