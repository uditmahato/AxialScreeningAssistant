"""Tests for user-facing text sanitisation.

Regression suite for a defect found by reading a generated PDF rather than by
running a test: corpus markdown and internal ``[[wiki-links]]`` were reaching
the reader verbatim, so a clinical report opened on ``**Neurocysticercosis**``
and ``See [[nepal-neurology-hospitals]]``.
"""

from __future__ import annotations

import pytest

from neuroscan.textfmt import (
    drop_leading_title,
    humanise_dashes,
    strip_wikilinks,
    to_display_markdown,
    to_plain_text,
    to_safe_html,
)


class TestWikilinks:
    def test_bare_link_becomes_readable(self):
        assert strip_wikilinks("See [[nepal-neurology-hospitals]].") == (
            "See Nepal neurology hospitals."
        )

    def test_labelled_link_uses_the_label(self):
        assert strip_wikilinks("See [[tuberculoma|TB of the brain]].") == (
            "See TB of the brain."
        )

    def test_multiple_links_in_one_line(self):
        out = strip_wikilinks("See [[a-doc]] and [[b-doc]].")
        assert "[[" not in out and "]]" not in out

    def test_text_without_links_is_untouched(self):
        text = "A ring-enhancing lesion needs specialist review."
        assert strip_wikilinks(text) == text

    @pytest.mark.parametrize(
        "text",
        [
            "See [[neurocysticercosis]]",
            "**bold** and [[link]]",
            "## Heading\n\nSee [[ring-enhancing-lesion-differential]].",
        ],
    )
    def test_no_link_syntax_survives(self, text):
        assert "[[" not in to_plain_text(text)
        assert "[[" not in to_display_markdown(text)


class TestPlainText:
    def test_strips_bold_keeping_content(self):
        assert to_plain_text("**Neurocysticercosis** is common.") == (
            "Neurocysticercosis is common."
        )

    def test_strips_headings_keeping_the_line(self):
        assert to_plain_text("## Why this matters\n\nBecause.") == (
            "Why this matters\n\nBecause."
        )

    def test_converts_bullets(self):
        out = to_plain_text("- first\n- second")
        assert out == "• first\n• second"

    def test_strips_blockquote_markers(self):
        assert to_plain_text("> This is decision support.") == "This is decision support."

    def test_removes_horizontal_rules(self):
        assert "---" not in to_plain_text("Above\n\n---\n\nBelow")

    def test_unwraps_inline_code_and_links(self):
        assert to_plain_text("Run `pytest` and see [docs](http://x)") == (
            "Run pytest and see docs"
        )

    def test_collapses_excess_blank_lines(self):
        assert "\n\n\n" not in to_plain_text("a\n\n\n\n\nb")

    def test_empty_input(self):
        assert to_plain_text("") == ""

    def test_no_markdown_markers_survive_a_real_corpus_excerpt(self):
        """The exact shape that leaked into the PDF."""
        excerpt = (
            "**Neurocysticercosis**\n\n"
            "# Neurocysticercosis\n\n"
            "## Why this matters for a system used in Nepal\n\n"
            "A ring-enhancing lesion is, on population grounds, **more likely to be "
            "neurocysticercosis than a brain tumour**.\n\n"
            "- **Emergency**: immediate transfer\n"
            "- **Urgent**: neurology referral\n\n"
            "See [[nepal-neurology-hospitals]] and [[nepal-referral-pathway]].\n"
        )
        out = to_plain_text(excerpt)
        for marker in ("**", "##", "[[", "]]"):
            assert marker not in out, f"{marker!r} survived sanitisation"
        # Content is preserved.
        assert "Neurocysticercosis" in out
        assert "Nepal neurology hospitals" in out
        assert "• Emergency: immediate transfer" in out


class TestDropLeadingTitle:
    def test_removes_a_duplicated_heading(self):
        """Every chunk carries the document H1, so a caller printing the title
        as a header would otherwise show it twice."""
        assert drop_leading_title("# Neurocysticercosis\n\nBody.", "Neurocysticercosis") == "Body."

    def test_matches_case_insensitively(self):
        assert drop_leading_title("# GLIOMA - OVERVIEW\n\nBody.", "Glioma - Overview") == "Body."

    def test_leaves_a_different_heading_alone(self):
        text = "# Why this matters\n\nBody."
        assert drop_leading_title(text, "Neurocysticercosis") == text

    def test_leaves_plain_text_alone(self):
        text = "Neurocysticercosis is common in Nepal."
        assert drop_leading_title(text, "Neurocysticercosis") == text

    def test_handles_empty_inputs(self):
        assert drop_leading_title("", "Title") == ""
        assert drop_leading_title("Body", "") == "Body"


class TestStripPassageRefs:
    """Retrieval passages are numbered for the model; the reader never sees
    them, so a cited "Passage 2" is scaffolding leaking through. Observed live
    with llama3.1: "I must refer to Passage 1 for guidance"."""

    def test_removes_parenthetical_citations(self):
        from neuroscan.textfmt import strip_passage_refs

        out = strip_passage_refs("Other slices may have been missed (see Passage 2).")
        assert out == "Other slices may have been missed."

    def test_removes_leading_attribution(self):
        from neuroscan.textfmt import strip_passage_refs

        out = strip_passage_refs("According to Passage 1, lesions can be missed.")
        assert "Passage" not in out
        assert "lesions can be missed" in out

    def test_rewrites_a_bare_reference_grammatically(self):
        from neuroscan.textfmt import strip_passage_refs

        out = strip_passage_refs("I must refer to Passage 1 for guidance on this.")
        assert out == "I must refer to the reference material for guidance on this."

    def test_capitalises_at_sentence_start(self):
        from neuroscan.textfmt import strip_passage_refs

        out = strip_passage_refs("Passage 3 explains the finding.")
        assert out == "The reference material explains the finding."

    def test_leaves_ordinary_prose_alone(self):
        from neuroscan.textfmt import strip_passage_refs

        text = "A passage of time may help. The scan needs review."
        assert strip_passage_refs(text) == text


class TestSafeHtml:
    """Advisory markdown -> HTML for the web result page.

    Regression for the page showing '## What this result means' and
    '**Clinical Advisory**' literally: the template rendered the raw markdown
    in a pre-wrap block with no conversion.
    """

    def test_escapes_html_before_converting(self):
        """The input is model output; markup injection must be impossible."""
        out = to_safe_html('<script>alert("x")</script> and **bold**')
        assert "<script" not in out
        assert "&lt;script&gt;" in out
        assert "<strong>bold</strong>" in out

    def test_headings_become_h3(self):
        out = to_safe_html("## What this result means\n\nBody text.")
        assert "<h3>What this result means</h3>" in out
        assert "<p>Body text.</p>" in out
        assert "##" not in out

    def test_bold_becomes_strong(self):
        assert "<strong>Neurocysticercosis</strong>" in to_safe_html(
            "1. **Neurocysticercosis**: a parasitic infection"
        )

    def test_bullet_list(self):
        out = to_safe_html("* Chest X-ray\n* HIV test")
        assert out.count("<li>") == 2
        assert "<ul>" in out and "</ul>" in out

    def test_numbered_list(self):
        out = to_safe_html("1. Review the scan\n2. Consider investigations")
        assert "<ol>" in out
        assert out.count("<li>") == 2

    def test_horizontal_rule(self):
        assert "<hr>" in to_safe_html("Advice.\n\n---\n\nDISCLAIMER text.")

    def test_wikilinks_removed(self):
        assert "[[" not in to_safe_html("See [[nepal-neurology-hospitals]].")

    def test_empty_input(self):
        assert to_safe_html("") == ""

    def test_real_advisory_shape_has_no_literal_markdown(self):
        """The exact shape from the screenshot that prompted this."""
        advisory = (
            "**Clinical Advisory**\n\n"
            "## What this result means\n"
            "This brain MRI scan has been classified as normal.\n\n"
            "## Recommended next steps\n"
            "* Chest X-ray for tuberculosis\n"
            "* HIV test\n\n"
            "---\n"
            "MEDICAL DISCLAIMER: Axial Screening Assistant is a research prototype."
        )
        out = to_safe_html(drop_leading_title(advisory, "Clinical Advisory"))
        for marker in ("**", "## ", "* "):
            assert marker not in out, f"{marker!r} leaked into the HTML"
        assert "<h3>" in out and "<ul>" in out and "<hr>" in out


class TestHumaniseDashes:
    """Em- and en-dash removal from generated text.

    The parenthetical em dash is the strongest single tell of machine-written
    prose, and the model produces it constantly. Applied at generation time so
    web, PDF and chat history all store the same cleaned text.
    """

    def test_aside_becomes_comma(self):
        assert humanise_dashes("The lesion — which is common — can heal.") == (
            "The lesion, which is common, can heal."
        )

    def test_en_dash_aside(self):
        assert humanise_dashes("Both causes – infection and tumour – matter.") == (
            "Both causes, infection and tumour, matter."
        )

    def test_numeric_range_keeps_a_hyphen(self):
        assert humanise_dashes("Travel takes 8–12 hours.") == "Travel takes 8-12 hours."

    def test_unspaced_dash_becomes_hyphen(self):
        assert humanise_dashes("a first—seizure event") == "a first-seizure event"

    def test_plain_text_untouched(self):
        text = "Take the scan to the district hospital."
        assert humanise_dashes(text) == text

    def test_empty(self):
        assert humanise_dashes("") == ""


class TestDropLeadingTitleBoldVariant:
    def test_drops_a_bold_wrapped_title(self):
        """The LLM opens with '**Clinical Advisory**' under a page heading
        that already says Clinical advisory."""
        assert drop_leading_title(
            "**Clinical Advisory**\n\n## Body", "Clinical Advisory"
        ).startswith("## Body")


class TestDisplayMarkdown:
    def test_keeps_structure_for_a_markdown_renderer(self):
        out = to_display_markdown("## Heading\n\n**bold** text\n\n- item")
        assert "##" in out
        assert "**" in out
        assert "- item" in out

    def test_still_removes_wikilinks(self):
        assert "[[" not in to_display_markdown("See [[a-doc]].")

    def test_empty_input(self):
        assert to_display_markdown("") == ""
