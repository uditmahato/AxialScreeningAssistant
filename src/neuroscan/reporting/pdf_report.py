"""PDF clinical report generation.

The report is laid out so that the things a clinician must not miss appear
first and cannot be cropped away: the disclaimer sits directly under the
header, and the emergency red flags are on page one, not appended at the end.

Every page carries a footer repeating that the document is decision support
rather than a diagnosis, because printed pages get separated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)

from neuroscan.reporting.devanagari import DevanagariRenderer
from neuroscan.safety import get_disclaimer, get_red_flags
from neuroscan.textfmt import strip_wikilinks, to_plain_text
from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.safety import Language

log = get_logger("reporting.pdf")

BRAND = colors.HexColor("#1d4e78")
ACCENT = colors.HexColor("#c0392b")
MUTED = colors.HexColor("#5a6570")
LIGHT = colors.HexColor("#eef2f6")


@dataclass
class ReportData:
    """Everything that goes into one report."""

    prediction: str
    confidence: float
    language: Language = "en"
    architecture: str = "efficientnet_b0"
    threshold: float = 0.5
    scan_image_path: Path | None = None
    heatmap_image_path: Path | None = None
    advisory_text: str = ""
    citations: list[str] = field(default_factory=list)
    heatmap_note: str = ""
    heatmap_is_diffuse: bool = False
    chat_history: list[dict[str, str]] = field(default_factory=list)
    generated_at: datetime | None = None
    report_id: str = ""
    model_metrics: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False

    @property
    def is_abnormal(self) -> bool:
        return self.prediction.strip().lower() not in {"normal", "सामान्य"}

    @property
    def timestamp(self) -> datetime:
        return self.generated_at or datetime.now(UTC)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "NSTitle", parent=base["Title"], fontSize=19, leading=23,
            textColor=BRAND, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "NSSubtitle", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "NSH2", parent=base["Heading2"], fontSize=12.5, leading=16,
            textColor=BRAND, spaceBefore=12, spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "NSH3", parent=base["Heading3"], fontSize=10.5, leading=13,
            textColor=colors.HexColor("#2c3e50"), spaceBefore=8, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "NSBody", parent=base["Normal"], fontSize=9.5, leading=13.5,
            alignment=TA_LEFT, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "NSSmall", parent=base["Normal"], fontSize=8, leading=10.5,
            textColor=MUTED,
        ),
        "disclaimer": ParagraphStyle(
            "NSDisclaimer", parent=base["Normal"], fontSize=8.5, leading=11.5,
            textColor=colors.HexColor("#7a2018"), backColor=colors.HexColor("#fdf2f0"),
            borderColor=ACCENT, borderWidth=1, borderPadding=7, spaceAfter=9,
        ),
        "emergency": ParagraphStyle(
            "NSEmergency", parent=base["Normal"], fontSize=9, leading=12.5,
            textColor=colors.HexColor("#7a2018"),
        ),
    }


def _escape(text: str) -> str:
    """Escape XML special characters for ReportLab's mini-markup parser."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _markdown_to_flowables(text: str, styles: dict[str, ParagraphStyle]) -> list:
    """Convert the light markdown the LLM produces into ReportLab flowables.

    Deliberately minimal - headings, bullets, bold and paragraphs. A full
    markdown engine would be more than this needs and would introduce a
    dependency for no benefit.
    """
    # Headings, bold and bullets are handled below; internal corpus links are
    # not renderable and must not be shown.
    text = strip_wikilinks(text)

    flowables: list = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            joined = " ".join(buffer).strip()
            if joined:
                flowables.append(Paragraph(_inline(joined), styles["body"]))
            buffer.clear()

    def _inline(chunk: str) -> str:
        chunk = _escape(chunk)
        # **bold** -> <b>bold</b>, applied after escaping so literal asterisks
        # in the source cannot inject markup.
        while "**" in chunk:
            chunk = chunk.replace("**", "<b>", 1)
            if "**" in chunk:
                chunk = chunk.replace("**", "</b>", 1)
            else:
                chunk += "</b>"
        return chunk

    bullets: list[str] = []

    def flush_bullets() -> None:
        if bullets:
            flowables.append(
                ListFlowable(
                    [ListItem(Paragraph(_inline(b), styles["body"]), leftIndent=10)
                     for b in bullets],
                    bulletType="bullet", start="•", leftIndent=12,
                )
            )
            bullets.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush()
            flush_bullets()
            continue

        if stripped.startswith("---"):
            flush()
            flush_bullets()
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d8dee4")))
            continue

        if stripped.startswith("#"):
            flush()
            flush_bullets()
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped.lstrip("#").strip()
            flowables.append(Paragraph(_inline(heading), styles["h3" if level >= 3 else "h2"]))
            continue

        if stripped.startswith(("- ", "* ", "• ")):
            flush()
            bullets.append(stripped[2:].strip())
            continue

        if stripped.startswith(">"):
            flush()
            flush_bullets()
            flowables.append(Paragraph(_inline(stripped.lstrip("> ").strip()), styles["small"]))
            continue

        flush_bullets()
        buffer.append(stripped)

    flush()
    flush_bullets()
    return flowables


def _page_furniture(canvas, doc) -> None:
    """Header rule and per-page footer disclaimer."""
    canvas.saveState()
    width, _ = A4

    canvas.setStrokeColor(BRAND)
    canvas.setLineWidth(2)
    canvas.line(18 * mm, 283 * mm, width - 18 * mm, 283 * mm)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        18 * mm, 12 * mm,
        "Axial Screening Assistant - clinical decision support only. Not a diagnosis. "
        "A qualified clinician must review.",
    )
    canvas.drawRightString(width - 18 * mm, 12 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _image_flowable(path: Path, max_width: float, max_height: float) -> RLImage | None:
    """Scale an image to fit a box while preserving aspect ratio."""
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as probe:
            width, height = probe.size
        if not width or not height:
            return None

        ratio = min(max_width / width, max_height / height)
        return RLImage(str(path), width=width * ratio, height=height * ratio)
    except Exception as exc:
        log.warning("Could not embed image %s: %s", path, exc)
        return None


#: Usable frame height for an A4 page with this document's margins, minus a
#: safety margin. A rendered image taller than the frame cannot be split by
#: ReportLab and aborts the build with a LayoutError.
MAX_BLOCK_HEIGHT_PT = 600


def _nepali_blocks(
    renderer: DevanagariRenderer,
    text: str,
    work_dir: Path,
    name: str,
    *,
    width_pt: float = 460,
    font_size_pt: float = 9.5,
) -> list[RLImage]:
    """Render a Nepali passage as one or more page-sized embeddable images.

    The text is flattened to plain prose first. This path rasterises, so unlike
    the English path it cannot hand markdown to a syntax-aware flowable - any
    ``**bold**`` or ``## heading`` left in the source would be drawn literally.
    """
    pages = renderer.render_paged(
        to_plain_text(text), work_dir, name,
        width_pt=width_pt, font_size_pt=font_size_pt,
        max_height_pt=MAX_BLOCK_HEIGHT_PT,
    )
    return [
        RLImage(str(page.path), width=width_pt, height=width_pt * page.aspect)
        for page in pages
    ]


def _nepali_block(
    renderer: DevanagariRenderer,
    text: str,
    work_dir: Path,
    name: str,
    *,
    width_pt: float = 460,
    font_size_pt: float = 9.5,
) -> RLImage | None:
    """Render a short Nepali passage as a single image.

    Only for passages known to be short - a disclaimer, a red-flag list. Use
    :func:`_nepali_blocks` for anything of unbounded length.
    """
    blocks = _nepali_blocks(
        renderer, text, work_dir, name, width_pt=width_pt, font_size_pt=font_size_pt
    )
    return blocks[0] if blocks else None


def build_report(data: ReportData, out_path: Path, *, work_dir: Path | None = None) -> Path:
    """Render the PDF report.

    Args:
        data: Report content.
        out_path: Destination PDF.
        work_dir: Scratch directory for rendered Nepali images. Defaults to a
            sibling of ``out_path``.

    Returns:
        The path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir or out_path.parent / ".report_assets")
    work_dir.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    nepali = data.language == "ne"
    renderer = DevanagariRenderer() if nepali else None

    document = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=18 * mm,
        title=f"Screening Report {data.report_id}".strip(),
        author="Axial Screening Assistant",
        subject="Brain MRI triage support report",
    )

    story: list = []

    # -- Header ------------------------------------------------------------
    story.append(Paragraph("Axial Screening Assistant", styles["title"]))
    story.append(Paragraph("Brain MRI Triage Support Report", styles["subtitle"]))

    # -- Disclaimer, immediately below the header --------------------------
    # Placed here rather than at the end so it cannot be missed or separated
    # from the result it qualifies.
    story.append(Paragraph(f"<b>{_escape(get_disclaimer('en'))}</b>", styles["disclaimer"]))

    if nepali and renderer is not None:
        block = _nepali_block(
            renderer, get_disclaimer("ne"), work_dir, "disclaimer_ne", font_size_pt=8.5
        )
        if block is not None:
            story.append(block)
            story.append(Spacer(1, 5))

    # -- Metadata table ----------------------------------------------------
    meta_rows = [
        ["Report ID", data.report_id or "-"],
        ["Generated", data.timestamp.strftime("%Y-%m-%d %H:%M UTC")],
        ["Model", data.architecture],
        ["Decision threshold", f"{data.threshold:.3f}"],
    ]
    meta_table = Table(meta_rows, colWidths=[42 * mm, 60 * mm])
    meta_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # -- Result ------------------------------------------------------------
    story.append(Paragraph("Analysis result", styles["h2"]))

    verdict_colour = ACCENT if data.is_abnormal else colors.HexColor("#1e7a4b")
    result_table = Table(
        [[
            Paragraph(
                f'<font size="15" color="{verdict_colour.hexval()}"><b>'
                f"{_escape(data.prediction.upper())}</b></font>",
                styles["body"],
            ),
            Paragraph(
                f'<font size="11"><b>{data.confidence:.1%}</b></font><br/>'
                f'<font size="7" color="{MUTED.hexval()}">model confidence</font>',
                styles["body"],
            ),
        ]],
        colWidths=[100 * mm, 40 * mm],
    )
    result_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#c9d4de")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.append(result_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This is a triage signal produced by an automated system, not a diagnosis. "
        "It is based on a single axial image and cannot exclude findings on other "
        "slices, planes or sequences.",
        styles["small"],
    ))
    story.append(Spacer(1, 8))

    # -- Images ------------------------------------------------------------
    images: list = []
    labels: list = []
    if data.scan_image_path and Path(data.scan_image_path).exists():
        flow = _image_flowable(Path(data.scan_image_path), 72 * mm, 72 * mm)
        if flow is not None:
            images.append(flow)
            labels.append(Paragraph("Uploaded scan (preprocessed)", styles["small"]))
    if data.heatmap_image_path and Path(data.heatmap_image_path).exists():
        flow = _image_flowable(Path(data.heatmap_image_path), 72 * mm, 72 * mm)
        if flow is not None:
            images.append(flow)
            labels.append(Paragraph("Grad-CAM: where the model looked", styles["small"]))

    if images:
        story.append(Paragraph("Visual explanation", styles["h2"]))
        image_table = Table([images, labels], colWidths=[76 * mm] * len(images))
        image_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 1), (-1, 1), 3),
            ])
        )
        story.append(image_table)

        note = (
            "The heat map shows which parts of the image most influenced the result. "
            "If the highlighted area is on the skull, the image border or scanner text "
            "rather than brain tissue, the result should not be relied upon."
        )
        if data.heatmap_is_diffuse:
            note += (
                " <b>Attention was diffuse rather than focused on a specific region, "
                "so this explanation does not localise a finding.</b>"
            )
        story.append(Spacer(1, 3))
        story.append(Paragraph(note, styles["small"]))
        story.append(Spacer(1, 8))

    # -- Emergency red flags, on page one ----------------------------------
    instruction, red_flags = get_red_flags("en")
    emergency_items = [
        ListItem(Paragraph(_escape(flag), styles["emergency"]), leftIndent=10)
        for flag in red_flags
    ]
    emergency_block = [
        Paragraph("Seek emergency care immediately if", styles["h2"]),
        Paragraph(_escape(instruction), styles["body"]),
        ListFlowable(emergency_items, bulletType="bullet", start="•", leftIndent=12),
        Spacer(1, 3),
        Paragraph("<b>Emergency numbers in Nepal: Ambulance 102 | Police 100</b>", styles["body"]),
    ]
    story.append(KeepTogether(emergency_block))

    if nepali and renderer is not None:
        ne_instruction, ne_flags = get_red_flags("ne")
        block = _nepali_block(
            renderer,
            ne_instruction + "\n" + "\n".join(f"• {f}" for f in ne_flags),
            work_dir, "redflags_ne", font_size_pt=9,
        )
        if block is not None:
            story.append(Spacer(1, 5))
            story.append(block)

    # -- Advisory ----------------------------------------------------------
    if data.advisory_text:
        story.append(PageBreak())
        story.append(Paragraph("Clinical advisory", styles["h2"]))

        if data.degraded:
            story.append(Paragraph(
                "<i>The language model was unavailable, so verified reference material "
                "is reproduced below without summarisation.</i>",
                styles["small"],
            ))
            story.append(Spacer(1, 4))

        if nepali and renderer is not None:
            # The advisory is of unbounded length, so it must be paged.
            blocks = _nepali_blocks(renderer, data.advisory_text, work_dir, "advisory_ne")
            if blocks:
                story.extend(blocks)
            else:
                story.extend(_markdown_to_flowables(data.advisory_text, styles))
        else:
            story.extend(_markdown_to_flowables(data.advisory_text, styles))

    # -- Citations ---------------------------------------------------------
    if data.citations:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Sources consulted", styles["h2"]))
        story.append(Paragraph(
            "The advisory above was generated only from these verified knowledge base "
            "documents. Review dates are shown because guidance and costs change.",
            styles["small"],
        ))
        story.append(Spacer(1, 3))
        story.append(ListFlowable(
            [ListItem(Paragraph(_escape(c), styles["small"]), leftIndent=10)
             for c in data.citations],
            bulletType="bullet", start="•", leftIndent=12,
        ))

    # -- Chat transcript ---------------------------------------------------
    if data.chat_history:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Questions asked", styles["h2"]))
        for turn in data.chat_history[:10]:
            question = turn.get("question", "")
            answer = turn.get("answer", "")
            if not question:
                continue
            story.append(Paragraph(f"<b>Q:</b> {_escape(question)}", styles["body"]))
            # Answers may be Nepali; render as an image when it would otherwise
            # come out malformed.
            if nepali and renderer is not None:
                blocks = _nepali_blocks(
                    renderer, answer, work_dir,
                    f"chat_{abs(hash(question)) % 100000}", font_size_pt=9,
                )
                if blocks:
                    story.extend(blocks)
                    story.append(Spacer(1, 4))
                    continue
            story.append(Paragraph(f"<b>A:</b> {_escape(answer)}", styles["small"]))
            story.append(Spacer(1, 4))

    # -- Footer ------------------------------------------------------------
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d8dee4")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This system is a research prototype. It is not a certified "
        "medical device and carries no regulatory approval. It must not be used as the "
        "sole basis for any clinical decision. No patient-identifying information is "
        "stored by the system.",
        styles["small"],
    ))

    document.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
    log.info("Report written: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path


__all__ = ["ReportData", "build_report"]
