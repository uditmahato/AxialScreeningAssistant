"""Correct Devanagari rendering for PDF output.

**The problem.** ReportLab draws text by mapping characters to glyphs through
the font's cmap, one at a time. Devanagari does not work that way. It needs
complex text shaping: the short-i vowel sign is typed after its consonant but
must be *drawn before* it, consonant clusters combine into conjunct ligatures,
and vowel marks position above and below the baseline. Handing Devanagari to a
renderer with no shaping engine produces text that is visibly wrong to any
Nepali reader - reordered vowels and broken conjuncts - which in a clinical
document is worse than not offering Nepali at all.

**The approach taken here.** Nepali passages are rendered to images with
Pillow, which applies the font's own layout tables, and the resulting images
are embedded in the PDF. Rendering was verified visually on this platform
before the approach was adopted. English text continues to use ordinary PDF
text so it stays selectable and searchable.

The trade-off is that Nepali text in the report is not selectable or
searchable. That is an acceptable price for it being *correct*, and it is
recorded here so the decision is not mistaken for an oversight.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from neuroscan.utils import get_logger

log = get_logger("reporting.devanagari")

#: Fonts known to carry Devanagari coverage, in preference order.
FONT_CANDIDATES: dict[str, list[str]] = {
    "Windows": [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\mangal.ttf",
        r"C:\Windows\Fonts\aparaj.ttf",
        r"C:\Windows\Fonts\utsaah.ttf",
        r"C:\Windows\Fonts\Kokila.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        "/usr/share/fonts/truetype/fonts-deva-extra/samanata.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ],
    "Darwin": [
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
        "/Library/Fonts/Kohinoor.ttc",
    ],
}


class DevanagariUnavailableError(RuntimeError):
    """Raised when no Devanagari-capable font can be located."""


def find_devanagari_font(extra_paths: list[Path] | None = None) -> Path | None:
    """Locate a Devanagari-capable TrueType font.

    Also checks a ``resources/fonts`` directory inside the package, so a font
    can be shipped with the project for deployment onto a machine that has
    none installed.
    """
    bundled = Path(__file__).resolve().parent.parent / "resources" / "fonts"
    search: list[Path] = []

    if extra_paths:
        search.extend(Path(p) for p in extra_paths)
    if bundled.exists():
        search.extend(sorted(bundled.glob("*.tt[fc]")))
    search.extend(Path(p) for p in FONT_CANDIDATES.get(platform.system(), []))

    for path in search:
        if path.exists() and path.is_file():
            log.debug("Using Devanagari font: %s", path)
            return path

    log.warning(
        "No Devanagari-capable font found on this system. Nepali text will be "
        "omitted from PDF reports. Install one (e.g. Noto Sans Devanagari) or "
        "place a .ttf in src/neuroscan/resources/fonts/."
    )
    return None


@dataclass
class RenderedText:
    """A rendered Nepali text block, ready to embed."""

    path: Path
    width_px: int
    height_px: int

    @property
    def aspect(self) -> float:
        return self.height_px / self.width_px if self.width_px else 1.0


class DevanagariRenderer:
    """Renders Nepali text to images for embedding in a PDF.

    Args:
        font_path: TrueType font with Devanagari coverage. Located
            automatically when omitted.
        scale: Supersampling factor. Text is rendered at this multiple of the
            target size and downscaled by the PDF, which is what keeps small
            Devanagari marks - the vowel signs above and below the baseline -
            legible rather than turning into single aliased pixels.
    """

    def __init__(self, font_path: Path | None = None, *, scale: int = 3) -> None:
        self.font_path = font_path or find_devanagari_font()
        self.scale = max(1, scale)
        self._available = self.font_path is not None

    @property
    def available(self) -> bool:
        return self._available

    def _font(self, size_pt: float):
        from PIL import ImageFont

        if self.font_path is None:
            raise DevanagariUnavailableError("No Devanagari font is available")
        return ImageFont.truetype(str(self.font_path), int(size_pt * self.scale))

    def _wrap(self, text: str, font, max_width_px: int) -> list[str]:
        """Greedy word wrap measured against the actual rendered glyph widths.

        Character counting is not usable here: Devanagari glyph widths vary
        considerably, and a conjunct occupies far less width than the number of
        code points suggests.
        """
        from PIL import Image, ImageDraw

        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        lines: list[str] = []
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue

            current = ""
            for word in paragraph.split():
                candidate = f"{current} {word}".strip()
                width = probe.textbbox((0, 0), candidate, font=font)[2]
                if width <= max_width_px or not current:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
        return lines

    def render(
        self,
        text: str,
        out_path: Path,
        *,
        width_pt: float = 460,
        font_size_pt: float = 10,
        line_spacing: float = 1.55,
        colour: str = "#1a1a1a",
        background: str = "#ffffff",
    ) -> RenderedText | None:
        """Render a Nepali passage to a PNG.

        Args:
            text: Nepali text, may contain newlines.
            out_path: Destination PNG.
            width_pt: Target width in PDF points.
            font_size_pt: Target font size in points.
            line_spacing: Multiple of font size between baselines. Devanagari
                needs more than Latin because marks extend above and below.
            colour: Text colour.
            background: Background colour.

        Returns:
            A :class:`RenderedText`, or None if no font is available.
        """
        if not self._available:
            return None

        from PIL import Image, ImageDraw

        try:
            font = self._font(font_size_pt)
            max_width_px = int(width_pt * self.scale)
            lines = self._wrap(text, font, max_width_px)

            line_height = int(font_size_pt * line_spacing * self.scale)
            padding = int(4 * self.scale)
            height = max(line_height, len(lines) * line_height) + padding * 2

            image = Image.new("RGB", (max_width_px + padding * 2, height), background)
            draw = ImageDraw.Draw(image)

            y = padding
            for line in lines:
                if line:
                    draw.text((padding, y), line, font=font, fill=colour)
                y += line_height

            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path, "PNG", optimize=True)

            return RenderedText(path=out_path, width_px=image.width, height_px=image.height)

        except Exception as exc:
            log.error("Failed to render Devanagari text: %s", exc)
            return None

    def render_paged(
        self,
        text: str,
        out_dir: Path,
        name_prefix: str,
        *,
        width_pt: float = 460,
        font_size_pt: float = 9.5,
        line_spacing: float = 1.55,
        max_height_pt: float = 620,
        **kwargs,
    ) -> list[RenderedText]:
        """Render a passage as one or more images, each short enough to fit a page.

        A rendered image is an indivisible flowable: ReportLab cannot split one
        across a page break, and a taller-than-frame image aborts the whole
        document with a ``LayoutError``. Long Nepali advisories exceed a single
        page routinely, so the text is split into page-sized blocks *before*
        rendering.

        Splitting happens at wrapped-line boundaries rather than by cropping
        the image, so no line is cut in half.

        Args:
            text: Nepali text.
            out_dir: Directory for the rendered images.
            name_prefix: Base filename; pages get ``_p0``, ``_p1``, ...
            max_height_pt: Maximum height of any one image, in points. Should
                be a little under the usable frame height.

        Returns:
            One :class:`RenderedText` per page block, in order. Empty if no
            font is available.
        """
        if not self._available:
            return []

        try:
            font = self._font(font_size_pt)
            lines = self._wrap(text, font, int(width_pt * self.scale))
        except Exception as exc:
            log.error("Failed to measure Devanagari text: %s", exc)
            return []

        line_height_pt = font_size_pt * line_spacing
        lines_per_page = max(1, int((max_height_pt - 12) / line_height_pt))

        pages: list[RenderedText] = []
        for index in range(0, len(lines), lines_per_page):
            block = "\n".join(lines[index : index + lines_per_page])
            rendered = self.render(
                block,
                out_dir / f"{name_prefix}_p{index // lines_per_page}.png",
                width_pt=width_pt,
                font_size_pt=font_size_pt,
                line_spacing=line_spacing,
                **kwargs,
            )
            if rendered is not None:
                pages.append(rendered)

        if len(pages) > 1:
            log.debug("Nepali passage %r split across %d image(s)", name_prefix, len(pages))
        return pages


__all__ = [
    "FONT_CANDIDATES",
    "DevanagariRenderer",
    "DevanagariUnavailableError",
    "RenderedText",
    "find_devanagari_font",
]
