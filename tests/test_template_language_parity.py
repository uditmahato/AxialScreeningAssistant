"""Bilingual parity enforcement for the Jinja templates.

The templates carry inline ``{% if language == 'ne' %}`` blocks, and an audit
found they had drifted: English users got an actionable command where Nepali
users got nothing, and whole sections had no Nepali branch. The string table
has a parity test; this is the equivalent for the conditionals that remain in
template form - every language conditional must offer both branches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATE_DIR = (
    Path(__file__).parent.parent / "src" / "neuroscan" / "web" / "templates"
)
TEMPLATES = sorted(TEMPLATE_DIR.glob("*.html"))

_TAG = re.compile(r"\{%-?\s*(if|elif|else|endif)\b")
_LANGUAGE_IF = re.compile(r"\{%-?\s*if\s+language\s*==")


def _language_blocks_missing_else(source: str) -> list[int]:
    """Line numbers of ``if language`` blocks with no ``else`` branch."""
    missing: list[int] = []
    for match in _LANGUAGE_IF.finditer(source):
        depth = 1
        has_else = False
        for tag in _TAG.finditer(source, match.end()):
            name = tag.group(1)
            if name == "if":
                depth += 1
            elif name == "endif":
                depth -= 1
                if depth == 0:
                    break
            elif name in ("else", "elif") and depth == 1:
                has_else = True
        if not has_else:
            missing.append(source.count("\n", 0, match.start()) + 1)
    return missing


def test_templates_exist():
    assert TEMPLATES, f"no templates found under {TEMPLATE_DIR}"


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_every_language_conditional_has_both_branches(template: Path):
    source = template.read_text(encoding="utf-8")
    missing = _language_blocks_missing_else(source)
    assert not missing, (
        f"{template.name}: 'if language' block(s) at line(s) {missing} have no "
        f"else branch - one language would see nothing there"
    )
