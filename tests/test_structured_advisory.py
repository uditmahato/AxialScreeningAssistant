"""Tests for the structured (JSON) advisory pipeline.

The model returns JSON and this code decides how it looks. Parsing is
deliberately unforgiving: anything that does not match the expected shape
degrades to source text, because a half-parsed advisory is worse than none.
"""

from __future__ import annotations

import json

from neuroscan.rag.advisory import (
    clean_structured_advisory,
    enforce_prediction_consistency,
    parse_structured_advisory,
    structured_to_markdown,
)
from neuroscan.rag.prompts import build_structured_advisory_prompt

WELL_FORMED = {
    "summary": "The scan was flagged as abnormal and needs clinical review.",
    "possible_causes": [
        {"name": "Neurocysticercosis", "note": "A common, treatable infection in Nepal."},
        {"name": "Tuberculoma", "note": ""},
    ],
    "next_steps": ["Arrange review by a physician.", "Take the report to the visit."],
}


class TestParse:
    def test_parses_well_formed_json(self):
        parsed = parse_structured_advisory(json.dumps(WELL_FORMED))
        assert parsed is not None
        assert parsed["summary"].startswith("The scan")
        assert parsed["possible_causes"][0]["name"] == "Neurocysticercosis"
        assert len(parsed["next_steps"]) == 2

    def test_tolerates_a_fenced_block(self):
        raw = "```json\n" + json.dumps(WELL_FORMED) + "\n```"
        assert parse_structured_advisory(raw) is not None

    def test_tolerates_prose_around_the_object(self):
        raw = "Here is the advisory:\n" + json.dumps(WELL_FORMED) + "\nHope that helps."
        assert parse_structured_advisory(raw) is not None

    def test_accepts_bare_string_causes(self):
        payload = dict(WELL_FORMED, possible_causes=["Tuberculoma", "Glioma"])
        parsed = parse_structured_advisory(json.dumps(payload))
        assert parsed["possible_causes"][0] == {"name": "Tuberculoma", "note": ""}

    def test_rejects_missing_summary(self):
        payload = {k: v for k, v in WELL_FORMED.items() if k != "summary"}
        assert parse_structured_advisory(json.dumps(payload)) is None

    def test_rejects_empty_next_steps(self):
        payload = dict(WELL_FORMED, next_steps=[])
        assert parse_structured_advisory(json.dumps(payload)) is None

    def test_rejects_non_json(self):
        assert parse_structured_advisory("## What this result means\n\nProse.") is None
        assert parse_structured_advisory("") is None
        assert parse_structured_advisory("{broken json") is None

    def test_caps_causes_and_steps(self):
        payload = dict(
            WELL_FORMED,
            possible_causes=[{"name": f"Cause {i}", "note": ""} for i in range(10)],
            next_steps=[f"Step {i}" for i in range(10)],
        )
        parsed = parse_structured_advisory(json.dumps(payload))
        assert len(parsed["possible_causes"]) == 5
        assert len(parsed["next_steps"]) == 6


class TestClean:
    def test_strips_markdown_and_humanises_dashes(self):
        dirty = dict(
            WELL_FORMED,
            summary="This **needs** review — soon.",
            possible_causes=[{"name": "[[Tuberculoma]]", "note": "See *sources*."}],
        )
        cleaned = clean_structured_advisory(dirty)
        assert "**" not in cleaned["summary"]
        assert "—" not in cleaned["summary"]
        assert cleaned["possible_causes"][0]["name"] == "Tuberculoma"
        assert "*" not in cleaned["possible_causes"][0]["note"]


class TestPredictionConsistency:
    """A normal classification must never carry a disease list.

    Regression: the JSON prompt demanded 2-5 possible_causes for every
    prediction, so a scan classified normal came back with tumours and
    infections listed directly under "No abnormal pattern detected"."""

    def test_normal_result_loses_its_causes(self):
        out = enforce_prediction_consistency(WELL_FORMED, "normal")
        assert out["possible_causes"] == []
        assert out["summary"] == WELL_FORMED["summary"]
        assert out["next_steps"] == WELL_FORMED["next_steps"]

    def test_nepali_normal_label_is_recognised(self):
        assert enforce_prediction_consistency(WELL_FORMED, "सामान्य")["possible_causes"] == []

    def test_abnormal_result_keeps_its_causes(self):
        out = enforce_prediction_consistency(WELL_FORMED, "abnormal")
        assert len(out["possible_causes"]) == 2

    def test_parse_accepts_an_empty_causes_array(self):
        payload = dict(WELL_FORMED, possible_causes=[])
        parsed = parse_structured_advisory(json.dumps(payload))
        assert parsed is not None
        assert parsed["possible_causes"] == []

    def test_prompt_asks_for_no_causes_on_a_normal_result(self):
        _, user = build_structured_advisory_prompt(
            prediction="normal", confidence=0.95, context="ctx"
        )
        assert "exactly []" in user
        assert "2 to 5" not in user

    def test_prompt_still_asks_for_causes_on_an_abnormal_result(self):
        _, user = build_structured_advisory_prompt(
            prediction="abnormal", confidence=0.95, context="ctx"
        )
        assert "2 to 5" in user


class TestCanonicalMarkdown:
    """The text form is built BY this code, so its shape is deterministic."""

    def test_english_sections(self):
        text = structured_to_markdown(WELL_FORMED, "en")
        assert "## What this result means" in text
        assert "## Possible causes" in text
        assert "## Suggested next steps" in text
        assert "1. **Neurocysticercosis**: A common, treatable infection in Nepal." in text
        assert "- Arrange review by a physician." in text

    def test_nepali_sections(self):
        text = structured_to_markdown(WELL_FORMED, "ne")
        assert "## यो नतिजाको अर्थ" in text
        assert "## सम्भावित कारणहरू" in text

    def test_omits_causes_section_when_empty(self):
        payload = dict(WELL_FORMED, possible_causes=[])
        text = structured_to_markdown(payload, "en")
        assert "## Possible causes" not in text
        assert "## Suggested next steps" in text
