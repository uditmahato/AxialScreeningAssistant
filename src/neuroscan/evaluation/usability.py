"""System Usability Scale scoring and analysis.

The usability requirement is "a usability test with a minimum of 20 users in which
there is a user satisfaction of at least 80% with documentation of system
weaknesses and future scope".

**A note on what "80% satisfaction" means.** A raw SUS score is not a
percentage, and reading it as one is the most common error in the literature.
A SUS of 80 sits at roughly the 88th percentile of systems tested, not at "80%
of users were satisfied". This module reports the raw score, its percentile,
its letter grade and the adjective rating, so the final report can state the
result precisely rather than implying a figure the instrument does not
support.

References:
    Brooke, J. (1996) SUS: A quick and dirty usability scale.
    Bangor, A., Kortum, P. and Miller, J. (2009) Determining what individual
        SUS scores mean: adding an adjective rating scale. JUS 4(3):114-123.
    Sauro, J. (2011) A Practical Guide to the System Usability Scale.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuroscan.utils import get_logger

log = get_logger("evaluation.usability")

N_ITEMS = 10
MIN_RESPONSE = 1
MAX_RESPONSE = 5

#: The standard SUS items. Odd items are positively worded, even items
#: negatively - the alternation is deliberate, to catch acquiescence bias.
SUS_ITEMS_EN: list[str] = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

SUS_ITEMS_NE: list[str] = [
    "मलाई लाग्छ कि म यो प्रणाली बारम्बार प्रयोग गर्न चाहन्छु।",
    "मलाई यो प्रणाली अनावश्यक रूपमा जटिल लाग्यो।",
    "मलाई यो प्रणाली प्रयोग गर्न सजिलो लाग्यो।",
    "मलाई लाग्छ यो प्रणाली प्रयोग गर्न प्राविधिक व्यक्तिको सहयोग चाहिन्छ।",
    "यस प्रणालीका विभिन्न सुविधाहरू राम्रोसँग मिलेका छन् जस्तो लाग्यो।",
    "यस प्रणालीमा धेरै असंगति छ जस्तो लाग्यो।",
    "मलाई लाग्छ अधिकांश मानिसले यो प्रणाली छिट्टै सिक्न सक्छन्।",
    "मलाई यो प्रणाली प्रयोग गर्न धेरै झन्झटिलो लाग्यो।",
    "यो प्रणाली प्रयोग गर्दा म धेरै आत्मविश्वासी महसुस गरें।",
    "यो प्रणाली सुरु गर्नुअघि मैले धेरै कुरा सिक्नुपर्‍यो।",
]

#: Bangor et al. (2009) adjective ratings, mapped to score bands.
ADJECTIVE_BANDS: list[tuple[float, str]] = [
    (84.1, "Best imaginable"),
    (80.8, "Excellent"),
    (71.4, "Good"),
    (51.7, "OK / Fair"),
    (25.6, "Poor"),
    (0.0, "Worst imaginable"),
]

#: Sauro (2011) curved grading, based on a database of 500+ studies.
GRADE_BANDS: list[tuple[float, str]] = [
    (80.3, "A"),
    (74.0, "B"),
    (68.0, "C"),
    (51.0, "D"),
    (0.0, "F"),
]

#: Approximate percentile lookup against Sauro's normative distribution.
PERCENTILE_TABLE: list[tuple[float, int]] = [
    (84.1, 96), (80.8, 90), (78.9, 85), (77.2, 80), (74.0, 70),
    (71.1, 60), (68.0, 50), (65.0, 40), (62.7, 35), (60.5, 30),
    (56.5, 20), (51.7, 15), (45.5, 10), (35.7, 5), (0.0, 1),
]


class UsabilityError(ValueError):
    """Raised when responses are malformed."""


@dataclass
class ParticipantResponse:
    """One participant's completed questionnaire."""

    participant_id: str
    responses: list[int]
    role: str = ""
    language: str = "en"
    years_experience: str = ""
    device: str = ""
    comments: str = ""
    difficulties: str = ""

    def __post_init__(self) -> None:
        if len(self.responses) != N_ITEMS:
            raise UsabilityError(
                f"Participant {self.participant_id}: expected {N_ITEMS} responses, "
                f"got {len(self.responses)}"
            )
        for index, value in enumerate(self.responses, start=1):
            if not MIN_RESPONSE <= value <= MAX_RESPONSE:
                raise UsabilityError(
                    f"Participant {self.participant_id}, item {index}: response {value} "
                    f"is outside the valid range {MIN_RESPONSE}-{MAX_RESPONSE}"
                )

    @property
    def sus_score(self) -> float:
        """The participant's SUS score, 0-100.

        Odd (positively worded) items contribute ``response - 1``; even
        (negatively worded) items contribute ``5 - response``. The sum, 0-40,
        is multiplied by 2.5.
        """
        total = 0
        for index, value in enumerate(self.responses):
            total += (value - 1) if index % 2 == 0 else (MAX_RESPONSE - value)
        return total * 2.5

    @property
    def is_straight_lined(self) -> bool:
        """Whether every item received the same response.

        On a scale that alternates positive and negative wording, an identical
        answer to all ten items is close to self-contradictory and usually
        means the participant did not read the items. Flagged rather than
        dropped - excluding data is the researcher's decision, not the
        script's.
        """
        return len(set(self.responses)) == 1


def adjective_rating(score: float) -> str:
    for threshold, label in ADJECTIVE_BANDS:
        if score >= threshold:
            return label
    return "Worst imaginable"


def letter_grade(score: float) -> str:
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def percentile(score: float) -> int:
    for threshold, value in PERCENTILE_TABLE:
        if score >= threshold:
            return value
    return 1


@dataclass
class UsabilityAnalysis:
    """Aggregate results across all participants."""

    participants: list[ParticipantResponse]
    target_score: float = 68.0
    min_participants: int = 20

    scores: list[float] = field(init=False)

    def __post_init__(self) -> None:
        if not self.participants:
            raise UsabilityError("No participant responses provided")
        self.scores = [p.sus_score for p in self.participants]

    @property
    def n(self) -> int:
        return len(self.participants)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores)

    @property
    def median(self) -> float:
        return statistics.median(self.scores)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.scores) if self.n > 1 else 0.0

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """95% confidence interval for the mean SUS score.

        Uses the t distribution, which matters at these sample sizes: with
        n=20 the t multiplier is about 2.09 against 1.96 for the normal, and
        using the wrong one understates the interval by roughly 7%.
        """
        if self.n < 2:
            return (self.mean, self.mean)

        # t critical values at 95%, indexed by degrees of freedom.
        t_table = {
            1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 14: 2.145,
            16: 2.120, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060,
            30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
        }
        df = self.n - 1
        t = next((v for k, v in sorted(t_table.items()) if df <= k), 1.96)

        margin = t * (self.stdev / (self.n ** 0.5))
        return (self.mean - margin, self.mean + margin)

    @property
    def meets_participant_target(self) -> bool:
        return self.n >= self.min_participants

    @property
    def meets_score_target(self) -> bool:
        return self.mean >= self.target_score

    def item_means(self) -> dict[int, float]:
        """Mean raw response per item.

        Reported per item because the aggregate score hides the diagnosis. A
        SUS of 72 with item 4 ("I would need technical support") averaging 4.1
        points at a specific, fixable problem; the aggregate alone does not.
        """
        return {
            index + 1: statistics.fmean([p.responses[index] for p in self.participants])
            for index in range(N_ITEMS)
        }

    def problem_items(self, threshold: float = 3.0) -> list[dict[str, Any]]:
        """Items whose responses indicate a usability problem.

        Normalised so that higher always means worse, regardless of the item's
        wording direction.
        """
        problems = []
        for index, mean_value in self.item_means().items():
            positively_worded = index % 2 == 1
            # Convert to a 1-5 "badness" scale.
            badness = (MAX_RESPONSE + 1 - mean_value) if positively_worded else mean_value
            if badness >= threshold:
                problems.append({
                    "item": index,
                    "text": SUS_ITEMS_EN[index - 1],
                    "mean_response": round(mean_value, 2),
                    "badness": round(badness, 2),
                })
        return sorted(problems, key=lambda p: p["badness"], reverse=True)

    def by_group(self, attribute: str) -> dict[str, dict[str, Any]]:
        """Break scores down by a participant attribute, e.g. ``role``."""
        groups: dict[str, list[float]] = {}
        for participant in self.participants:
            key = str(getattr(participant, attribute, "") or "unspecified")
            groups.setdefault(key, []).append(participant.sus_score)

        return {
            key: {
                "n": len(values),
                "mean": round(statistics.fmean(values), 1),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
            }
            for key, values in sorted(groups.items())
        }

    def flagged_responses(self) -> list[str]:
        """Participant ids whose responses warrant a manual look."""
        return [p.participant_id for p in self.participants if p.is_straight_lined]

    def qualitative_feedback(self) -> list[dict[str, str]]:
        """Free-text comments, which is where the actionable detail lives."""
        return [
            {
                "participant_id": p.participant_id,
                "role": p.role,
                "sus_score": round(p.sus_score, 1),
                "comments": p.comments,
                "difficulties": p.difficulties,
            }
            for p in self.participants
            if p.comments.strip() or p.difficulties.strip()
        ]

    def to_dict(self) -> dict[str, Any]:
        low, high = self.confidence_interval_95
        return {
            "n_participants": self.n,
            "mean_sus": round(self.mean, 2),
            "median_sus": round(self.median, 2),
            "stdev": round(self.stdev, 2),
            "min": round(min(self.scores), 1),
            "max": round(max(self.scores), 1),
            "ci_95_low": round(low, 2),
            "ci_95_high": round(high, 2),
            "adjective_rating": adjective_rating(self.mean),
            "letter_grade": letter_grade(self.mean),
            "percentile": percentile(self.mean),
            "target_score": self.target_score,
            "meets_score_target": self.meets_score_target,
            "min_participants": self.min_participants,
            "meets_participant_target": self.meets_participant_target,
            "item_means": {str(k): round(v, 2) for k, v in self.item_means().items()},
            "problem_items": self.problem_items(),
            "by_role": self.by_group("role"),
            "by_language": self.by_group("language"),
            "flagged_participants": self.flagged_responses(),
            "individual_scores": [
                {"participant_id": p.participant_id, "sus": round(p.sus_score, 1),
                 "role": p.role, "language": p.language}
                for p in self.participants
            ],
        }


def load_responses(csv_path: Path) -> list[ParticipantResponse]:
    """Load responses from the CSV template.

    Expected columns: ``participant_id``, ``q1``..``q10``, and optionally
    ``role``, ``language``, ``years_experience``, ``device``, ``comments``,
    ``difficulties``.

    Raises:
        UsabilityError: If the file is missing, empty, or malformed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise UsabilityError(f"Response file not found: {csv_path}")

    participants: list[ParticipantResponse] = []
    errors: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise UsabilityError(f"Response file is empty: {csv_path}")

        columns = {c.strip().casefold() for c in reader.fieldnames}
        missing = [c for c in ["participant_id", *[f"q{i}" for i in range(1, 11)]]
                   if c not in columns]
        if missing:
            raise UsabilityError(f"Response file is missing columns: {missing}")

        for line_no, row in enumerate(reader, start=2):
            clean = {(k or "").strip().casefold(): (v or "").strip() for k, v in row.items()}
            participant_id = clean.get("participant_id", "")
            if not participant_id:
                continue

            try:
                responses = [int(clean[f"q{i}"]) for i in range(1, 11)]
            except (KeyError, ValueError) as exc:
                errors.append(f"line {line_no} ({participant_id}): {exc}")
                continue

            try:
                participants.append(ParticipantResponse(
                    participant_id=participant_id,
                    responses=responses,
                    role=clean.get("role", ""),
                    language=clean.get("language", "en"),
                    years_experience=clean.get("years_experience", ""),
                    device=clean.get("device", ""),
                    comments=clean.get("comments", ""),
                    difficulties=clean.get("difficulties", ""),
                ))
            except UsabilityError as exc:
                errors.append(f"line {line_no}: {exc}")

    if errors:
        log.warning("Skipped %d malformed row(s):\n  %s", len(errors), "\n  ".join(errors))
    if not participants:
        raise UsabilityError(f"No valid responses could be read from {csv_path}")

    log.info("Loaded %d participant response(s) from %s", len(participants), csv_path)
    return participants


def get_items(language: str = "en") -> list[str]:
    return SUS_ITEMS_NE if language == "ne" else SUS_ITEMS_EN


__all__ = [
    "ADJECTIVE_BANDS",
    "GRADE_BANDS",
    "N_ITEMS",
    "SUS_ITEMS_EN",
    "SUS_ITEMS_NE",
    "ParticipantResponse",
    "UsabilityAnalysis",
    "UsabilityError",
    "adjective_rating",
    "get_items",
    "letter_grade",
    "load_responses",
    "percentile",
]
