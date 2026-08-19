#!/usr/bin/env python
"""Score and analyse the usability study.

Usage:
    python scripts/analyse_usability.py --responses evaluation/usability/responses.csv
    python scripts/analyse_usability.py --responses ... --target 80
    python scripts/analyse_usability.py --demo        # synthetic data, to check the pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.evaluation.usability import (
    ParticipantResponse,
    UsabilityAnalysis,
    UsabilityError,
    adjective_rating,
    letter_grade,
    load_responses,
    percentile,
)
from neuroscan.utils import get_logger, setup_logging, write_json

log = get_logger("scripts.usability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse SUS usability responses")
    parser.add_argument("--responses", type=Path, default=None)
    parser.add_argument("--target", type=float, default=68.0,
                        help="Target mean SUS. 68 is the established average; "
                             "pass 80 for the stricter reading of the usability requirement.")
    parser.add_argument("--min-participants", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON results here")
    parser.add_argument("--plot", action="store_true", help="Write a score distribution chart")
    parser.add_argument("--demo", action="store_true",
                        help="Run on synthetic responses to verify the pipeline")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def demo_responses() -> list[ParticipantResponse]:
    """Synthetic responses, for verifying the analysis pipeline end to end.

    Clearly labelled as synthetic. These are NOT study results and must never
    appear in the final report.
    """
    import random

    rng = random.Random(42)
    roles = ["medical_student", "nurse", "chv", "radiographer", "public"]
    responses: list[ParticipantResponse] = []

    for i in range(1, 25):
        # Positive items skew high, negative items skew low, with noise.
        answers = []
        for item in range(1, 11):
            if item % 2 == 1:
                answers.append(min(5, max(1, round(rng.gauss(4.0, 0.8)))))
            else:
                answers.append(min(5, max(1, round(rng.gauss(2.1, 0.8)))))
        responses.append(ParticipantResponse(
            participant_id=f"P{i:02d}",
            responses=answers,
            role=rng.choice(roles),
            language=rng.choice(["en", "en", "ne"]),
            device=rng.choice(["laptop", "phone", "phone"]),
            comments="(synthetic demo data)",
            difficulties="(synthetic demo data)" if i % 4 == 0 else "",
        ))
    return responses


def plot_distribution(analysis: UsabilityAnalysis, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_hist, ax_role) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_hist.hist(analysis.scores, bins=10, range=(0, 100),
                 color="#2c7fb8", edgecolor="white")
    ax_hist.axvline(analysis.mean, color="#d62728", lw=2,
                    label=f"mean = {analysis.mean:.1f}")
    ax_hist.axvline(68, color="#888888", ls="--", lw=1.4, label="average system (68)")
    ax_hist.axvline(analysis.target_score, color="#1a9641", ls=":", lw=1.6,
                    label=f"target ({analysis.target_score:.0f})")
    ax_hist.set_xlabel("SUS score")
    ax_hist.set_ylabel("Participants")
    ax_hist.set_title(f"SUS distribution (n = {analysis.n})")
    ax_hist.legend(fontsize=8, frameon=False)

    by_role = analysis.by_group("role")
    names = list(by_role)
    means = [by_role[n]["mean"] for n in names]
    ax_role.barh(names, means, color="#4a90c4")
    ax_role.axvline(68, color="#888888", ls="--", lw=1.2)
    ax_role.set_xlim(0, 100)
    ax_role.set_xlabel("Mean SUS")
    ax_role.set_title("By participant role")
    for i, (name, value) in enumerate(zip(names, means, strict=False)):
        ax_role.text(value + 1.5, i, f"{value:.0f} (n={by_role[name]['n']})",
                     va="center", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if args.demo:
        print("\n*** SYNTHETIC DEMO DATA - NOT STUDY RESULTS ***")
        participants = demo_responses()
    elif args.responses:
        participants = load_responses(args.responses)
    else:
        print("Provide --responses <file.csv>, or --demo to verify the pipeline.",
              file=sys.stderr)
        return 1

    analysis = UsabilityAnalysis(
        participants, target_score=args.target, min_participants=args.min_participants
    )
    low, high = analysis.confidence_interval_95

    print("\n" + "=" * 78)
    print("USABILITY EVALUATION - SYSTEM USABILITY SCALE")
    print("=" * 78)
    print(f"  Participants        : {analysis.n}")
    print(f"  Mean SUS            : {analysis.mean:.1f}")
    print(f"  95% CI              : {low:.1f} - {high:.1f}")
    print(f"  Median              : {analysis.median:.1f}")
    print(f"  Std deviation       : {analysis.stdev:.1f}")
    print(f"  Range               : {min(analysis.scores):.1f} - {max(analysis.scores):.1f}")
    print()
    print(f"  Adjective rating    : {adjective_rating(analysis.mean)}")
    print(f"  Letter grade        : {letter_grade(analysis.mean)}")
    print(f"  Percentile          : ~{percentile(analysis.mean)}th")

    print("\n  " + "-" * 74)
    print("  USABILITY REQUIREMENT")
    print("  " + "-" * 74)
    ok_n = analysis.meets_participant_target
    ok_score = analysis.meets_score_target
    print(f"    >= {args.min_participants} participants   : "
          f"{'MET' if ok_n else 'NOT MET'} ({analysis.n})")
    print(f"    mean SUS >= {args.target:.0f}       : "
          f"{'MET' if ok_score else 'NOT MET'} ({analysis.mean:.1f})")
    print()
    print("    Note: a SUS score is NOT a percentage of satisfied users. A mean of")
    print(f"    {analysis.mean:.1f} places the system at roughly the {percentile(analysis.mean)}th percentile")
    print("    of systems tested, rated "
          f"'{adjective_rating(analysis.mean)}'. State it this way in the report.")

    print("\n  " + "-" * 74)
    print("  PER-ITEM MEANS  (this is where the actionable detail is)")
    print("  " + "-" * 74)
    from neuroscan.evaluation.usability import SUS_ITEMS_EN

    for item, mean_value in analysis.item_means().items():
        direction = "+" if item % 2 == 1 else "-"
        bar = "#" * round(mean_value * 5)
        print(f"    {item:2d} [{direction}] {mean_value:.2f}  {bar:<25} {SUS_ITEMS_EN[item - 1][:44]}")

    problems = analysis.problem_items()
    print("\n  " + "-" * 74)
    print("  IDENTIFIED WEAKNESSES")
    print("  " + "-" * 74)
    if problems:
        for problem in problems:
            print(f"    Item {problem['item']} (badness {problem['badness']:.2f}): "
                  f"{problem['text']}")
    else:
        print("    No individual item scored in the problem range.")

    flagged = analysis.flagged_responses()
    if flagged:
        print(f"\n    ! {len(flagged)} participant(s) gave an identical answer to every item, "
              f"which on an alternating-polarity scale suggests the items were not read: "
              f"{', '.join(flagged)}")

    print("\n  " + "-" * 74)
    print("  BREAKDOWN BY ROLE")
    print("  " + "-" * 74)
    for role, stats in analysis.by_group("role").items():
        print(f"    {role:22s} n={stats['n']:<3} mean={stats['mean']:5.1f}  "
              f"range {stats['min']:.0f}-{stats['max']:.0f}")

    print("\n  BREAKDOWN BY INTERFACE LANGUAGE")
    for language, stats in analysis.by_group("language").items():
        print(f"    {language:22s} n={stats['n']:<3} mean={stats['mean']:5.1f}")

    feedback = analysis.qualitative_feedback()
    if feedback:
        print("\n  " + "-" * 74)
        print(f"  QUALITATIVE FEEDBACK ({len(feedback)} participant(s))")
        print("  " + "-" * 74)
        for entry in feedback[:10]:
            print(f"    [{entry['participant_id']} · {entry['role']} · "
                  f"SUS {entry['sus_score']}]")
            if entry["difficulties"]:
                print(f"      difficulty: {entry['difficulties'][:100]}")
            if entry["comments"]:
                print(f"      comment   : {entry['comments'][:100]}")

    out_path = args.out or Path("artifacts/usability/usability_analysis.json")
    write_json(out_path, analysis.to_dict())
    print(f"\n  Results written to: {out_path}")

    if args.plot:
        chart = plot_distribution(analysis, out_path.parent / "sus_distribution.png")
        print(f"  Chart written to  : {chart}")

    print()
    return 0 if (ok_n and ok_score) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UsabilityError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc
