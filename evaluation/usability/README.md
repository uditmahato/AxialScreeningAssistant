# Usability Evaluation Protocol

The usability requirement: *"conducting a usability test with a minimum of 20 users in which
there is a user satisfaction of at least 80% with documentation of system
weaknesses and future scope."*

## Contents

| File | Purpose |
|---|---|
| `participant_information_sheet.md` | Given to every participant before consent (EN + NE) |
| `consent_form.md` | Signed before participation (EN + NE) |
| `task_scenarios.md` | The four tasks each participant attempts |
| `sus_questionnaire.md` | The System Usability Scale, both languages |
| `responses_template.csv` | Data collection template |
| `../../scripts/analyse_usability.py` | Scoring and analysis |

## A necessary clarification about the "80%" target

**A SUS score is not a percentage.** This matters because the objective as
written invites a misreading that would misstate the project's result.

The System Usability Scale produces a number from 0 to 100, but it is *not* the
percentage of satisfied users and it is not a percentage of anything. A raw
score of 68 is the established average across hundreds of studies; 80.3 sits at
roughly the 90th percentile.

The analysis script therefore reports four things, and the final report should
quote them together:

- **Raw SUS score** with a 95% confidence interval
- **Percentile** against Sauro's normative database
- **Letter grade** (A-F, curved)
- **Adjective rating** (Bangor et al.: "OK", "Good", "Excellent")

Read as "a mean SUS of at least 80", the objective is demanding but achievable
and corresponds to roughly the 90th percentile. Read as "80% of users satisfied"
it is a different and weaker claim. **State which one is meant.** The default
target in the script is 68 (the established average); pass `--target 80` to
evaluate against the stricter reading.

## Recruitment

Target: **at least 24 participants**: to leave headroom for exclusions.

Aim for a spread across the intended user population rather than convenience
sampling from one group, since the system is meant for non-specialists:

| Group | Target | Why |
|---|---|---|
| Medical students / junior doctors | 8 | Primary intended user |
| Nurses / health assistants | 6 | Frequent first point of contact |
| Community health volunteers (FCHV) | 4 | Lowest assumed technical background |
| Radiographers / technicians | 3 | Judge the imaging aspects |
| General public / patients | 3 | Judge comprehensibility |

Record whether each participant used the English or Nepali interface, and on
what device. A system that scores well on a laptop and badly on a phone has a
problem that the aggregate score would hide.

## Session structure (about 30 minutes)

1. **Introduction (3 min)**: purpose, that the *system* is being tested and
   not the participant, and that they may stop at any time
2. **Information sheet and consent (5 min)**: no session begins without a
   signed form
3. **Tasks (15 min)**: the four scenarios in `task_scenarios.md`, thinking
   aloud. Record completion, time, and any errors or hesitations.
4. **SUS questionnaire (5 min)**: completed by the participant, unaided
5. **Debrief (5 min)**: open questions on difficulties and suggestions

## Facilitator rules

- **Do not help** unless the participant is genuinely stuck. A prompt given too
  early conceals the exact problem the test exists to find.
- **Record hesitations**: not just failures. A task completed after 40 seconds
  of confusion is a finding.
- **Note the participant's own words** for the debrief. Free-text feedback is
  where actionable detail lives; the aggregate score tells you *that* there is
  a problem, never *what* it is.
- **Do not use real patient scans.** Use the anonymised demonstration images
  in `sample_scans/`.

## Ethics requirements

Before any session:

- [ ] Institutional research ethics approval obtained
- [ ] Information sheet and consent form approved and translated
- [ ] Demonstration scans confirmed anonymised
- [ ] Data storage location confirmed (encrypted, access-restricted)
- [ ] Participants know they may withdraw without giving a reason

No identifying information is recorded. Participants are referred to as `P01`,
`P02`, and so on, and the linking list, if one is kept at all, is stored
separately from the responses.

## Analysis

```bash
python scripts/analyse_usability.py --responses evaluation/usability/responses.csv
```

Produces the mean SUS with a confidence interval, per-item means, a breakdown
by role and by interface language, identification of problem items, and a
weaknesses report.

**Per-item means are the important output.** An aggregate of 72 with item 4
("I would need technical support") averaging 4.1 identifies a specific, fixable
problem. The aggregate alone does not.
