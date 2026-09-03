# Security and safety policy

This project has two kinds of report worth making. Software vulnerabilities
are the usual kind. Clinical safety defects are the kind that matter more
here, and they are covered below as well.

## Supported versions

| Version | Supported |
|---|---|
| `main` | Yes |
| v1.0 | Yes, for safety defects |
| Anything earlier | No |

This is a research prototype with a single maintainer. There is no security
support contract and no guaranteed response time. Please read that as
honesty rather than indifference.

## Reporting a vulnerability

Please report privately, not in a public issue.

Use **Report a vulnerability** under the repository's Security tab, which
opens a private advisory visible only to the maintainer. If that is not
available to you, contact the maintainer through their GitHub profile and ask
for a private channel before sending details.

Please include what you were doing, what happened, what you expected, and the
smallest reproduction you have. If you have a suggested fix, say so; if you
would like to be credited in the advisory, say that too.

What to expect: an acknowledgement when the report is read, an assessment of
whether it reproduces, and a fix or an explanation of why it will not be
fixed. Please give a reasonable period before disclosing publicly, and get in
touch if you have heard nothing.

## Reporting a clinical safety defect

Report these the same private way, and treat them as at least as serious as a
vulnerability. Examples, all of which have precedent in this codebase:

- A safety screen failing open, so an unsafe question reaches retrieval or
  unsafe generated text reaches the user. This has happened: an earlier
  English-only trigger set missed 16 of 24 realistic probes, and a later
  versioned probe corpus found six further gaps, including phrasings in
  Devanagari and romanised Nepali.
- Generated text containing a dosage, a prognosis, a survival figure, or a
  definitive diagnosis.
- A normal classification whose advisory lists disease possibilities.
- Red flag guidance or the disclaimer missing from any output surface,
  including the PDF and the print stylesheet.
- Advice that is wrong for the setting, for example recommending a pathway or
  cost that does not exist in Nepal.
- Anything that would lead a non-specialist to delay care.

A report saying only "the advisory said something that worries me" is welcome.
You do not need to identify the mechanism.

## Scope

In scope: the application in `src/`, the scripts in `scripts/`, the knowledge
base in `knowledge_base/`, the safety screens and their probe corpus, and the
released model checkpoint.

Out of scope: vulnerabilities in third-party dependencies without a
demonstrated impact here (report those upstream), issues that require an
already-compromised host, and the deliberate limitations documented in the
README, such as single-slice input and the absence of subtype classification.

## Known limitations, deliberately not treated as defects

These are documented and accepted rather than hidden. Reporting them is fine,
but they are already known:

- Zero-shot external specificity is about 22%, so the system over-refers on
  unfamiliar scanners.
- Confidence is poorly calibrated on unfamiliar data, with an expected
  calibration error above 0.12 externally.
- Grad-CAM shows where the model looked, not where a lesion is, and the
  diffuse-attention warning is a heuristic with no validated relationship to
  lesion location.
- Generated text is not checked for entailment against the retrieved
  passages. The validator screens for prohibited patterns only.
- The knowledge base is English only, so the degraded advisory path serves
  English source text with a Nepali explanatory notice.

## Handling of user data

Uploaded images are stored under a random identifier and purged after 24
hours, and generated PDF reports are purged on the same schedule. Results are
held in process memory only. No clinical or hospital data is included in this
repository or in any release.
