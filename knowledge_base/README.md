# Axial Screening Assistant Knowledge Base

The retrieval corpus behind the RAG advisory module (a carefully selected set
of medical data comprising a minimum of 50 medical documents) and the Nepali
healthcare database (at least 10 neurology hospitals, government healthcare
programmes and associated costs).

## Layout

```
knowledge_base/
  medical/
    tumours/      Primary and secondary brain neoplasms
    non_tumour/   Vascular, infective, inflammatory and structural abnormalities
    imaging/      MRI sequences, appearances, artefacts and their limits
    symptoms/     Presenting features, red flags, escalation
    pathways/     What to do after a result; referral and follow-up
  nepal/          Hospitals, government schemes, costs, transport, referral
  SOURCES.md      Full provenance for every clinical claim
```

## Document format

Every document carries YAML frontmatter. The retrieval layer reads it to filter
and to attribute answers, so it is required, not decorative.

```yaml
---
id: glioma-overview           # unique, kebab-case, stable across edits
title: Glioma - Overview
category: tumour              # tumour | non_tumour | imaging | symptom | pathway | nepal
subcategory: primary-brain-tumour
audience: [clinician, health_worker, patient]
maps_to_class: [abnormal]     # links a document to a classifier output
severity: high                # emergency | high | moderate | low | informational
last_reviewed: 2026-08-13
sources:
  - "WHO Classification of Tumours of the CNS, 5th ed. (2021)"
---
```

## Editorial rules

These exist because the corpus feeds a system that speaks to non-specialists,
and a retrieval corpus is only as safe as its least careful document.

1. **No dosages, no regimens.** Name a management *category* if the source
   supports it; never a drug and a number. The system refuses dosage questions
   outright (`neuroscan.safety.REFUSAL_TRIGGERS`), and the corpus must not
   contain material that would tempt the model to answer anyway.
2. **No prognosis or survival figures.** Survival statistics are population
   estimates that a frightened reader will apply to themselves. They are out of
   scope for a triage tool.
3. **Hedged language throughout.** "Features that may be consistent with" -
   never "this is". The model mirrors the register of what it retrieves, so
   confident source text produces confident, unsafe answers.
4. **Every clinical claim is attributable.** If it is not in `sources`, it does
   not go in the document.
5. **Emergency content comes first.** Where a topic has a red flag, it is stated
   in the opening lines, not buried in a later section. Retrieval may return
   only the first chunk of a document.
6. **Nepal context is not an afterthought.** Where epidemiology in Nepal differs
   materially from the Western literature - and for brain lesions it does,
   substantially - that difference is stated explicitly.

## A note on epidemiology

A corpus assembled from Western sources alone would badly misdirect a Nepali
health worker. In Nepal and the wider South Asian region, **neurocysticercosis
and tuberculoma are among the most common causes of a focal brain lesion**,
particularly in a patient presenting with new-onset seizures. Both are
treatable, and both are frequently mistaken for tumour on imaging.

A system that ranks "glioma" above "neurocysticercosis" for a ring-enhancing
lesion in a young Nepali patient is not merely unhelpful - it points toward
neurosurgical referral and a cost the family may not survive financially, when
the actual condition is often managed medically. The corpus therefore carries
dedicated documents on both, and the advisory prompts require differential
possibilities to be presented in an order appropriate to the local population.

## Verification status of Nepali figures

Costs, scheme entitlements and contact details in `nepal/` change frequently.
Each file records `last_reviewed` and marks figures as indicative. The
application surfaces that date to the user rather than presenting the numbers
as current fact. See `nepal/VERIFICATION.md` for what needs re-checking and how
often.

## Rebuilding the index

```bash
python scripts/build_index.py --rebuild
```
