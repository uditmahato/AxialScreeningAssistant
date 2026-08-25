---
id: normal-variants
title: Normal Variants Mistaken for Abnormalities
category: non_tumour
subcategory: background
audience: [clinician, health_worker, patient]
maps_to_class: [normal]
severity: informational
last_reviewed: 2026-08-13
sources:
  - "Radiopaedia.org - peer-reviewed radiology reference (accessed 2026)"
---

# Normal Variants Mistaken for Abnormalities

Several normal anatomical variants are routinely mistaken for pathology, both
by inexperienced readers and by automated systems. Recognising them prevents
unnecessary investigation, cost and anxiety.

## Common variants

- **Cavum septum pellucidum** - a fluid space between the leaves of the septum
  pellucidum. Present in a proportion of adults and entirely normal.
- **Cavum vergae** - a posterior extension of the above.
- **Virchow-Robin (perivascular) spaces** - small fluid spaces around
  penetrating vessels, following cerebrospinal fluid signal on all sequences.
  Common in the basal ganglia and increase with age. Mistaken for lacunar
  infarcts or small cysts.
- **Pineal cyst** - very common; small ones are of no significance.
- **Mega cisterna magna** - an enlarged fluid space behind the cerebellum,
  without mass effect.
- **Asymmetric lateral ventricles** - mild asymmetry is common and normal.
- **Developmental venous anomaly** - a "caput medusae" of small veins draining
  into a larger one. A normal variant of venous drainage, not a malformation,
  and should not be treated.
- **Prominent perivascular spaces in the basal ganglia** - increasingly common
  with age and hypertension.
- **Age-related white matter hyperintensities** - small punctate T2 bright
  spots are common in older adults and usually reflect small vessel change
  rather than demyelination or metastases.

## Why this matters for an automated system

A binary normal/abnormal classifier trained on datasets where "normal" means
"no tumour" may flag any of these variants as abnormal. That is a **false
positive with real cost**: in a setting where a patient has travelled 8-12
hours and paid a substantial share of monthly income for the scan, an
unnecessary referral is a genuine harm and not merely an inefficiency.

The practical mitigations are:

1. Present the model's output as a **triage signal**, not a finding
2. Always route through radiological review before onward referral
3. Use the Grad-CAM heatmap to check whether the model attended to a plausible
   region or to a normal structure - a heatmap centred on the ventricles or on
   the image border is a strong indication the prediction should be distrusted

## Distinguishing features

The most useful general rule: a normal fluid-containing variant **follows
cerebrospinal fluid signal on every sequence**, shows **no enhancement**, and
causes **no mass effect** on surrounding structures. A lesion that differs from
cerebrospinal fluid on any sequence, enhances, or displaces adjacent structures
needs proper assessment.


> This document supports clinical decision-making. It is not a diagnosis and
> does not replace assessment by a qualified clinician.
