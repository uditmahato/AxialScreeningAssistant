---
id: after-abnormal-result
title: Next Steps After an Abnormal Result
category: pathway
subcategory: next-steps
audience: [clinician, health_worker, patient]
maps_to_class: [abnormal]
severity: high
last_reviewed: 2026-08-13
sources:
  - "NICE NG99: Brain tumours (primary) and brain metastases in over 16s (2021)"
  - "NICE NG12: Suspected cancer - recognition and referral (2023 update)"
---

# Next Steps After an Abnormal Result

An abnormal flag from an automated system is a prompt for review, not a
diagnosis. This document sets out what should follow.

## Step 1 - Assess the patient, not just the image

Before acting on the scan, establish clinically:

- Is the patient stable? Any reduced consciousness, new focal deficit, or
  seizure requires **emergency transfer regardless of the scan result**.
- What are the symptoms, and how long have they been present?
- Fever, weight loss, night sweats? Points toward infection - see
  [[tuberculoma]].
- Known cancer? Consider metastases.
- HIV status, TB contact, immunosuppression?
- Head injury?

## Step 2 - Check the heatmap

Look at the Grad-CAM overlay. If it highlights the skull, the image border, or
scanner annotation rather than brain tissue, the prediction is likely driven by
an artefact and should carry little weight. If attention is diffuse across the
whole image, the model has not localised anything specific.

This 10-second check is the most useful safeguard available to a non-specialist
user.

## Step 3 - Obtain radiological review

**Every abnormal result needs review by a qualified radiologist or physician
before any onward referral or treatment decision.** This is non-negotiable and
is the single most important step in the pathway.

Where a radiologist is not locally available, teleradiology or transferring
images to a district or referral hospital may be feasible.

## Step 4 - Cheap investigations that reshape the differential

Before an expensive referral, several inexpensive tests substantially change
the picture:

- **Chest X-ray** - TB, lung primary
- **HIV test**
- Full blood count, ESR/CRP, blood glucose
- Fundoscopy
- Blood pressure

In the Nepali context these may resolve the question, or redirect the patient
from a costly neurosurgical pathway to treatable infection management.

## Step 5 - Referral

- **Emergency** (reduced consciousness, rapid deterioration, suspected acute
  stroke or haemorrhage, suspected abscess): immediate transfer to the nearest
  hospital with CT and, ideally, neurosurgery
- **Urgent** (suspected tumour, progressive deficit): neurology or neurosurgery
  referral, with imaging
- **Routine** (likely incidental finding, stable, asymptomatic): outpatient
  review with interval imaging as advised

See [[nepal-neurology-hospitals]] and [[nepal-referral-pathway]].

## Step 6 - Address cost

Cost is a clinical variable in this setting, not an administrative one. A
patient who cannot afford the referral will not attend it. Discuss:

- Government support schemes - see [[nepal-government-schemes]]
- The nearest facility that can provide the needed service
- Whether a less expensive investigation would answer the question


> This document supports clinical decision-making. It is not a diagnosis and
> does not replace assessment by a qualified clinician.
