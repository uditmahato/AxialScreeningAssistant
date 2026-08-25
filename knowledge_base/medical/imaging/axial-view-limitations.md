---
id: axial-view-limitations
title: Limitations of Single Axial Slice Analysis
category: imaging
subcategory: limitations
audience: [clinician, health_worker]
maps_to_class: [normal, abnormal]
severity: informational
last_reviewed: 2026-08-13
sources:
  - "Radiopaedia.org - peer-reviewed radiology reference (accessed 2026)"
  - "Selvaraju RR et al. (2017) Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV"
---

# Limitations of Single Axial Slice Analysis

This document states plainly what the Axial Screening Assistant cannot do. It exists because
the most dangerous failure mode of a medical AI system is being trusted beyond
its evidence.

## What a single axial slice omits

**Other planes.** Sagittal and coronal views are essential for the pituitary
region, the posterior fossa, the corpus callosum and midline structures. A
pituitary macroadenoma may be almost invisible on a standard axial brain slice.

**Other slices.** A lesion may be entirely absent from the slice provided. A
"normal" result on one image says nothing about the rest of the brain. This is
the most important limitation to communicate: **normal on this slice is not
normal brain.**

**Other sequences.** As set out in [[mri-sequences]], diagnosis requires
comparing T1, T2, FLAIR, DWI and post-contrast images. A single image cannot
provide the diffusion information that distinguishes abscess from tumour, nor
the enhancement pattern that discriminates much of the differential.

**Clinical context.** Age, symptoms, duration, fever, weight loss, HIV status,
TB contact, known cancer and travel history all change the differential
substantially - often more than the image does.

**Prior imaging.** Whether a lesion is new, stable or growing is frequently the
single most informative fact, and it is invisible to a single-timepoint system.

## What the system can legitimately claim

Within these limits, an automated classifier can:

- Flag that the provided image shows features that warrant expert review
- Provide a confidence estimate for that flag
- Show, via Grad-CAM, which region of the image drove the prediction
- Supply contextual information about the possibilities and the appropriate
  next steps

That is triage, and in a setting with roughly one neurosurgeon per 256,000
people it is a genuinely useful function. It is not diagnosis.

## Using the heatmap as a sanity check

The Grad-CAM overlay is not decoration. It is a practical check on whether the
prediction deserves attention:

- Heatmap centred on a plausible lesion: the prediction may be meaningful
- Heatmap on the skull, the image border, or scanner text: the model has
  latched onto an artefact and the prediction should be **distrusted**
- Heatmap diffuse across the whole brain: the model has not localised anything
  specific, and the output should be treated as low-information

Reporting this honestly is more useful than a confident number.


> This document supports clinical decision-making. It is not a diagnosis and
> does not replace assessment by a qualified clinician.
