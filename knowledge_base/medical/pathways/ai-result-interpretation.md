---
id: ai-result-interpretation
title: How to Interpret the AI Result
category: pathway
subcategory: system-use
audience: [clinician, health_worker, patient]
maps_to_class: [normal, abnormal]
severity: informational
last_reviewed: 2026-08-13
sources:
  - "FDA Guidance: Clinical Decision Support Software (2022)"
  - "Selvaraju RR et al. (2017) Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV"
---

# How to Interpret the AI Result

This document explains what the system's outputs mean and, importantly, where
they should not be trusted.

## The classification

The system reports **normal** or **abnormal** for the uploaded axial image,
with a confidence value.

This is a **triage signal**. Its purpose is to help prioritise which scans get
expert review first, in a setting where review may otherwise take weeks. It is
not a diagnosis and carries no regulatory approval as a diagnostic device.

## The confidence value

The confidence is the model's estimated probability, not a probability that the
patient has disease. Two cautions:

1. **High confidence can be wrong.** Neural networks are frequently confident
   on inputs unlike their training data - a different scanner, a different
   sequence, a photograph of a film.
2. **The operating threshold is set deliberately low.** The system is tuned to
   favour sensitivity: it would rather flag a normal scan for review than miss
   an abnormal one. This means false positives are expected and are an accepted
   trade-off, not a malfunction.

## The Grad-CAM heatmap

The heatmap shows which image regions most influenced the prediction. Use it as
a sanity check:

| Heatmap appearance | Interpretation |
|---|---|
| Focused on a plausible brain lesion | Prediction may be meaningful |
| On skull, border, or scanner text | **Distrust** - artefact-driven |
| Diffuse over the whole brain | Low information; no localisation |
| On a normal structure (ventricles) | Treat with caution |

The system reports when attention is diffuse, and that flag should be believed.

## Known limitations

- Validated for **adult axial** brain MRI only
- Single slice; other slices and planes not assessed
- Cannot distinguish tumour from infection - a central limitation given that in
  Nepal infection is a leading cause of brain lesions
- No access to clinical context, prior imaging, or other sequences
- Performance on images from scanners unlike those in the training data is
  unknown and may be materially worse

## The rule that overrides everything else

**Clinical assessment outranks the system.** If the patient has red-flag
symptoms, act on them regardless of what the model reported. If the model flags
an abnormality but the patient is well and a radiologist reads the scan as
normal, the radiologist is correct.

An automated system that is treated as authoritative is more dangerous than no
system at all.


> This document supports clinical decision-making. It is not a diagnosis and
> does not replace assessment by a qualified clinician.
