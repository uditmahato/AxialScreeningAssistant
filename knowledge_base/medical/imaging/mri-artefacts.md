---
id: mri-artefacts
title: MRI Artefacts and Their Effect on Automated Analysis
category: imaging
subcategory: limitations
audience: [clinician, health_worker]
maps_to_class: [normal, abnormal]
severity: informational
last_reviewed: 2026-08-13
sources:
  - "Radiopaedia.org - peer-reviewed radiology reference (accessed 2026)"
---

# MRI Artefacts and Their Effect on Automated Analysis

Artefacts are image features that do not correspond to real anatomy. They
mislead human readers occasionally and automated systems considerably more
often, because a model has no concept of physical plausibility.

## Common artefacts

**Motion** - blurring and ghosting from patient movement. Common in children,
in patients in pain, and in the distressed. Can mimic or obscure lesions.

**Susceptibility** - signal loss and distortion near metal (dental work,
surgical clips, implants) or at air-tissue boundaries. Prominent at the skull
base and near the sinuses.

**Chemical shift** - a bright and dark rim at fat-water interfaces.

**Aliasing (wrap-around)** - anatomy outside the field of view folding onto the
opposite side of the image.

**Truncation (Gibbs)** - parallel bright and dark lines near sharp boundaries;
can mimic a syrinx in the spinal cord.

**Partial volume averaging** - a structure only partly within the slice appears
faint and blurred, which can look like a subtle lesion or hide a real one.

## Artefacts introduced outside the scanner

These matter more here than the physics artefacts, because uploaded images in
this setting are frequently photographs:

- **Photographing a film or screen** - glare, reflections, moire patterns, skew
- **JPEG compression** - blocking artefacts and loss of fine texture
- **Screenshot scaling** - interpolation blur
- **Burned-in annotation** - patient name, scanner text, orientation markers.
  A model can learn to associate such text with a class if it correlated with
  the label in training data. This is a well-documented failure mode in medical
  imaging AI.

## Why this is a serious issue for automated systems

A classifier has no way to know that a bright band is a motion artefact rather
than a lesion. Worse, models have repeatedly been shown to exploit incidental
markers - a hospital's annotation style, a scanner's characteristic noise - as
shortcuts that correlate with the label in training but carry no clinical
meaning and fail entirely on new data.

## The practical safeguard

**Check the Grad-CAM heatmap.** If the model's attention sits on scanner text,
the image border, the skull, or an obvious artefact rather than on brain tissue,
the prediction is being driven by something that is not anatomy, and it should
be disregarded.

This is the single most valuable use of the explainability output, and it is
why the heatmap is shown for every prediction rather than being an optional
extra. See [[axial-view-limitations]].


> This document supports clinical decision-making. It is not a diagnosis and
> does not replace assessment by a qualified clinician.
