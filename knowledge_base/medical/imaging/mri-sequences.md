---
id: mri-sequences
title: MRI Sequences and What They Show
category: imaging
subcategory: technique
audience: [clinician, health_worker]
maps_to_class: [normal, abnormal]
severity: informational
last_reviewed: 2026-08-13
sources:
  - "Radiopaedia.org - peer-reviewed radiology reference (accessed 2026)"
---

# MRI Sequences and What They Show

An MRI is not one image but a set of sequences, each sensitising the scanner to
different tissue properties. Interpreting a brain MRI means comparing across
them - which is precisely what a single-image automated classifier cannot do.

## The core sequences

**T1-weighted** - anatomical detail. Fat is bright, water is dark, and
cerebrospinal fluid appears dark. White matter appears brighter than grey
matter. Good for anatomy and for detecting fat, blood products and contrast
enhancement.

**T2-weighted** - sensitive to water, and therefore to most pathology. Water
and cerebrospinal fluid are bright. Most lesions contain increased water and so
appear bright. Grey matter appears brighter than white matter.

**FLAIR** (Fluid Attenuated Inversion Recovery) - a T2 image with the
cerebrospinal fluid signal suppressed. This makes lesions adjacent to the
ventricles and at the brain surface far more conspicuous, since they are no
longer competing with bright fluid. The workhorse sequence for white matter
disease.

**DWI** (Diffusion Weighted Imaging) - measures restriction of water movement.
**Critical for two things**: acute stroke, which restricts within minutes, and
distinguishing abscess (restricts in the cavity) from necrotic tumour (does
not). Interpreted alongside the ADC map to avoid "T2 shine-through" artefact.

**Gradient echo / SWI** - highly sensitive to blood products and calcification,
which appear as marked dark signal. Detects microbleeds invisible on other
sequences.

**Post-contrast T1** - after intravenous gadolinium. Enhancement indicates
breakdown of the blood-brain barrier, and its **pattern** (solid, ring,
incomplete ring, dural) carries much of the diagnostic information.

## Implication for this system

The Axial Screening Assistant analyses a single axial image, usually T1 or T2 weighted. This
is a deliberate scope decision (Project Scope: "axial view MRI brain scan
images in JPG or PNG format"), and it is a real limitation.

A radiologist reaches a diagnosis by comparing sequences, viewing multiple
planes and scrolling through the full stack. A single-slice classifier cannot
replicate that, and should not be presented as though it can. What it can
legitimately do is triage: flag scans that warrant prompt expert review, in a
setting where that review is otherwise delayed by weeks.


> This document supports clinical decision-making. It is not a diagnosis and
> does not replace assessment by a qualified clinician.
