---
id: nepal-verification
title: Verification Status of Nepali Data
category: nepal
subcategory: governance
audience: [clinician]
maps_to_class: [normal, abnormal]
severity: informational
last_reviewed: 2026-08-13
sources:
  - "Nepal Ministry of Health and Population - programme documentation"
  - "Health Insurance Board (Swasthya Bima Board), Nepal - programme documentation"
---

# Verification Status of Nepali Data

Honesty about data currency is a safety requirement, not a formality. A patient
who travels on the basis of a stale figure bears a real cost.

## Status of each file

| File | Content | Volatility | Re-check |
|---|---|---|---|
| `hospitals.json` | Facility names, locations, services | Low-moderate | Annually |
| `nepal-neurology-hospitals.md` | As above, plus guidance | Low-moderate | Annually |
| `nepal-government-schemes.md` | Scheme names, entitlements, amounts | **High** | **Every 6 months** |
| `nepal-imaging-costs.md` | Indicative prices | **High** | **Every 6 months** |
| `nepal-referral-pathway.md` | Structural pathway | Low | Annually |
| `nepal-tb-programme.md` | Free-service entitlement | Low-moderate | Annually |
| `nepal-transport-and-access.md` | Geography, practical advice | Low | Annually |

## What is deliberately NOT included

**Telephone numbers, email addresses and named consultants.** These change
frequently, and a stale number that a patient calls in an emergency is worse
than no number - it wastes time at exactly the wrong moment. The system directs
users to confirm contact details rather than presenting numbers that may be
wrong.

The exception is national emergency numbers (102 ambulance, 100 police), which
are stable.

## How figures are presented to users

Every cost and entitlement is presented with:

1. The word **indicative**, not a definite price
2. A visible **last-reviewed date**
3. An instruction to confirm directly with the facility or scheme

The application surfaces the review date alongside the figures rather than
hiding it in metadata.

## Verification procedure

For each re-check:

1. Consult current Ministry of Health and Population and Health Insurance Board
   publications
2. Telephone-sample a small number of listed facilities to sanity-check
   indicative costs
3. Confirm scheme entitlement ceilings and the covered-condition list
4. Update `last_reviewed` in the frontmatter of each amended file
5. **Rebuild the FAISS index** - `python scripts/build_index.py --rebuild` -
   otherwise the retrieval layer continues serving the previous text
6. Record the review in the project log

## Known limitations of this dataset

- Compiled from published programme documentation and public information, not
  from a primary survey of facilities
- Service availability at a given hospital can change with staff movement; a
  listed service is not a guarantee it is available on a given day
- Costs vary between facilities and over time, and private-sector pricing is
  not centrally published
- The facility list prioritises neurological capability and is not a complete
  register of Nepali hospitals
