# Ethics and Data Governance

This document records what the code actually enforces, so that every ethics
commitment the project makes can be checked against a mechanism in the
repository.

---

## 1. The blocking rule on clinical data

> *"Although a preliminary set of anonymised MRI scans has been received for
> review purposes, none of the data will be used for training until full
> ethical approval has been granted."*, project commitment

**No training may be run against the Grande International Hospital dataset
until both of the following are signed:**

1. Institutional research ethics approval
2. Grande International Hospital Data Sharing Agreement

### How this is enforced

`configs/grande_clinical.yaml` is the only configuration that reads the
clinical data, and it carries the requirement at the top of the file. Before
using it, place a marker file at `data/raw/grande/ETHICS_APPROVED` containing
the approval date and reference. Its absence is the reminder.

Everything else in the project, every result reported so far, uses the
public Br35H dataset, which carries no such restriction.

---

## 2. Anonymisation

### What must be removed before data reaches this repository

Removed by qualified radiologists at Grande International Hospital **before**
transfer:

- Patient name, address, telephone number
- Date of birth, hospital and national ID numbers
- Referring clinician, and any free-text identifying a person
- DICOM header identifiers
- Any identifier burned into the image pixels

### What the code checks

`neuroscan.data.adapters.DicomAdapter` reads `PatientName` from every DICOM
header during discovery. If any file still carries a value, it logs at ERROR
level with a count and examples:

```
ANONYMISATION FAILURE: 3 DICOM file(s) still carry a PatientName ...
```

**This is a detector, not a guarantee.** It cannot see text burned into pixel
data. Visual inspection of a sample remains necessary, and the pre-upload
checklist in the web interface asks users to crop out visible identifiers.

---

## 3. Storage and retention

| Data | Location | Protection | Retention |
|---|---|---|---|
| Clinical MRI (Grande) | `data/raw/grande/` | Encrypted disk, password-protected, git-ignored | Destroyed after examination |
| Public MRI (Br35H) | `data/raw/br35h/` | Git-ignored (size, not sensitivity) | Indefinite |
| Web uploads | `instance/uploads/<random-id>/` | Auto-purged | **24 hours** |
| Generated reports | `instance/reports/` | Local only | Until manually cleared |
| Usability responses | `evaluation/usability/` | Pseudonymised (`P01`…) | Deleted after marking |

### Enforcement

`.gitignore` excludes `data/` in its entirety, plus `instance/`, `*.pt` and
`artifacts/`. Clinical images cannot reach git history through ordinary use,
including in a private repository.

`InferenceService.purge_old_uploads()` deletes upload directories older than
`web.retain_uploads_hours` (default 24). It runs opportunistically on each
upload rather than from a scheduler, so an offline clinic machine needs no cron
job for the retention policy to hold.

---

## 4. What the running system stores

**In the session cookie:** an opaque analysis id and the chat transcript.
Nothing patient-identifying.

**On disk, per analysis:** the uploaded image, the preprocessed image and the
Grad-CAM overlay, under a random 12-character id. Purged after 24 hours.

**Never stored:** patient names, ages, identifiers, IP-linked records, or any
linkage between an upload and a person.

Analysis results live in process memory and do not survive a restart. This is
deliberate: persisting them would require a database and a retention policy
that contradicts the commitment above.

---

## 5. Clinical safety

### Mandatory disclaimers

Every user-facing surface routes its disclaimer through `neuroscan.safety`.
Centralising the text means it is reviewed once and cannot drift between the
screen and the printout.

| Surface | Placement |
|---|---|
| Web, every page | Persistent band below the header, **not dismissible** |
| Web result page | Repeated under the verdict |
| PDF report | Directly under the header on page 1, plus a footer on every page |
| Chatbot | Appended to every reply, idempotently |
| Advisory | Appended to the generated text |

A dismissible banner is a banner that is absent when it matters, which is why
the web disclaimer cannot be closed.

### Prohibited outputs

`neuroscan.safety.REFUSAL_TRIGGERS` screens questions **before** retrieval, so a
prohibited request never reaches the model: medication doses, prognosis and
survival, and requests to manage a brain abnormality without medical
supervision.

`neuroscan.rag.advisory.PROHIBITED_PATTERNS` screens generated text **after**
the model, covering definitive diagnosis, dosages and prognosis. A violation
causes the whole response to be discarded and replaced with retrieved source
text, the model has demonstrably ignored its instructions, so the rest of that
output is not trustworthy either.

### Emergency escalation is unconditional

Red-flag symptoms are shown on every result **regardless of the prediction**.
A "normal" result on a deteriorating patient is the most dangerous output this
system can produce, so the escalation advice is never gated behind the model's
confidence.

---

## 6. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| User treats an abnormal flag as a cancer diagnosis | Non-dismissible disclaimer; advisory presents infection first for the Nepali context; task 2 of the usability study tests specifically for this misunderstanding |
| Model confidently wrong on out-of-distribution input | Brain-plausibility check rejects non-MRI uploads; Grad-CAM lets the user see when attention is on an artefact; diffuse attention is flagged in the UI |
| A missed abnormality | Threshold tuned against a recall floor rather than left at 0.5; recall reported as the headline metric, not accuracy |
| Model degrades on an unfamiliar scanner | Stated as a limitation in the corpus and the About page; no claim of external validity beyond the training distribution |
| Stale Nepali cost or scheme figures | Every figure carries a review date shown to the user; `knowledge_base/nepal/VERIFICATION.md` sets a re-check schedule |
| Patient identifier visible in an uploaded image | Pre-upload checklist asks users to crop; uploads purged after 24 hours |

---

## 7. Usability study governance

- Information sheet and consent form in English and Nepali, both required
  before any session
- Participation voluntary; withdrawal at any time without reason; deletion on
  request up to two weeks afterwards
- Participants identified only as `P01`, `P02`, …
- **No real patient scans** used in sessions, demonstration images only
- Only role, experience band, interface language, device, responses and
  comments recorded

See `evaluation/usability/`.

---

## 8. Regulatory position

Axial Screening Assistant is **a research prototype**. It is:

- **Not** a certified medical device
- **Not** approved by any regulator
- **Not** validated for clinical use
- **Not** to be used as the sole basis for any clinical decision

Medical device certification is explicitly outside the project scope (interim
report, section 3.3). The About page and every PDF report state this.

---

## 9. Pre-flight checklist for clinical data

- [ ] Institutional research ethics approval obtained and filed
- [ ] Grande International Hospital Data Sharing Agreement signed by both parties
- [ ] Access Request Letter approved
- [ ] Anonymisation confirmed by the hospital in writing
- [ ] Sample visually inspected for burned-in identifiers
- [ ] `data/raw/grande/ETHICS_APPROVED` marker created with date and reference
- [ ] Encrypted storage confirmed
- [ ] Supervisor informed that clinical training is beginning
- [ ] Destruction date recorded in the project log
