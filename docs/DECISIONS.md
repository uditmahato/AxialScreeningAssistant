# Design Decisions

Each entry records a real fork in the road: what was chosen, what was rejected,
and what evidence settled it. Written so the choices can be defended in a viva
rather than rediscovered.

---

## 1. PyTorch instead of TensorFlow/Keras

**Report says:** TensorFlow, Keras (section 4.1, 5.1).

**Built with:** PyTorch + torchvision.

**Why.** TensorFlow has had no native-Windows GPU support since version 2.11.
On the development machine, an RTX 4060 with 8 GB, TensorFlow would have run
CPU-only, leaving the GPU idle and making the three-architecture comparison
impractically slow.

**What is unchanged.** Every *method* the report commits to: a custom CNN
baseline, VGG16 and EfficientNetB0 transfer learning, CLAHE preprocessing,
224×224 input, the same augmentations, Grad-CAM, and accuracy / precision /
recall / F1 / AUC-ROC / cross-validation. Only the library differs.

**Cost.** One sentence in the final report. The alternative was TensorFlow-GPU
under WSL2, which would have split the Flask application and the training
environment across two operating systems for no scientific gain.

---

## 2. E5 embeddings instead of a paraphrase model

**First choice:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

**Final choice:** `intfloat/multilingual-e5-base`.

**Why.** The paraphrase model is trained for *symmetric* similarity, judging
whether two sentences of similar length mean the same thing. Retrieval here is
*asymmetric*: a four-word question against a several-hundred-word clinical
passage.

**The evidence.** Measured on this corpus:

| Question | Paraphrase model | E5 |
|---|---|---|
| "Could it be an infection rather than cancer?" | ependymoma, cns-lymphoma, craniopharyngioma | **neurocysticercosis, ring-enhancing differential, DWI, abscess** |
| "How much will it cost?" | *nothing above threshold* | **nepal-imaging-costs** |
| "Where do I go?" | *nothing* | **hospitals, referral pathway, transport** |
| "Is it serious?" | *nothing* | **emergency-recognition** |

The first row is the one that mattered. The ring-enhancing differential exists
specifically to stop infection being mistaken for cancer in a Nepali patient,
and the paraphrase model did not return it in the top ten.

**Note.** E5 requires `query: ` and `passage: ` prefixes. They are not optional
decoration, they are how the model distinguishes the two roles, and omitting
them costs a large share of its retrieval quality. See
`vectorstore._make_e5_embeddings`.

---

## 3. Relative score floor instead of a fixed absolute one

**Problem.** A fixed threshold of 0.25 discarded every short question. Cosine
similarity against a long chunk is systematically lower for a four-word query
than a fifteen-word one, so any absolute cut-off either rejects brief questions
or admits noise on long ones.

**Solution.** A low absolute floor (0.10) to remove genuine garbage, plus a
per-query relative floor, keep chunks scoring at least half the best hit *for
that query*. This adapts to query length automatically and assumes no absolute
scale.

---

## 4. Multi-query retrieval, primary query authoritative

**Rejected first attempt.** Merge all query variants by best score. The longer,
keyword-stuffed expansion reliably outscored the user's own wording, and every
question, including "How much will it cost?", returned the same generic set
of tumour documents. A verbose expansion embeds as "generic clinical text" and
matches every long document equally, destroying the signal the user supplied.

**Final design.** The user's wording is query one and is never displaced.
Expansions only fill remaining slots, are filtered harder, and are applied only
to questions that actually depend on an unstated referent (`_is_anaphoric`).

**Note.** After switching to E5, single-query retrieval performs as well as
multi-query on the test cases. The machinery remains as a safety net but is no
longer load-bearing, the embedding model was the real problem.

---

## 5. Threshold at the midpoint of the tied plateau

**Problem.** When classes separate well, every threshold in the gap between the
highest-scoring normal and the lowest-scoring abnormal gives identical, perfect
validation numbers. Choosing either end places the operating point flush
against a validation score with zero margin.

**Evidence.** Selecting the plateau endpoint gave 0.857 test accuracy with
0.714 recall. Selecting the midpoint, from the same trained weights, gave 0.964
accuracy and 1.000 recall.

**Solution.** Take the median of the tied plateau, the standard max-margin
argument.

---

## 6. Recall floor raised from 0.90 to 0.95

**Evidence.** With a 0.90 floor on Br35H, the F1-maximising tuner chose high
thresholds and produced:

| Model | Threshold | Missed abnormalities | False alarms |
|---|---|---|---|
| baseline_cnn | 0.895 | 14 | 0 |
| efficientnet_b0 | 0.783 | 17 | 0 |

EfficientNetB0 achieved an AUC of 0.9994, near-perfect separation was
available, yet missed 17 of 231 abnormal scans purely because of where the
threshold sat.

**Why that is the wrong trade.** A false alarm costs a radiologist's review. A
miss costs the diagnostic delay this project exists to prevent. F1 treats them
as equal; triage does not.

**First attempt: raise the floor to 0.95.** This did not help, and the reason
is instructive. A floor is a *constraint*: it stops binding the moment it is
satisfied. EfficientNetB0 cleared a 0.95 validation recall floor at threshold
0.783 (val recall 0.967), so the search had no reason to look any lower, and
still missed 17 abnormal scans on test. The constraint was the wrong lever.

**Actual fix: change the objective to F2.** F-beta with beta=2 weights recall
twice as heavily as precision. It states the clinical trade-off explicitly in
what the search optimises, rather than hoping a side constraint enforces it.
The 0.95 floor is retained as a backstop.

`evaluation.threshold_metric = "f2"`, `evaluation.min_recall = 0.95`.

**Measured effect.** Identical weights and splits; only the operating point
differs, so this isolates the objective:

| Model | Objective | Threshold | Accuracy | Recall | Missed | False alarms |
|---|---|---|---|---|---|---|
| baseline_cnn | F1 | 0.895 | 0.9697 | 0.9394 | 14 | 0 |
| baseline_cnn | **F2** | 0.535 | **0.9870** | **0.9827** | **4** | 2 |
| vgg16 | F1 | 0.516 | 0.9762 | 0.9740 | 6 | 5 |
| vgg16 | **F2** | 0.193 | 0.9719 | **0.9913** | **2** | 11 |
| efficientnet_b0 | F1 | 0.783 | 0.9632 | 0.9264 | 17 | 0 |
| efficientnet_b0 | **F2** | 0.536 | **0.9870** | **0.9827** | **4** | 2 |

**27 abnormal scans across the three models are no longer missed.**

Two of the three improved on accuracy *as well as* recall, which shows the F1
thresholds were not merely a different trade-off but genuinely badly placed:
they maximised F1 on a narrow validation plateau and generalised poorly. Only
VGG16 made a real trade, giving up 0.43% accuracy for four fewer missed
tumours, which is the trade this system should make every time.

---

## 6a. Deployment selection leads on recall, not latency

**The bug.** `select_best_architecture` ranked every eligible model by CPU
latency and took the cheapest. On the measured results that selected
EfficientNetB0, **the worst-recall model of the three**: missing 17 abnormal
scans against VGG16's 6, in order to save 360 ms per scan.

**Why it is wrong.** A district hospital reads a handful of scans a day. The
difference between 47 ms and 409 ms is imperceptible in that setting. Eleven
additional missed tumours is not.

**Fix.** Rank eligible models by recall. Efficiency now only breaks ties among
models within 2 percentage points of the best recall, which is where it
genuinely is a free choice. The trade actually made, if any, is quantified in
the recorded rationale so the decision is auditable.

---

## 7. Near-duplicate grouping before splitting

**Discovery.** A visual check of an augmented training batch showed two
near-identical images. Investigating with a difference hash found **1,853
exact-duplicate pairs covering 1,377 of 3,000 Br35H images, 46%**.

**Consequence if ignored.** A large share of the test set has a twin in the
training set, so the reported accuracy measures memorisation.

**Solution.** Cluster duplicates by dHash and give each cluster a shared
patient id, which the existing patient-grouped splitter then keeps together.

**On the threshold.** It must be tight. Brain MRIs are globally similar, so
union-find chains transitively. Measured largest-cluster size: 14 at Hamming 0,
27 at 1, **140 at 2**: 1381 at 5. The step change at 2 is chaining, not
duplication, so 1 is the largest defensible value.

---

## 8. Nepali rendered as images in PDFs

**Problem.** ReportLab maps characters to glyphs one at a time with no complex
text shaping. Devanagari needs it: the short-i vowel sign is typed after its
consonant but drawn before it, and consonant clusters form conjunct ligatures.
Without shaping the output is visibly wrong to any Nepali reader.

**Solution.** Render Nepali passages through Pillow, which applies the font's
own layout tables, and embed the result as images. Verified visually before
adoption.

**Cost.** Nepali text in the PDF is not selectable or searchable. Correctness
is worth more, and the decision is recorded so it is not mistaken for an
oversight.

---

## 9. Degrade rather than fail

**Principle.** For a clinic machine, degraded is far better than down.

- No Ollama → advisory shows retrieved source text verbatim, attributed
- No FAISS index → classification and Grad-CAM still work
- No model → the app starts and explains what is missing
- Unreadable image in a batch → a blank frame is substituted and logged once

The exception is safety: if retrieval returns nothing above threshold,
generation is **skipped entirely** rather than degraded. An unanswered question
is safe; a fabricated medical answer is not.

---

## 10. Validation only on generated text

**Bug found in testing.** The prohibited-pattern screen was being applied to
verbatim corpus text, firing on ordinary clinical prose, "the diagnosis is
usually built from", and "replacing" corpus text with corpus text.

**Fix.** Providers declare `is_generative`. The screen runs only on model
output; the curated corpus has already been reviewed. The over-broad
`diagnosis is` pattern was also narrowed to assertive constructions only.

---

## 13. The operating threshold is the least stable part of the system

**What was observed.** Fixing the deduplication grouping changed the *names* of
patient groups, and `build_splits` seeds its shuffle from the sorted group keys,
so the partition shifted - same data, same seed, same code otherwise. The
results moved far more than that change should warrant:

| Model | Partition | Tuned threshold | Accuracy | Recall | Missed | False alarms |
|---|---|---|---|---|---|---|
| baseline_cnn | A | 0.535 | 0.9870 | 0.9827 | 4 | 2 |
| baseline_cnn | B | **0.160** | 0.9089 | 1.0000 | **0** | **41** |
| vgg16 | A | 0.193 | 0.9719 | 0.9913 | 2 | 11 |
| vgg16 | B | 0.340 | 0.9756 | 0.9648 | 8 | 3 |

**Diagnosis.** The model weights are stable - AUC barely moves (0.9934 vs
0.9969 for baseline_cnn). What moves is the **threshold**: which is fit on a
single validation split of roughly 450 images by maximising F2. On partition B
that maximum sat at 0.160, a very permissive operating point, and the plateau
had size 1 - no tied alternatives to average over. The classifier is not
unstable; the operating point chosen for it is.

**Why this matters more than it looks.** Every headline number in this project
is reported at that threshold. A figure that moves from "4 missed, 2 false
alarms" to "0 missed, 41 false alarms" on a repartition is not a property of the
model, and quoting either as *the* result would be misleading.

**Response.** Report cross-validated results with a standard deviation rather
than a single split - which is what Project Design 4.1 committed to in the first
place - and state the threshold sensitivity as a limitation. AUC, being
threshold-independent, is the more stable comparison statistic and is already
what breaks ties in model selection.

A defensible further step, not taken here for scope: select the threshold across
cross-validation folds rather than on one validation split, so the operating
point inherits the same averaging as the metrics.

**Resolution - measured.** 5-fold cross-validation on efficientnet_b0, each
fold tuning its own threshold:

| Metric | Mean ± sd | Fold range |
|---|---|---|
| Accuracy | 0.9733 ± 0.0128 | 0.9552-0.9912 |
| Recall | 0.9895 ± 0.0082 | 0.9778-1.0000 |
| AUC-ROC | 0.9970 ± 0.0019 | 0.9941-0.9989 |
| Specificity | 0.9548 ± 0.0245 | 0.9296-0.9956 |

Across five partitions and five independently tuned thresholds, recall stays
within a 2.2-point band and every fold clears the 90% accuracy target. The
instability observed on baseline_cnn was real, but for the deployed
architecture the F2 objective lands on a consistently sensitive operating
point. These cross-validated figures are the ones any write-up should quote;
any single-split number gets a standard deviation next to it or does not get
quoted.

---

## 12. A corrected workforce statistic

**The claim, as inherited from the original project brief.** "Nepal has 30-40
neurosurgeons for 30 million people, one per 750,000, against a WHO-referenced
target of one per 100,000", cited to Shrestha (2024) and Alokozay et al. (2025).

**Both halves are wrong, and the citation refutes the claim it supports.**

Shrestha P (2024), *Neurosurgical Workforce of Nepal*, World Neurosurgery
189:161-165, the paper cited, reports **114 neurosurgeons** registered with the
Nepalese Society of Neurosurgeons, a density of **0.39 per 100,000**: i.e. about
one per 256,000. Alokozay et al. (2025) independently reports the same 114 and
0.39/100,000. The "30-40" figure is roughly a third of the true count; even
Nepal's 2016 density of 0.166/100,000 implies about 48.

The **one-per-100,000 benchmark is not WHO's.** Alokozay et al. attribute it to
the American College of Surgeons and the American Association of Neurological
Surgeons. WHO material in this area cites a different figure entirely.

**Corrected wording used throughout the project:**

> Nepal has 114 neurosurgeons, 0.39 per 100,000, roughly one per 256,000, > against a benchmark of one per 100,000 cited by the American College of
> Surgeons and the AANS. Closing that gap needs approximately 178 more
> neurosurgeons, at a current training rate of about 11 per year.

**Why this is recorded rather than quietly fixed.** The corrected figures still
describe a severe access problem, so the project's motivation is unchanged. But
a reviewer who opens the cited paper will find it contradicts the claim, and a
project that carries that unremarked has a real defect. Any document that
inherited the original figure needs the same correction.

**How it was found.** Not by a test, by an adversarial reviewer that checked the
document's assertions against its own cited sources, then verified independently.

---

## 11. In-memory analysis store, no database

**Why.** A single-process offline deployment does not warrant one, and
persisting scan results would contradict the retention commitment in
`docs/ETHICS.md`. Results live in process memory and do not survive a restart, which is the correct behaviour for clinical images, not a limitation.
