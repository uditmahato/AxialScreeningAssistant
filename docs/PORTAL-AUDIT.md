# Portal design audit: `design/portal-refinement`

Audit-first refinement of the web portal. The existing clinical design system
was kept: its colour discipline, system-font stack, neutral normal-verdict
styling and bilingual structure are deliberate and were not touched.

Four independent review passes were run, craft floor, mobile/low-end device,
accessibility (WCAG 2.1 AA), and bilingual/clinical clarity. Contrast ratios
were computed rather than eyeballed. Findings that contradict a deliberate
decision are recorded below as **dismissed**: with the reason, rather than
silently dropped.

---

## Fixed

### Clinical safety

**The disclaimer was not routed through `neuroscan.safety`.**
`app.py` built `disclaimer` and `disclaimer_short` from `get_disclaimer()` into
every template context, and **no template referenced either**. The footer
rendered `t.footer_disclaimer` and the result page rendered
`t.clinical_support_body`, a second copy of the disclaimer living in
`chatbot/language.py`. This is the exact duplication the project's
non-negotiables forbid: two sources of truth for the one string that must never
drift. The templates now use the context values; the duplicate keys are gone
from the string table, with a note pointing at `neuroscan.safety`.

**The printed sheet had no disclaimer and no emergency numbers.**
`@media print` hid `.site-footer` outright. The printout is the artefact that
leaves the building and gets filed in a patient record. The footer now prints
in a quiet 9pt form, with the nav and tagline still suppressed.

**A printed record dropped its own provenance.** The print block never forced
`<details>` open, so a sheet printed without expanding first lost the
architecture, the decision threshold, the scan reference and the entire
citation list, everything needed to audit the finding later.

**The confidence bar printed as nothing.** It is a background-colour-only
indicator over a background-colour-only track, and browsers default background
graphics off. Now has `print-color-adjust: exact` plus a border fallback.

### Accessibility

**`--text-muted` failed AA.** `#7A8580` computed to 3.82:1 on white and 3.53:1
on the page background, at body sizes. It carries the upload instructions, the
Grad-CAM caveat caption, the inactive scan-tab label, table headers, and the
emergency phone numbers in the footer, read on cheap panels in daylight.
Darkened to `#616B67`: 5.52:1 on white, 5.10:1 on page, and ≥4.74:1 on every
alert tint.

**Form control boundaries failed 1.4.11.** `--border-strong` is 1.57:1 on
white. That is fine for a decorative panel edge and not fine for the only
visual boundary of a text input. Rather than darken every border and change
the design's quiet character, a separate `--border-control` (`#86918B`,
3.26:1) now applies to the dropzone, the question input and the history
search, the three controls whose border is their only affordance.

**The scan-tab focus ring outlined both tabs at once.**
`.tab-radio:focus-visible ~ .tab-labels label`, the descendant combinator
matched every label in the strip, so arrowing between tabs showed no focus
movement at all. Replaced with per-radio pairing.

**The dropzone hid its own file input from screen readers.** The wrapper was
`role="button"`, which is Children Presentational, stripping the file input,
the format list and the "ready to analyse" confirmation from the accessibility
tree, while adding a second nameless tab stop. The role and `tabindex` are
gone; the native input is the control, labelled and `aria-describedby` the
format hint and the error.

**The upload error never announced.** `role="alert"` only fires on a mutation
to a node already in the accessibility tree; the code set `textContent` while
the element was `hidden`, then unhid it. Order reversed, plus `aria-invalid`
on the input so the state is not conveyed by the red border alone.

**Skip link did not move focus.** `<main>` is not natively focusable, so on
WebKit the hash changed and focus stayed put, the next Tab walked the whole
header again. Added `tabindex="-1"`.

**The confidence bar contradicted the visible number.** It carried
`role="img"` with an aria-label rounding to `%.0f` while the adjacent text
rendered `%.1f`, "87.3%" on screen, "87 percent" announced, in a clinical
context. The bar is now `aria-hidden`; the text already conveys the value.

**Hardcoded English inside `lang="ne"`.** The skip link and both scan `alt`
texts were English strings rendered under a Nepali document language, so a
Nepali TTS voice read them phonetically. Now translated, and the alt text
describes the image rather than restating the caption.

**JS smooth scroll ignored reduced-motion.** The CSS `prefers-reduced-motion`
block cannot reach `scrollIntoView({behavior:"smooth"})`. Now checks
`matchMedia`.

### Mobile and low-end device

**The Grad-CAM was never fetched until tapped.** A `loading="lazy"` image
inside a `display:none` panel generates no box, never intersects the viewport,
and is not requested. Tapping "Highlighted" started the download at that
moment, on a link that may already be gone, showing a black square with no
spinner meanwhile. Lazy loading removed from both scan images; the primary one
was also the LCP element.

**Back button left the form permanently dead.** Submit disabled the button and
started a `setInterval` that was never stored or cleared. Returning via the
Android back gesture restores that same DOM from the bfcache, so the next
patient's scan met the previous thumbnail, a running ticker and a greyed-out
Analyse button, recoverable only by a full reload. Added a `pageshow` reset
that respects a server-set disable.

**Every page jumped on load.** The mobile nav collapse is gated on `html.js`,
set by a deferred script at the end of `<body>`, so each page painted the
expanded five-link nav and then shifted ~160px. Now set in `<head>`.

**No client-side size check.** The server limit was enforced only after the
whole upload, so a camera photo of an MRI film could spend minutes uploading
into a guaranteed 413. The limit is now passed to the template, checked before
preview, and stated in the format hint.

**`FileReader` had no error handler.** A revoked SD card or a denied storage
permission meant `onload` never fired: no preview, no error, Analyse stuck
disabled, nothing on screen explaining why.

**Touch targets below 44px.** Raised: the red-flag disclosure (23px, the most
safety-critical progressive disclosure in the product), the help-page
summaries (25px, the entire page is nine of them), the citation disclosures
(20-22px), the chat/filter chips (38px), the scan tabs (40.8px), the header
nav links (41-43px).

**The inactive scan tab looked disabled.** It used `--text-muted` next to a
`--text-primary` active tab, with hover as its only other affordance, so on
touch the Grad-CAM view read as greyed-out metadata rather than a second tab.
Now `--text-secondary`.

### Craft

- `:active` was identical to `:hover` on the primary button, so a tap produced
  no feedback at all, and on touch there is no hover, making `:active` the
  only feedback that exists. Added to primary, secondary and ghost.
- `:hover` on secondary and ghost buttons lacked the `:not(:disabled)` guard
  the primary button has.
- Filtering history to zero rows left a bare header, indistinguishable from a
  broken filter. Added a no-match state.
- `::selection` and `caret-color` were OS defaults, the only non-palette
  colours on the page. Scroll regions were unthemed.
- The error code was rendered in a border colour at ~1.5:1.
- 12 inline `style=` attributes became classes, including two different
  `font-size` values on `h2` three lines apart, and one that overrode
  `.info-note`'s own margin rule. The one remaining inline style is the
  computed confidence-bar width, which cannot be a class.
- Long-form safety caveats on the About page ran ~160 characters per line at
  1280px. Capped at 70ch.

---

## Found and deliberately NOT changed

**Dismissed, these are correct decisions, not defects.** A generic design
review flags all of them; in this codebase each is right.

| Flagged | Why it stays |
|---|---|
| `Inter` heads the font stack | Deliberate: the system stack renders Devanagari correctly on both Windows and Android. A "premium font" swap would break the Nepali interface. |
| Four accent colours | Semantic colour coding in a clinical tool, teal structure, orange abnormal, red emergency, green operational. Collapsing to one accent would destroy meaning. |
| `.eyebrow` uppercase labels | The quiet voice for section metadata. Some design skills hard-ban eyebrows; that rule is for marketing pages. |
| Normal verdict is not green | Explicitly commented: green would read as "the patient is fine". |
| No framework, no build step | Required by the deployment target. |
| `--border` at 1.31:1 on panels | Decorative panel edge, not a control boundary. 1.4.11 does not apply. |

---

## Open, not addressed on this branch

Real findings, left for a decision rather than changed unilaterally.

1. **The question form does not degrade without JavaScript.** `<form
   class="chat-form">` has no `action` and no `method`; the endpoint lives only
   in `data-endpoint`. With JS blocked, submitting issues a GET and silently
   loses the question. `app.py` already reads `request.form.get("question")`
   as a POST fallback, that path is unreachable from the markup. This
   contradicts the progressive-enhancement promise in `app.js`'s own header
   comment.

2. **No failure state for a heat map that did not generate.**
   `result.heatmap_image_url` renders unconditionally. Per the pipeline's
   documented failure modes,
   Grad-CAM silently produces a blank map under autocast and with an
   `inplace=True` VGG16 ReLU, so this is a known failure mode with no UI for
   it. The user cannot tell "the model saw nothing" from "the image failed".

3. **Chat suggestion chips are not disabled during an in-flight request.**
   `setBusy()` covers the send button and input but not the chips, and `ask()`
   has no re-entrancy guard. Two rapid taps fire two concurrent POSTs, and
   `session["chat_history"]` is appended in arrival order, the transcript,
   which is embedded in the PDF report, can attach answers to the wrong
   questions.

4. **The dropzone stays interactive when no model is loaded.** The server
   disables the submit button; `showPreview` then sets `submitBtn.disabled =
   false` unconditionally. The `data-server-disabled` marker added here fixes
   the bfcache path but not this one.

5. **62 inline `{% if language == 'ne' %}` blocks**: and they have already
   drifted: `base.html` gives English users the training command and Nepali
   users nothing actionable; `about.html`'s system-status list and every
   metric label have no Nepali branch at all. Parity is not enforceable in
   this form, these belong in the string table.

6. **`.eyebrow` sets `letter-spacing: .12em`.** Positive tracking breaks
   Devanagari conjunct ligatures, and this label sits directly above the
   verdict. Needs a `:lang(ne)` override rather than removal.

7. **Devanagari numerals are used in exactly one place** (`base.html`
   emergency numbers) while the same ambulance number renders as `102` in
   `result.html`, and confidence percentages are always Latin.

8. **`.chat-log` is a `max-height: 560px` scroll region**: roughly a whole
   phone viewport, that cannot receive keyboard focus, so a keyboard user
   cannot scroll back through earlier answers.

9. **The full 60-item finding list** is longer than this. Items not listed
   here were either low severity or refuted on verification.

---

## Verification

- Contrast computed for all 17 token pairs before and after; every previously
  failing pair now passes at its required threshold.
- All seven templates parse under Jinja2.
- `app.py` and `language.py` compile; `app.js` parses.
- **`pytest -q` was not run**: the connected-folder VM has no pytest and no
  project environment. Run it locally before merging.
