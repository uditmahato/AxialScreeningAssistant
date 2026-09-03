# What this changes

<!-- What was wrong, and what this does about it. A reader should not have to
open the diff to understand the point. -->

## Why

<!-- The reasoning, including anything you deliberately did not do and why.
That is often the most useful part of a pull request a year later. -->

## Checks

<!-- Both must be clean. Paste the result rather than asserting it. -->

- [ ] `ruff check src scripts tests`
- [ ] `pytest -q`

```
paste the summary lines here
```

## Safety

<!-- Delete the ones that do not apply, but read them all first. The rules
are set out in CONTRIBUTING.md. -->

- [ ] No user-facing safety text was inlined outside `src/neuroscan/safety.py`
- [ ] The model still answers only from retrieved context, and empty retrieval
      still skips generation
- [ ] Question screening before retrieval and output validation after are both
      still in place
- [ ] Red flag guidance is still unconditional
- [ ] No dosage, prognosis, or survival figure was added to the corpus or the
      prompts
- [ ] A normal classification still lists no disease possibilities
- [ ] Leakage assertions in the splitting code were not weakened
- [ ] Nothing under `data/` is committed, and no clinical data was used

## Tests

<!-- What did you add, and would it fail without this change? Changes to
safety.py, the advisory pipeline, or splitting need a test that would. -->

## Anything a human still needs to check

<!-- For example: Nepali wording that needs a native speaker, a clinical claim
that needs a second opinion, or a measurement someone should reproduce. Say
"nothing" if there is nothing. -->
