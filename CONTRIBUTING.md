# Contributing

Thank you for looking at this. Before anything else, please read the safety
rules below. They are not style preferences; they are the constraints that
make this project defensible, and a change that breaks one will be declined
however good the rest of it is.

## What this project is, and is not

It is a research prototype that triages a single axial brain MRI slice as
normal or abnormal, and pairs that with a retrieval-grounded advisory in
English or Nepali. It is decision support.

It is not a certified medical device, it does not diagnose, and it has not
been validated on hospital data. On decontaminated external data it keeps
recall above 98% but reaches only about 60% balanced accuracy, so it
over-refers. Please keep contributions honest about that.

## Safety rules that are not negotiable

1. **Every user-facing output carries a disclaimer, routed through
   `src/neuroscan/safety.py`.** Never inline disclaimer text anywhere else.
   If you need a new user-facing safety string, add it to that module beside
   the others so there is one place to audit.
2. **The language model answers only from retrieved context.** If retrieval
   returns nothing above threshold, generation is skipped and the fixed safe
   response is returned. Never let the model fall back on its own knowledge
   for a medical question.
3. **Questions are screened before retrieval, generated text after.** Both
   are required. Removing or weakening either screen needs a very good reason
   and a test that shows what replaces it.
4. **Red flag emergency advice is shown regardless of what the model
   predicted.** A normal verdict on a deteriorating patient is the most
   dangerous output this system can produce, so the escalation path is
   unconditional.
5. **No dosages, prognosis, or survival figures** in the knowledge base, the
   prompts, or anywhere the model can reach them.
6. **A normal classification never lists disease possibilities.** This is
   enforced in more than one place on purpose.

If you find a way around any of these, that is a security report rather than
a pull request. See [SECURITY.md](SECURITY.md).

## Data rules

- **Never commit anything under `data/`.** It is git-ignored for a reason.
- No clinical or hospital data is used anywhere in this project, and none may
  be added without ethics approval. A dataset configuration can declare
  `requires_ethics_approval: true`, and data discovery then refuses to load it
  until an approval marker file exists. Do not weaken that guard.
- Splits are grouped by patient **and** by near-duplicate cluster, and
  `SplitResult` asserts against leakage when it is constructed. Do not relax
  the assertion to make a test pass; the assertion is the test.

## Getting set up

```bash
conda activate neuroscan
pip install -e .
python scripts/download_data.py --dataset br35h
python scripts/build_index.py --rebuild
ollama pull llama3.1:8b     # optional; without it the advisory degrades safely
python scripts/run_app.py
```

The model checkpoint is not tracked in git. Download
`best_efficientnet_b0.pt` from the latest release and place it under
`artifacts/models/`.

## Before you open a pull request

```bash
ruff check src scripts tests
pytest -q                    # full suite
pytest -m "not slow" -q      # skip the long tests while iterating
```

Both must be clean. Please say in the pull request what you ran and what the
result was, rather than leaving the reviewer to guess.

New behaviour needs a test. Changes touching `safety.py`, the advisory
pipeline, or the splitting code need a test that would fail without the
change, because those are the parts where a silent regression matters most.

## Conventions

- British English in prose and in identifiers: `standardise`, `normalise`.
- Comments explain **why**, never what. If a line needs a comment to say what
  it does, rewrite the line.
- Type hints everywhere, with `from __future__ import annotations` at the top.
- Configuration objects are passed in, never read from a global.
- Log through `neuroscan.utils.get_logger(__name__)`.
- New public functions get a docstring with Args, Returns, and Raises.
- No em dashes or en dashes in prose. They render badly on some of the
  surfaces this project writes to.

## Things that will bite you

- **E5 embeddings need `query: ` and `passage: ` prefixes.** They are applied
  in `build_embeddings`. Changing the embedding model means rebuilding the
  index; the loader refuses a mismatch rather than returning nonsense.
- **Grad-CAM must run in float32.** Under autocast the gradients underflow
  and the heat map comes out blank.
- **VGG16's Grad-CAM target ReLU must be `inplace=False`,** or PyTorch
  refuses the backward hook.
- **Devanagari in PDFs is rendered as images,** because ReportLab does no
  text shaping.
- **`build_dataloaders` with `num_workers > 0` deadlocks on Windows** when
  called from a module without an `if __name__ == "__main__":` guard, because
  spawn re-imports the caller. It hangs at near zero CPU rather than raising,
  so it looks like a slow job. Everything in `scripts/` is guarded; use
  `num_workers: 0` from notebooks and ad hoc scripts.

## Translations

The knowledge base is currently English only, and the Nepali interface
strings live in `src/neuroscan/safety.py` and `src/neuroscan/rag/prompts.py`.
Corrections to the Nepali from native speakers are especially welcome, and
are worth more to this project than most code changes.

Please do not add machine-translated clinical content. An unreviewed
translation of medical guidance is a safety risk of its own, and a wrong
Nepali sentence is worse than a correct English one a reader can take to a
health worker.

## Code of conduct

Participation is covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
