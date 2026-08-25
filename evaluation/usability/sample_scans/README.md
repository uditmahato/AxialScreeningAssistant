# Demonstration Scans

Used in usability sessions. **Never use real patient scans in a usability
session**, participants have not consented to view clinical data, and the
patients have not consented to it being shown.

The images themselves are not committed. Br35H's redistribution terms are the
publisher's to set, and the repository should not silently re-host a
third-party dataset.

## Populate

After `python scripts/download_data.py --dataset br35h`:

```bash
python scripts/prepare_demo_scans.py
```

This selects a fixed, seed-stable set from the public dataset and copies it
here as `demo_01.jpg` … `demo_06.jpg`.

## What the set contains

| File | Expected | Used in |
|---|---|---|
| `demo_01.jpg` | abnormal, clear lesion | Task 1, 2, 3, 4 |
| `demo_02.jpg` | abnormal, subtler | Spare |
| `demo_03.jpg` | abnormal | Spare |
| `demo_04.jpg` | normal | Contrast case |
| `demo_05.jpg` | normal | Contrast case |
| `demo_06.jpg` | normal | Spare |

Facilitators should **not** tell participants what the expected answer is, task 2 tests whether the participant understands that the system does not
provide a diagnosis, and priming them defeats the measurement.

## A note on expectations

These are public research images, not Nepali clinical scans, and the model may
be wrong on any of them. That is fine for a usability study: the question is
whether people can *use and correctly interpret* the system, not whether the
classifier is right. A wrong prediction during a session is in fact a useful
prompt for the debrief question about trust.
