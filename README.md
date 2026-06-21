[README.md](https://github.com/user-attachments/files/29177422/README.md)
# eeg-motor-imagery-decoding# EEG Motor Imagery Decoding — PhysioNet EEGBCI Baseline

A from-scratch reproduction of motor-imagery decoding on the PhysioNet EEG Motor
Movement/Imagery dataset (EEGBCI). The decoder classifies whether a subject is
**imagining moving their left vs. right fist**, using only 64-channel scalp EEG.

This is a clean, unoptimized **baseline** using the classic Common Spatial
Patterns + Linear Discriminant Analysis (CSP+LDA) pipeline, run across the full
104-subject cohort with proper cross-validation.

## Results

| Metric | Value |
|---|---|
| Mean accuracy (n=104) | **57.5%** |
| Std deviation | 14.5% |
| Best subject | 100.0% |
| Worst subject | 31.1% |
| Subjects above 70% | 18 |
| Chance level | ~50% |

The mean sits above chance, confirming the method extracts real discriminative
signal across the population. The **large standard deviation is the main finding**:
this fixed-parameter pipeline decodes some subjects extremely well and others at
or below chance. Cross-subject variability — not mean accuracy — is the central
challenge in motor-imagery BCI, and it's visible directly in these results.

This 57.5% is intentionally an *unoptimized* baseline: a single fixed frequency
band (7–30 Hz) and time window (0.5–2.5 s) are applied to every subject, with no
per-subject tuning. Per-subject optimization (the standard in the literature)
typically pushes the cohort mean higher.

## Method

1. **Load** — runs 4/8/12 (imagined left vs. right fist) per subject, via MNE.
2. **Filter** — bandpass 7–30 Hz, isolating the mu and beta rhythms where motor
   imagery is expressed.
3. **Epoch** — extract the 0.5–2.5 s window following each imagery cue.
4. **Features** — CSP (4 components) finds spatial filters maximizing the
   between-class variance difference.
5. **Classify** — LDA, evaluated with 10-fold ShuffleSplit cross-validation
   (80/20), so reported accuracy is on held-out data.

Subjects 88, 89, 92, 100, and 104 are excluded (known non-standard recordings).

## Usage

```bash
pip install mne scikit-learn numpy
python decode_motor_imagery.py   # single subject, fast
python decode_cohort.py          # full 104-subject cohort, saves results.csv
```

First run downloads the dataset (~50 MB/subject) and caches it locally.

## Files

- `decode_motor_imagery.py` — single-subject decoder (start here)
- `decode_cohort.py` — full cohort run, writes per-subject `results.csv`
- `results.csv` — per-subject accuracies

## Next steps

- **Per-subject tuning** of frequency band and time window
- **EEGNet** or other compact CNNs in place of CSP+LDA
- **Subject-specific channel selection** over the motor cortex
- Investigating *why* some subjects decode near-perfectly and others near chance

## Data

Schalk, G., et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface
System. *IEEE TBME*. Dataset via PhysioNet (physionet.org), accessed through
MNE-Python.
