#!/usr/bin/env python3
"""
Motor-imagery decoding across the FULL PhysioNet EEGBCI cohort (109 subjects).
-----------------------------------------------------------------------------
This is the reproduction artifact. It runs the same CSP+LDA decoder you already
validated on subject 1, but across every subject, and reports the cohort mean.

That cohort mean is the number that matches how the literature reports this
benchmark — it's what makes this a *reproduction* rather than a one-off.

RUN:
    python decode_cohort.py

Takes a few minutes (downloads ~50MB per subject the first time, then caches).
Results are saved to results.csv and a summary is printed at the end.
"""

import warnings
import numpy as np
import csv
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import mne

mne.set_log_level("ERROR")          # quiet the per-subject chatter
warnings.filterwarnings("ignore")

RUNS = [4, 8, 12]                    # imagined left vs right fist
# These subjects have known recording problems (different sample rate / bad
# annotations) and are conventionally excluded in the literature:
BAD_SUBJECTS = {88, 89, 92, 100, 104}
ALL_SUBJECTS = [s for s in range(1, 110) if s not in BAD_SUBJECTS]


def decode_subject(subject):
    """Run the full pipeline for one subject, return CV accuracy or None."""
    raw_fnames = eegbci.load_data(subjects=subject, runs=RUNS)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                       eog=False, exclude="bads")

    epochs = Epochs(raw, events, event_id, tmin=-1.0, tmax=4.0, proj=True,
                    picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=0.5, tmax=2.5)

    y = epochs.events[:, -1] - 2
    X = epochs_train.get_data(copy=False)

    clf = Pipeline([("CSP", CSP(n_components=4, log=True, norm_trace=False)),
                    ("LDA", LinearDiscriminantAnalysis())])
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    return scores.mean()


def main():
    print(f"Decoding {len(ALL_SUBJECTS)} subjects "
          f"(excluding {sorted(BAD_SUBJECTS)} — known bad recordings)\n")

    results = []
    for i, subject in enumerate(ALL_SUBJECTS, 1):
        try:
            acc = decode_subject(subject)
            results.append((subject, acc))
            print(f"  [{i:3d}/{len(ALL_SUBJECTS)}] subject {subject:3d}: "
                  f"{acc*100:5.1f}%")
        except Exception as e:
            print(f"  [{i:3d}/{len(ALL_SUBJECTS)}] subject {subject:3d}: "
                  f"SKIPPED ({type(e).__name__})")

    accs = np.array([a for _, a in results])

    # Save for your GitHub repo
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "accuracy"])
        for s, a in results:
            w.writerow([s, f"{a:.4f}"])

    print("\n" + "=" * 55)
    print(f"COHORT RESULTS  (n = {len(accs)} subjects)")
    print("=" * 55)
    print(f"  Mean accuracy:    {accs.mean()*100:5.1f}%")
    print(f"  Std deviation:    {accs.std()*100:5.1f}%")
    print(f"  Best subject:     {accs.max()*100:5.1f}%")
    print(f"  Worst subject:    {accs.min()*100:5.1f}%")
    print(f"  Above 70%:        {(accs > 0.70).sum()} subjects")
    print(f"  Chance level:     ~50%")
    print("=" * 55)
    print("\nSaved per-subject results to results.csv")
    print("This cohort mean is your reproduction number. Put it in the README.")


if __name__ == "__main__":
    main()
