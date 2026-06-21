#!/usr/bin/env python3
"""
Your first brain decoder, on REAL human EEG.
-------------------------------------------------
Dataset: PhysioNet EEG Motor Movement/Imagery (EEGBCI), via MNE.
Task: decode whether the subject is imagining moving their LEFT vs RIGHT fist,
      purely from 64-channel scalp EEG.

This is the canonical, published motor-imagery decoding benchmark. The method
below (Common Spatial Patterns + LDA) is the classic approach reported across
the BCI literature, so your result is directly comparable to published numbers
(typically ~70-85% for this 2-class task depending on subject).

RUN THIS ON YOUR MAC (PhysioNet must be reachable):
    pip install mne scikit-learn numpy
    python decode_motor_imagery.py

First run downloads ~50MB of real EEG. Later runs use the cache.
"""

import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

# ----------------------------------------------------------------------
# 1. LOAD REAL DATA
#    Runs 4, 8, 12 = imagined movement of left vs right fist.
# ----------------------------------------------------------------------
SUBJECT = 1
RUNS = [4, 8, 12]

print(f"Loading EEGBCI subject {SUBJECT}, runs {RUNS} (real human EEG)...")
raw_fnames = eegbci.load_data(subjects=SUBJECT, runs=RUNS)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)                      # fix channel names to standard 10-05
raw.set_montage("standard_1005")             # electrode positions

# ----------------------------------------------------------------------
# 2. PREPROCESS
#    Motor imagery lives in the mu (8-12 Hz) and beta (13-30 Hz) bands,
#    so we bandpass 7-30 Hz. This is where the discriminative signal is.
# ----------------------------------------------------------------------
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

# Events: annotation 'T1' = left fist, 'T2' = right fist
events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude="bads")

# ----------------------------------------------------------------------
# 3. EPOCH
#    Cut 1-2s windows time-locked to each imagined movement.
# ----------------------------------------------------------------------
tmin, tmax = -1.0, 4.0
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                picks=picks, baseline=None, preload=True)
# Use the 0.5-2.5s window after cue: that's when imagery is strongest
epochs_train = epochs.copy().crop(tmin=0.5, tmax=2.5)

labels = epochs.events[:, -1] - 2            # -> 0 = left, 1 = right
X = epochs_train.get_data(copy=False)        # shape: (n_trials, n_channels, n_times)
y = labels
print(f"\nReal data loaded: {X.shape[0]} trials, "
      f"{X.shape[1]} channels, {X.shape[2]} samples/trial")
print(f"Class balance: {int((y==0).sum())} left, {int((y==1).sum())} right")

# ----------------------------------------------------------------------
# 4. DECODE
#    CSP extracts spatial filters that maximize variance difference
#    between the two classes; LDA classifies. Cross-validated so the
#    accuracy is honest (tested on data the model never saw).
# ----------------------------------------------------------------------
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([("CSP", csp), ("LDA", lda)])

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

chance = max(np.mean(y == 0), np.mean(y == 1))
print("\n" + "=" * 50)
print(f"DECODING ACCURACY: {scores.mean()*100:.1f}%  (+/- {scores.std()*100:.1f}%)")
print(f"Chance level:      {chance*100:.1f}%")
print("=" * 50)
print("\nYou just decoded imagined movement from a human brain.")
print("Next: loop over subjects 1-109, or swap RUNS=[6,10,14] for hands-vs-feet.")
