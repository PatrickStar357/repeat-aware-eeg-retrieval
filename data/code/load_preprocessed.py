"""Memory-mapped access to strict preprocessing output; no averaging of repeats."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load_subject(subject='sub-01',task='train'):
    folder=ROOT/'processed'/subject
    eeg=np.load(folder/(task+'_eeg.npy'),mmap_mode='r',allow_pickle=False)
    ids=dict(np.load(folder/(task+'_identities.npz'),allow_pickle=False))
    if task=='train' and eeg.shape[:2]!=(16540,4):raise ValueError('Expected canonical four-repeat training pool')
    return eeg,ids
def official_test_four(session_ids):
    """Deterministic one-per-session R4 control; no selection using EEG/labels."""
    out=[]
    for row in session_ids:
        sessions=sorted(set(row))
        if len(sessions)!=4:raise ValueError('Official test must retain four source sessions')
        out.append([int(np.flatnonzero(row==s)[0]) for s in sessions])
    return np.asarray(out)
