"""Category-disjoint protocol and repeat grouping. No data are downloaded."""
from itertools import combinations
import numpy as np

COUNTS = (1254, 100, 100, 200)
NAMES = ('train', 'validation', 'calibration', 'fine_test')
SPLIT_SEED = 20261102

def category_split(class_ids, counts=COUNTS, seed=SPLIT_SEED):
    classes = np.array(sorted(set(map(str, class_ids))))
    if len(classes) != sum(counts):
        raise ValueError(f'Expected {sum(counts)} classes, found {len(classes)}')
    shuffled = np.random.default_rng(seed).permutation(classes)
    cuts = np.cumsum((0,) + tuple(counts))
    return {name: shuffled[cuts[i]:cuts[i+1]] for i,name in enumerate(NAMES)}

def repeat_groups(session_ids):
    """Deterministically prefer non-overlapping session sets; no outcome input."""
    if len(session_ids) != 4:
        raise ValueError('Main protocol requires exactly four recordings per image')
    sessions = np.asarray(session_ids).astype(str)
    options=[]
    for a in combinations(range(4), 2):
        b=tuple(i for i in range(4) if i not in a)
        overlap=len(set(sessions[list(a)]) & set(sessions[list(b)]))
        options.append((overlap,a,b))
    _,a,b=min(options)
    return np.array(a),np.array(b)

def audit_cache(data, splits):
    required={'tokens','global_eeg','vision','image_id','class_id','session_id',
              'recording_id','preprocessing_fit_classes','encoder_fit_classes',
              'token_fit_classes','encoder_sha256','preprocessing_sha256','subject_id'}
    if not required <= set(data):
        raise ValueError('Missing cache provenance/arrays: '+str(required-set(data)))
    ids=np.asarray(data['image_id']).astype(str)
    labels=np.asarray(data['class_id']).astype(str)
    n=len(ids)
    if np.asarray(data['subject_id']).ndim!=0 or not str(np.asarray(data['subject_id']).item()):
        raise ValueError('One non-empty scalar subject_id is required per cache')
    if len(set(ids)) != n: raise ValueError('image_id must be globally unique')
    if data['tokens'].ndim!=4 or data['tokens'].shape[:2]!=(n,4):
        raise ValueError('tokens must be [images, 4 repetitions, positions, dim]')
    if data['global_eeg'].ndim!=3 or data['global_eeg'].shape[:2]!=(n,4):
        raise ValueError('global_eeg must retain four repetitions')
    if data['vision'].shape != (n,data['global_eeg'].shape[-1]):
        raise ValueError('vision and aligned global EEG dimensions must match')
    for name in ('session_id','recording_id'):
        if data[name].shape!=(n,4): raise ValueError(name+' must be [images,4]')
    if len(set(np.asarray(data['recording_id']).astype(str).ravel())) != n*4:
        raise ValueError('recording_id must be unique across all images/repeats')
    train=set(splits['train'])
    for name in ('preprocessing_fit_classes','encoder_fit_classes','token_fit_classes'):
        fit=set(np.asarray(data[name]).astype(str).ravel())
        if not fit or not fit <= train:
            raise ValueError(name+': held-out categories or missing provenance')
    for name in ('encoder_sha256','preprocessing_sha256'):
        value=str(np.asarray(data[name]).item())
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value.lower()):
            raise ValueError(name+' must be an actual SHA256 string')
    for name in ('tokens','global_eeg','vision'):
        if not np.isfinite(data[name]).all(): raise ValueError('Non-finite '+name)
    for c in set(labels):
        if np.count_nonzero(labels==c)!=10:
            raise ValueError('Fine-grained pool requires ten real images per class')
    # Metadata declarations must also be checked against actual source/training code.
    return n

def candidate_rows(labels, query_indices, pool_indices, n=10, cross=False, seed=2026):
    """Deterministic shared candidate identities, with the true image always included."""
    labels=np.asarray(labels).astype(str)
    pool=np.asarray(pool_indices)
    if n<2: raise ValueError('Need at least two candidates')
    rng=np.random.default_rng(seed)
    by_class={}
    for item in pool:
        by_class.setdefault(labels[item],[]).append(item)
    by_class={c:np.asarray(items) for c,items in by_class.items()}
    classes=np.array(sorted(by_class))
    class_index={c:j for j,c in enumerate(classes)}
    rows=[]; targets=[]
    for i in query_indices:
        if cross:
            excluded=class_index[labels[i]]
            chosen=rng.choice(len(classes)-1,n-1,replace=False)
            chosen=chosen+(chosen>=excluded)
            distractors=[rng.choice(by_class[classes[j]]) for j in chosen]
        else:
            same=by_class[labels[i]]
            eligible=same[same!=i]
            distractors=rng.choice(eligible,n-1,replace=False)
        row=rng.permutation(np.r_[i,distractors]).astype(int)
        rows.append(row); targets.append(int(np.flatnonzero(row==i)[0]))
    return np.asarray(rows),np.asarray(targets)

def within_class_mismatch(labels, indices, seed):
    """Derangement within each class: never accidentally keep the true pairing."""
    labels=np.asarray(labels).astype(str); indices=np.asarray(indices)
    rng=np.random.default_rng(seed); out=indices.copy()
    for c in sorted(set(labels[indices])):
        pos=np.flatnonzero(labels[indices]==c)
        if len(pos)<2: raise ValueError('Mismatch requires at least two images/class')
        order=rng.permutation(pos)
        out[order]=indices[np.roll(order,1)]
    assert np.all(out!=indices)
    return out
