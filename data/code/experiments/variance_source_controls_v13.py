"""Separate residual weighting from within-image variation and ordinary fit.

Training concepts alone determine each variance. Four methods receive the
same fixed half mixture and complementary-validation temperature fitting.
Repeated-view differences are a variation proxy, not pure independent noise.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import json
import numpy as np
import torch
from native_temporal_screen import Banks, ROOT, sha, save
from protocol import candidate_rows
from compare_noise_controls import crossfit_components
from reliability_screen import evaluate

def shrunk_variance(x):
    centered = x - x.mean(0)
    variance = np.mean(centered**2, axis=0)
    return .9*variance + .1*variance.mean()

def main():
    torch.set_num_threads(1)
    out = ROOT / 'results/variance_source_controls_v13'
    out.mkdir(exist_ok=True)
    target = out / 'summary.json'
    if target.exists():
        raise FileExistsError('Preserve the completed variance-source control')
    records = {}; provenance = {}
    for subject in ['sub-01', 'sub-02', 'sub-03']:
        data = Banks(subject)
        modelpath = ROOT / 'results/cross_trial_likelihood_v6' / subject / 'model.npz'
        model = np.load(modelpath)
        assert set(model['fit_classes']) == set(data.splits['train'])
        emb = np.asarray(data.emb[:, :2], dtype=np.float64)
        x = ((emb.reshape(-1, 1024)-model['neural_center']) @ model['neural_basis'] / model['neural_scale']).reshape(-1, 2, 64)
        mu = (data.vision.numpy().astype(np.float64) @ model['visual_to_neural'] + model['offset'] + model['residual_mean']).reshape(-1, 2, 64)
        pooled = x.mean(1); predicted = mu.mean(1)
        c = model['residual_covariance']; d = 64
        cm = (c[:d,:d]+c[d:,d:]+c[:d,d:]+c[d:,:d])/4
        # Shrinkage already entered joint covariance: preserve the locked
        # diagonal treatment exactly, rather than shrink the pooled score again.
        variances = {'isotropic': np.ones(64), 'OOF_residual': np.diag(cm),
                     'in_sample_residual': shrunk_variance((pooled-predicted)[data.train]),
                     'repeat_difference_proxy': shrunk_variance(((x[:,0]-x[:,1])/2)[data.train])}
        q = data.val; rows, y = candidate_rows(data.labels, q, q, seed=20261103)
        raw = np.einsum('bd,bkd->bk', np.asarray(data.emb[q,2]), data.vision.numpy()[rows])
        delta = pooled[q,None,:]-predicted[rows]
        fold = np.array([0 if data.labels[j] in set(data.splits['validation'][:50]) else 1 for j in q])
        records[subject] = {}
        for name, variance in variances.items():
            assert np.all(variance > 0)
            local = -.5*np.sum(delta**2/variance, axis=-1)
            z, params = crossfit_components([raw, local], y, fold)
            records[subject][name] = {'metrics': evaluate(z,y), 'parameters': params,
                'train_variance': variance.tolist()}
            np.savez_compressed(out/(subject+'_'+name+'.npz'), logits=z, targets=y,
                                query_ids=data.ids[q], candidate_ids=data.ids[rows])
        prior = np.load(ROOT/'results/noise_control_comparison_v1'/(subject+'_raw4_plus_diagonal_noise.npz'))
        # The small FP32 vs FP64 transformation difference must not change
        # the candidate ranking of this exact replay.
        assert np.array_equal(np.load(out/(subject+'_OOF_residual.npz'))['logits'].argmax(1), prior['logits'].argmax(1))
        provenance[subject] = {'mapping_sha256':sha(modelpath), 'bank': data.manifest}
        print('VARIANCE_CONTROL', subject, {n:r['metrics']['top1'] for n,r in records[subject].items()}, flush=True)
    aggregate = {m:{k:float(np.mean([records[s][m]['metrics'][k] for s in records]))
                   for k in records['sub-01'][m]['metrics']} for m in records['sub-01']}
    save(target, {'source_sha256':sha(__file__), 'scope':'Three participants, seed2026, validation-only variance attribution',
        'records':records, 'aggregate':aggregate, 'provenance':provenance,
        'formal_calibration_or_test_accessed':False,
        'boundary':'Repeated-view difference covariance is a variation proxy with session structure; residuals also include mapping error. No unique OOF-mechanism claim without comparison.'})
    print(json.dumps(aggregate,indent=2))

if __name__ == '__main__':
    main()
