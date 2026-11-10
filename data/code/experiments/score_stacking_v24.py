"""Compare probability averaging with proper-score-fitted score stacking.

This decision-affecting stacker is fitted within each development-calibration
fold, not silently described as accuracy-preserving temperature scaling. All
two-branch controls receive the identical fit opportunity. No final EEG access.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import torch
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
from native_temporal_screen import ROOT,sha,save
from reliability_repair_v15 import load_scores
from metric_suite_v23 import evaluate_metrics
from compare_noise_controls import crossfit_components
from load_visual_features import load_features

def fit_stack(components,y,fit,objective):
    # Centering/scaling only use scores, with fit-side scales shared by all
    # candidates. Candidate position never receives a class-specific weight.
    centered=np.stack([s-s.mean(1,keepdims=True) for s in components],axis=-1)
    scale=np.std(centered[fit],axis=(0,1)).clip(1e-8);h=centered/scale
    x=h[fit];target=y[fit];n=len(target);r=np.arange(n)
    def value_gradient(w):
        z=np.einsum('nkd,d->nk',x,w);p=softmax(z,axis=1)
        if objective=='nll':
            loss=np.mean(logsumexp(z,axis=1)-z[r,target]);g=p.copy();g[r,target]-=1;g/=n
        elif objective=='brier':
            delta=p-np.eye(p.shape[1])[target]
            loss=np.mean(np.sum(delta**2,axis=1));g=2*p*(delta-np.sum(delta*p,axis=1,keepdims=True))/n
        else:raise ValueError(objective)
        # Fixed weak ridge, identical for every comparator.
        return float(loss+1e-5*np.sum(w*w)),np.einsum('nk,nkd->d',g,x)+2e-5*w
    opt=minimize(value_gradient,np.ones(len(components)),jac=True,method='L-BFGS-B',
        bounds=[(0,30)]*len(components),options={'ftol':1e-12,'gtol':1e-7,'maxiter':200})
    assert opt.success,opt.message
    return np.einsum('nkd,d->nk',h,opt.x),{'weights':opt.x.tolist(),'fit_scales':scale.tolist(),'objective':objective}

def main():
    torch.set_num_threads(1);out=ROOT/'results/score_stacking_v24';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Preserve completed stacker screen')
    save(out/'configuration.json',{'source_sha256':sha(__file__),'fit':'two complementary 50-concept development calibration folds',
        'objectives':['nll','brier'],'weights':'nonnegative, candidate-position shared; ridge1e-5',
        'boundary':'Stacking can change retrieval ranking; this is a new development comparator, not temperature-only calibration',
        'formal_calibration_or_test_accessed':False})
    visual,canonical=load_features('train');visual=torch.nn.functional.normalize(torch.tensor(visual,dtype=torch.float32),dim=-1).numpy().astype(np.float64)
    specs=[('NICE',s,seed) for s in ['sub-01','sub-02','sub-03'] for seed in [2026,2027]]+[('ATM',s,2026) for s in ['sub-01','sub-02','sub-03']]
    records={};provenance={}
    for arch,subject,seed in specs:
        key=f'{arch}_{subject}_seed{seed}';scores,y,fold,_,ids,cids,classes,prov=load_scores(arch,subject,seed,visual,canonical)
        local={m:scores[m] for m in ['isotropic','OOF_locked','repeat_difference_proxy']}
        if arch=='NICE' and seed==2026:
            # Reconstruct the raw MLP score. The same frozen raw-input global
            # path is used as every mixture; do not fit stacks to OOF logits.
            p=np.load(ROOT/'results/raw_joint_core_v2'/subject/'seed2026/mlp/within10_validation.npz')
            assert np.array_equal(p['targets'],y)
            identity=np.load(ROOT/'processed'/subject/'train_identities.npz')['image_id']
            assert np.array_equal(identity[p['query_indices']],ids) and np.array_equal(identity[p['candidate_rows']],cids)
            local['MLP']=p['logits']
        records[key]={};provenance[key]=prov
        for branch,score in local.items():
            comp=[scores['global'],score]
            z,params=crossfit_components(comp,y,fold);v,_=evaluate_metrics(z,y)
            records[key][branch]={'equal_probability':{'metrics':v,'parameters':params}}
            for objective in ['nll','brier']:
                prediction=np.empty_like(z);pars=[]
                for f in [0,1]:
                    fit=fold!=f;test=~fit;assert set(classes[fit]).isdisjoint(classes[test])
                    zz,p=fit_stack(comp,y,fit,objective);prediction[test]=zz[test];pars.append(p)
                v,_=evaluate_metrics(prediction,y);records[key][branch][objective]={'metrics':v,'parameters':pars}
                np.savez_compressed(out/(key+'_'+branch+'_'+objective+'.npz'),logits=prediction,targets=y,query_ids=ids,candidate_ids=cids,concepts=classes,fold=fold)
        print('STACK_CHECK',key,{b:{o:round(v['metrics']['top1'],4) for o,v in rr.items()} for b,rr in records[key].items()},flush=True)
    aggregate={};participant={}
    for arch,seeds in [('NICE',[2026,2027]),('ATM',[2026])]:
        participant[arch]={s:{b:{o:{k:float(np.mean([records[f'{arch}_{s}_seed{seed}'][b][o]['metrics'][k] for seed in seeds]))
            if records[f'{arch}_{s}_seed{seeds[0]}'][b][o]['metrics'][k] is not None else None
            for k in records[f'{arch}_{s}_seed{seeds[0]}'][b][o]['metrics']}
            for o in ['equal_probability','nll','brier']} for b in ['isotropic','OOF_locked','repeat_difference_proxy']} for s in ['sub-01','sub-02','sub-03']}
        aggregate[arch]={b:{o:{k:float(np.mean([participant[arch][s][b][o][k] for s in participant[arch]])) if participant[arch]['sub-01'][b][o][k] is not None else None
            for k in participant[arch]['sub-01'][b][o]} for o in ['equal_probability','nll','brier']} for b in ['isotropic','OOF_locked','repeat_difference_proxy']}
    save(out/'summary.json',{'source_sha256':sha(__file__),'records':records,'aggregate':aggregate,'participant_seed_means':participant,
        'provenance':provenance,'formal_calibration_or_test_accessed':False,'scope':'Development-only stacker with matched calibration fit; encoders were validation selected',
        'metrics_source_sha256':sha(Path(__file__).with_name('metric_suite_v23.py'))})
    print(json.dumps({a:aggregate[a]['repeat_difference_proxy'] for a in aggregate},indent=2))

if __name__=='__main__':main()
