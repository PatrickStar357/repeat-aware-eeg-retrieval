"""Fair calibration and strong equal-mixture controls for the noise redesign.

All mixture weights are exactly 1/2, never selected on validation outcomes.
Each outer concept fold fits component temperatures and the final mixture
temperature on its complementary concepts. Checkpoint-selection bias remains:
these are development diagnostics, not held-out final performance estimates.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.special import logsumexp
from native_temporal_screen import Banks,sha,save
from cross_trial_likelihood_v6 import quadratic
from reliability_screen import temperature,evaluate
from protocol import candidate_rows

def logprob(x):return x-logsumexp(x,axis=-1,keepdims=True)

def crossfit_components(components,y,fold):
    z=np.empty_like(components[0],dtype=float);parameters=[]
    for f in [0,1]:
        ci=fold!=f;ti=~ci
        ts=[temperature(s[ci],y[ci]) for s in components]
        if len(components)==1:
            raw=components[0]/ts[0];final=1.
        else:
            lp=[logprob(s/t) for s,t in zip(components,ts)]
            raw=logsumexp(np.stack(lp),axis=0)-np.log(len(components))
            final=temperature(raw[ci],y[ci])
        z[ti]=raw[ti]/final;parameters.append({'component_temperatures':ts,'final_temperature':final})
    return z,parameters

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',default=['sub-01','sub-02','sub-03']);a=p.parse_args()
    records={};allpred={}
    for subject in a.subjects:
        data=Banks(subject);r=data.base.parent.parent
        fit=r/'cross_trial_likelihood_v6'/subject;model=dict(np.load(fit/'model.npz'))
        q=data.val;rows,y=candidate_rows(data.labels,q,q,seed=20261103)
        x=((np.array(data.emb[:,:2]).reshape(-1,1024)-model['neural_center'])@model['neural_basis']/model['neural_scale']).reshape(-1,128)
        mu=data.vision.numpy()@model['visual_to_neural']+model['offset']+model['residual_mean'];d=64;c=model['residual_covariance']
        cm=(c[:d,:d]+c[d:,d:]+c[:d,d:]+c[d:,:d])/4
        m=(x[q,:d]+x[q,d:])/2;candidate=(mu[rows][:,:,:d]+mu[rows][:,:,d:])/2
        qs=quadratic(m,candidate,np.eye(d));qd=quadratic(m,candidate,np.diag(1/np.diag(cm)))
        sources={}
        for name,file in [('raw4',fit/'within10_global.npz'),('noise',fit/'within10_pooled_gaussian.npz'),
                          ('joint',fit/'within10_joint_gaussian.npz')]:
            v=dict(np.load(file));assert np.array_equal(v['targets'],y)
            assert np.array_equal(v['query_ids'],data.ids[q]) and np.array_equal(v['candidate_ids'],data.ids[rows])
            sources[name]=v['logits']
        sources['isotropic']=-.5*qs;sources['diagonal_noise']=-.5*qd
        ev=np.array(data.emb[q,:2]);v=data.vision.numpy()[rows]
        ez=ev.mean(1);ez/=np.linalg.norm(ez,axis=1,keepdims=True)
        sources['two_group_mean']=np.einsum('bd,bkd->bk',ez,v)
        for name,mode,key in [('embed4','cross','global_logits'),('MLP','mlp','logits'),('two_view','two_view','logits')]:
            pp=dict(np.load(r/'raw_joint_core_v2'/subject/'seed2026'/mode/'within10_validation.npz'))
            original=data.original[q]
            assert np.array_equal(pp['query_indices'],original) and np.array_equal(pp['candidate_rows'],data.original[rows])
            assert np.array_equal(pp['targets'],y);sources[name]=pp[key]
        first=set(data.splits['validation'][:50]);fold=np.array([0 if data.labels[j] in first else 1 for j in q])
        methods={name:[s] for name,s in sources.items()}
        for name in sources:
            if name!='raw4':methods['raw4+'+name]=[sources['raw4'],sources[name]]
        records[subject]={};allpred[subject]={}
        for name,comp in methods.items():
            z,params=crossfit_components(comp,y,fold)
            records[subject][name]={'metrics':evaluate(z,y),'parameters':params}
            allpred[subject][name]=(z,y,data.ids[q],data.ids[rows])
    out=r/'noise_control_comparison_v1';out.mkdir(exist_ok=True)
    for subject,models in allpred.items():
        for name,(z,y,ids,candidates) in models.items():
            np.savez_compressed(out/(subject+'_'+name.replace('+','_plus_')+'.npz'),logits=z,targets=y,query_ids=ids,candidate_ids=candidates)
    names=list(records[a.subjects[0]])
    aggregate={name:{k:float(np.mean([records[s][name]['metrics'][k] for s in a.subjects])) for k in records[a.subjects[0]][name]['metrics']} for name in names}
    report={'scope':'validation-only diagnostic; no final/calibration access; base checkpoints validation-selected',
        'source_sha256':sha(__file__),'primary':'raw4+noise','subjects':a.subjects,
        'fixed_mixture_weight':.5,'temperature_policy':'component + final temperatures fit on complementary validation concepts',
        'records':records,'aggregate':aggregate}
    target=out/('comparison_'+str(len(a.subjects))+'subjects.json');save(target,report)
    print(json.dumps({n:aggregate[n] for n in ['raw4','MLP','two_view','raw4+embed4','raw4+MLP','raw4+two_view','raw4+isotropic','raw4+diagonal_noise','raw4+noise']},indent=2))

if __name__=='__main__':main()
