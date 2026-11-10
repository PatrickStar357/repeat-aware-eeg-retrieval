"""Bounded calibration diagnosis and repetition-aware confidence controls.

Fit all calibration parameters on complementary validation concepts. Rebuild
raw component scores: never recalibrate concatenated out-of-fold logits whose
training-side scores may depend on evaluation-side labels. Neural checkpoint
selection still used the whole validation partition: development evidence only.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from native_temporal_screen import ROOT,sha,save
from protocol import category_split,candidate_rows
from load_visual_features import load_features
from reliability_screen import temperature,evaluate
from variance_source_controls_v13 import shrunk_variance

SPECS=[('NICE',s,seed) for s in ['sub-01','sub-02','sub-03'] for seed in [2026,2027]]
SPECS += [('ATM',s,2026) for s in ['sub-01','sub-02','sub-03']]
METHODS=['global','isotropic','OOF_locked','repeat_difference_proxy']
CALIBRATORS=['temperature','variance_nll','shuffled_variance_nll','confidence_nll','joint_scalar','joint_variance']

def objective_gradient(ab,x,y,f,joint=False):
    correct=(x.argmax(1)==y).astype(float);r=np.arange(len(y))
    a,b=ab;zz=x*np.exp(-a-b*f)[:,None];p=softmax(zz,axis=1)
    ez=np.sum(p*zz,axis=1);nll=logsumexp(zz,axis=1)-zz[r,y]
    g=zz[r,y]-ez
    if joint:
        c=p.max(1).clip(1e-12,1-1e-12)
        binary=-correct*np.log(c)-(1-correct)*np.log1p(-c)
        topg=(c-correct)*(ez-zz.max(1))/(1-c)
        loss=.5*nll.mean()+.5*binary.mean();g=.5*g+.5*topg
    else:loss=nll.mean()
    return float(loss+.01*b*b),np.array([g.mean(),np.mean(g*f)+.02*b])

def relative_fit(z,y,feature=None,joint=False):
    """Positive row temperature; candidate ranking is preserved exactly."""
    x=np.asarray(z,dtype=float);x=x-x.max(1,keepdims=True)
    f=np.zeros(len(x)) if feature is None else np.asarray(feature,dtype=float)
    def objective(ab):return objective_gradient(ab,x,y,f,joint)
    bounds=[(-np.log(10),np.log(10)),(0.,3.) if feature is not None else (0.,0.)]
    opt=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',bounds=bounds,
                 options={'maxiter':100,'ftol':1e-12,'gtol':1e-7})
    assert opt.success or np.linalg.norm(opt.jac)<1e-5,(opt.message,opt.jac)
    return {'intercept':float(opt.x[0]),'slope':float(opt.x[1]),'objective':float(opt.fun)}

def apply_relative(z,feature,pars):
    f=np.zeros(len(z)) if feature is None else feature
    return np.asarray(z)*np.exp(-pars['intercept']-pars['slope']*f)[:,None]

def extra_metrics(z,y):
    r=evaluate(z,y);p=softmax(z,axis=1);c=p.max(1);ok=p.argmax(1)==y
    for bins in [5,10,20]:
        b=np.minimum((c*bins).astype(int),bins-1)
        r['ece'+str(bins)]=float(sum(np.mean(b==j)*abs(c[b==j].mean()-ok[b==j].mean()) for j in range(bins) if (b==j).any()))
    order=np.argsort(c,kind='stable')
    r['equal_mass_ece10']=float(sum(len(ix)/len(c)*abs(c[ix].mean()-ok[ix].mean()) for ix in np.array_split(order,10)))
    r['signed_confidence_gap']=float(c.mean()-ok.mean())
    for threshold in [.5,.7,.9]:
        r['wrong_and_confident_'+str(threshold)]=float(((~ok)&(c>=threshold)).mean())
    return r

def matched_errors(base,new,y):
    pb=softmax(base,axis=1);pn=softmax(new,axis=1);cb=pb.max(1);cn=pn.max(1)
    ob=pb.argmax(1)==y;on=pn.argmax(1)==y;out={}
    for name,mask in [('both_wrong',~ob&~on),('corrected',~ob&on),('regressed',ob&~on),('both_correct',ob&on)]:
        out[name]={'n':int(mask.sum()),'base_confidence':float(cb[mask].mean()) if mask.any() else None,
                   'new_confidence':float(cn[mask].mean()) if mask.any() else None}
    return out

def bins(z,y):
    p=softmax(z,axis=1);c=p.max(1);correct=p.argmax(1)==y
    groups=np.minimum((c*15).astype(int),14);result=[]
    for j in range(15):
        ix=groups==j
        if ix.any():result.append({'left':j/15,'n':int(ix.sum()),'mean_confidence':float(c[ix].mean()),'accuracy':float(correct[ix].mean())})
    return result

def load_scores(arch,subject,seed,visual,canonical):
    ident=np.load(ROOT/'processed'/subject/'train_identities.npz');labels=ident['class_id'].astype(str)
    split=category_split(labels);original=np.flatnonzero(np.isin(labels,np.r_[split['train'],split['validation']]))
    labels=labels[original];ids=ident['image_id'][original];assert np.array_equal(ids,canonical[original])
    tr=np.flatnonzero(np.isin(labels,split['train']));q=np.flatnonzero(np.isin(labels,split['validation']))
    if arch=='ATM':
        base=ROOT/'results/atm_noise_control_v9'/subject/f'seed{seed}';mp=base/'statistics/model.npz';bp=base/'banks/embeddings.npy'
    elif seed==2027:
        base=ROOT/'results/noise_seed_replication_v8'/subject/f'seed{seed}';mp=base/'statistics/model.npz';bp=base/'banks/embeddings.npy'
    else:
        mp=ROOT/'results/cross_trial_likelihood_v6'/subject/'model.npz';bp=ROOT/'results/native_temporal_screen_v3'/subject/'banks/embeddings.npy'
    model=np.load(mp);assert set(model['fit_classes'])==set(split['train'])
    emb=np.load(bp,mmap_mode='r');v=visual[original]
    x=((np.asarray(emb[:,:2],dtype=np.float64).reshape(-1,1024)-model['neural_center'])@model['neural_basis']/model['neural_scale']).reshape(-1,2,64)
    mu=(v@model['visual_to_neural']+model['offset']+model['residual_mean']).reshape(-1,2,64)
    pooled=x.mean(1);pred=mu.mean(1);c=model['residual_covariance'];d=64
    cm=(c[:d,:d]+c[d:,d:]+c[:d,d:]+c[d:,:d])/4
    delta=x[:,0]-x[:,1];variation_variance=shrunk_variance(delta[tr]);dc=delta-delta[tr].mean(0)
    observed=np.log1p(np.mean(dc**2/variation_variance,axis=1))
    feature=np.clip((observed-observed[tr].mean())/observed[tr].std(),-5,5)[q]
    rows,y=candidate_rows(labels,q,q,seed=20261103)
    raw=np.einsum('bd,bkd->bk',np.array(emb[q,2]),v[rows]);dif=pooled[q,None,:]-pred[rows]
    variances={'isotropic':np.ones(d),'OOF_locked':np.diag(cm),
               'repeat_difference_proxy':shrunk_variance(((x[:,0]-x[:,1])/2)[tr])}
    scores={'global':raw}
    for name,var in variances.items():scores[name]=-.5*np.sum(dif**2/var,axis=-1)
    first=set(split['validation'][:50]);fold=np.array([0 if labels[j] in first else 1 for j in q])
    provenance={'mapping_sha256':sha(mp),'bank_manifest_sha256':sha(bp.with_name('manifest.json')),
        'feature_fit_classes':split['train'].tolist()}
    return scores,y,fold,feature,ids[q],ids[rows],labels[q],provenance

def base_score(components,y,fit):
    ts=[temperature(s[fit],y[fit]) for s in components]
    if len(components)==1:return components[0]/ts[0],{'component_temperatures':ts,'final_temperature':1.}
    lp=[s/t-logsumexp(s/t,axis=1,keepdims=True) for s,t in zip(components,ts)]
    z=logsumexp(np.stack(lp),axis=0)-np.log(len(lp));t=temperature(z[fit],y[fit])
    return z/t,{'component_temperatures':ts,'final_temperature':t}

def main():
    torch.set_num_threads(1);out=ROOT/'results/reliability_repair_v15';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Preserve completed calibration study')
    source=sha(__file__)
    save(out/'configuration.json',{'source_sha256':source,'methods':METHODS,'calibrators':CALIBRATORS,
        'outer_folds':'first50 vs second50 validation concepts','slope_bounds':[0,3],'log_relative_temperature_bounds':[-np.log(10),np.log(10)],
        'slope_L2':.01,'joint_objective':'0.5 multiclass NLL + 0.5 top-correctness binary NLL',
        'new_EEG_training':False,'formal_calibration_or_test_accessed':False})
    visual,canonical=load_features('train');visual=torch.nn.functional.normalize(torch.tensor(visual,dtype=torch.float32),dim=-1).numpy().astype(np.float64)
    records={};provenance={};diagnosis={}
    for arch,subject,seed in SPECS:
        key=f'{arch}_{subject}_seed{seed}';scores,y,fold,variation,ids,cids,classes,prov=load_scores(arch,subject,seed,visual,canonical)
        provenance[key]=prov;records[key]={};predictions={}
        for method in METHODS:
            components=[scores['global']] if method=='global' else [scores['global'],scores[method]]
            preds={cal:np.empty_like(scores['global']) for cal in CALIBRATORS};parameters=[]
            for f in [0,1]:
                fit=fold!=f;test=~fit;assert set(classes[fit]).isdisjoint(classes[test])
                z,pars=base_score(components,y,fit);preds['temperature'][test]=z[test]
                conf=softmax(z,axis=1).max(1).clip(1e-6,1-1e-6)
                cf=-np.log(conf/(1-conf));cf=np.clip((cf-cf[fit].mean())/max(cf[fit].std(),1e-8),-5,5)
                rng=np.random.default_rng(20261103+f);shuffled=np.empty_like(variation)
                shuffled[fit]=rng.permutation(variation[fit]);shuffled[test]=rng.permutation(variation[test])
                settings={'variance_nll':(variation,False),'shuffled_variance_nll':(shuffled,False),
                          'confidence_nll':(cf,False),'joint_scalar':(None,True),'joint_variance':(variation,True)}
                fitted={}
                for cal,(feature,joint) in settings.items():
                    rp=relative_fit(z[fit],y[fit],None if feature is None else feature[fit],joint)
                    zz=apply_relative(z[test],None if feature is None else feature[test],rp)
                    assert np.array_equal(zz.argmax(1),z[test].argmax(1))
                    preds[cal][test]=zz;fitted[cal]=rp
                parameters.append({'evaluation_fold':f,'base':pars,'relative':fitted})
            records[key][method]={}
            for cal,z in preds.items():
                records[key][method][cal]={'metrics':extra_metrics(z,y),'reliability_bins':bins(z,y)}
                np.savez_compressed(out/(key+'_'+method+'_'+cal+'.npz'),logits=z,targets=y,
                    query_ids=ids,candidate_ids=cids,concepts=classes,fold=fold,variation_feature=variation)
                predictions[(method,cal)]=z
            reference=np.load(ROOT/'results/variance_source_comparison_v14'/(key+'_'+method+'.npz'))
            assert np.array_equal(reference['candidate_ids'],cids) and np.array_equal(reference['targets'],y)
            assert np.max(np.abs(reference['logits']-preds['temperature']))<1e-4
            assert np.array_equal(reference['logits'].argmax(1),preds['temperature'].argmax(1))
            records[key][method]['fit_parameters']=parameters
        diagnosis[key]={'global_vs_repeat_errors':matched_errors(predictions[('global','temperature')],predictions[('repeat_difference_proxy','temperature')],y),
            'variance_vs_joint_scalar_errors':matched_errors(predictions[('repeat_difference_proxy','joint_scalar')],predictions[('repeat_difference_proxy','joint_variance')],y)}
        print('CALIBRATION_CHECK',key,{c:round(records[key]['repeat_difference_proxy'][c]['metrics']['ece15'],5) for c in CALIBRATORS},flush=True)
    pm={};avg={}
    for arch,seeds in [('NICE',[2026,2027]),('ATM',[2026])]:
        pm[arch]={s:{m:{cal:{k:float(np.mean([records[f'{arch}_{s}_seed{seed}'][m][cal]['metrics'][k] for seed in seeds]))
                           for k in records[f'{arch}_{s}_seed{seeds[0]}'][m][cal]['metrics']} for cal in CALIBRATORS} for m in METHODS}
                  for s in ['sub-01','sub-02','sub-03']}
        avg[arch]={m:{cal:{k:float(np.mean([pm[arch][s][m][cal][k] for s in pm[arch]]))
                          for k in pm[arch]['sub-01'][m][cal]} for cal in CALIBRATORS} for m in METHODS}
    save(out/'summary.json',{'source_sha256':source,'records':records,'participant_seed_means':pm,'aggregate':avg,
        'diagnosis':diagnosis,'provenance':provenance,'formal_calibration_or_test_accessed':False,
        'scope':'Three development participants; neural checkpoint selection already used all validation concepts; calibration parameters are outer-concept separated',
        'support_source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['reliability_screen.py','protocol.py','variance_source_controls_v13.py']}})
    print(json.dumps({a:{c:avg[a]['repeat_difference_proxy'][c] for c in CALIBRATORS} for a in avg},indent=2))

if __name__=='__main__':main()
