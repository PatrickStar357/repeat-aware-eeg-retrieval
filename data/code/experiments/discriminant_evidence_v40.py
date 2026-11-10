"""Candidate-specific signed evidence, normalized by repeated-trial variation.

Revised advancement gate emphasizes accuracy, ranking and proper scores, not
all-metric dominance. Dependencies are estimated in joint NICE/ATM coordinates.
Readout diagnostics use concept-cross-fitted templates on training concepts;
frozen encoders have seen these concepts, so this is not unseen-category testing.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from scipy.special import ndtr
from native_temporal_screen import ROOT,sha,save
from protocol import category_split,candidate_rows
from load_visual_features import load_features
from cross_trial_likelihood_v6 import ridge
from score_stacking_v24 import fit_stack
from metric_suite_v23 import evaluate_metrics
from variance_source_controls_v13 import shrunk_variance

SUBJECTS=['sub-01','sub-02','sub-03']
VARIANTS=['strong_base','joint_distance','pair_diagonal','pair_block','pair_joint']

def covariance(delta,kind):
    z=delta-delta.mean(0);emp=z.T@z/len(z);diagonal=np.diag(np.diag(emp));d=len(emp)
    if kind=='diagonal':return .9*diagonal+.1*np.trace(emp)/d*np.eye(d)
    if kind=='block':emp[:d//2,d//2:]=0;emp[d//2:,:d//2]=0
    if kind not in ['joint','block']:raise ValueError(kind)
    # Correlation shrinkage preserves heterogeneous variances while ensuring
    # invertibility. Same fixed .1 shrinkage, no validation-selected strength.
    return .9*emp+.1*diagonal

def pair_support(x,mu,cov):
    """Average marginal probit support over rivals, never multiply as independent.

    Supports are scores rather than calibrated probabilities of image identity.
    Signed contrasts allow contradiction. Covariance accounts for correlated
    latent coordinates and cross-backbone measurement variation.
    """
    precision=np.linalg.inv(cov);n,k,d=mu.shape
    centered=mu-mu.mean(1,keepdims=True)
    # Candidate centering changes neither margins nor pair distances.
    observation=x-mu.mean(1);a=centered@precision
    scores=np.sum(observation[:,None,:]*a,axis=-1)-.5*np.sum(centered*a,axis=-1)
    margin=scores[:,:,None]-scores[:,None,:]
    norms=np.sum(centered*a,axis=-1)
    distance2=np.maximum(norms[:,:,None]+norms[:,None,:]-2*np.einsum('nkd,nrd->nkr',a,centered),0)
    t=margin/np.sqrt(distance2.clip(1e-10))
    probability=ndtr(t);diag=np.arange(k);probability[:,diag,diag]=0
    mean=probability.sum(-1)/(k-1);mean=np.clip(mean,1e-6,1-1e-6)
    support=np.log(mean)-np.log1p(-mean)
    return support,scores,t

def load_geometry(subject,arch,visual,canonical):
    ident=np.load(ROOT/'processed'/subject/'train_identities.npz');labels=ident['class_id'].astype(str);split=category_split(labels)
    original=np.flatnonzero(np.isin(labels,np.r_[split['train'],split['validation']]))
    ids=ident['image_id'][original];labels=labels[original];assert np.array_equal(ids,canonical[original])
    tr=np.flatnonzero(np.isin(labels,split['train']));q=np.flatnonzero(np.isin(labels,split['validation']))
    if arch=='NICE':mp=ROOT/'results/cross_trial_likelihood_v6'/subject/'model.npz';bp=ROOT/'results/native_temporal_screen_v3'/subject/'banks/embeddings.npy'
    else:
        b=ROOT/'results/atm_noise_control_v9'/subject/'seed2026';mp=b/'statistics/model.npz';bp=b/'banks/embeddings.npy'
    m=np.load(mp);assert set(m['fit_classes'])==set(split['train'])
    emb=np.load(bp,mmap_mode='r');v=visual[original]
    x=((np.asarray(emb[:,:2],dtype=float).reshape(-1,1024)-m['neural_center'])@m['neural_basis']/m['neural_scale']).reshape(-1,2,64)
    mu=(v@m['visual_to_neural']+m['offset']+m['residual_mean']).reshape(-1,2,64).mean(1)
    # True out-of-fold template predictions on training concepts for diagnostics.
    oof=np.empty((len(tr),64));classes=np.random.default_rng(2026).permutation(split['train']);which={c:i%5 for i,c in enumerate(classes)}
    fold=np.array([which[c] for c in labels[tr]])
    vt=torch.tensor(v[tr],dtype=torch.float64,device='cuda');xt=torch.tensor(x[tr].mean(1),dtype=torch.float64,device='cuda')
    for f in range(5):
        fit=fold!=f;test=~fit;w,b,_=ridge(vt[fit],xt[fit]);oof[test]=(vt[test]@w+b).cpu().numpy()
    del vt,xt;torch.cuda.empty_cache()
    # Keep OOF data centered by a training-derived constant only, shared by all candidates.
    oof+=m['residual_mean'].reshape(2,64).mean(0)
    return {'x':x.mean(1),'difference':(x[:,0]-x[:,1])/2,'mu':mu,'oof_mu':oof,'raw4':np.asarray(emb[:,2],dtype=float),
        'tr':tr,'q':q,'ids':ids,'labels':labels,'split':split,'v':v,'sources':{'mapping_sha256':sha(mp),'bank_manifest_sha256':sha(bp.with_name('manifest.json'))}}

def run_partition(gs,partition):
    a=gs['NICE'];indices=a['tr'] if partition=='train_diagnostic' else a['q'];labels=a['labels'];rows,y=candidate_rows(labels,indices,indices,seed=20261103)
    base=[];sources={};full_mu=[];full_x=[];full_delta=[]
    for arch in ['NICE','ATM']:
        g=gs[arch];assert np.array_equal(g['ids'],a['ids']);mu=g['mu'].copy()
        if partition=='train_diagnostic':mu[g['tr']]=g['oof_mu']
        variance=shrunk_variance(g['difference'][g['tr']])
        global_score=np.einsum('nd,nkd->nk',g['raw4'][indices],g['v'][rows])
        local=-.5*np.sum((g['x'][indices,None,:]-mu[rows])**2/variance,axis=-1)
        sources[arch]=(global_score,local)
        full_mu.append(mu[rows]);full_x.append(g['x'][indices]);full_delta.append(g['difference'][g['tr']])
    base=[sources['NICE'][0],sources['ATM'][0],sources['NICE'][1],sources['ATM'][1]]
    x=np.concatenate(full_x,-1);mu=np.concatenate(full_mu,-1);delta=np.concatenate(full_delta,-1)
    supports={};diagnostics={}
    for kind in ['diagonal','block','joint']:
        cov=covariance(delta,kind);s,d,t=pair_support(x,mu,cov);supports['pair_'+kind]=s
        if kind=='joint':supports['joint_distance']=d
        diagnostics[kind]={'minimum_eigenvalue':float(np.linalg.eigvalsh(cov).min()),
            'signed_support_min':float(s.min()),'signed_support_max':float(s.max())}
    if partition=='train_diagnostic':fold=np.array([int(hashlib.sha256(c.encode()).hexdigest(),16)%2 for c in labels[indices]])
    else:fold=np.array([0 if c in set(a['split']['validation'][:50]) else 1 for c in labels[indices]])
    result={};predictions={};params={}
    for kfold in ([2] if partition=='train_diagnostic' else [2,5]):
        ff=fold if kfold==2 else np.searchsorted(np.unique(labels[indices]),labels[indices])//20
        result[str(kfold)]={};predictions[str(kfold)]={};params[str(kfold)]={}
        for variant in VARIANTS:
            comp=base if variant=='strong_base' else base+[supports[variant]];z=np.empty_like(base[0]);pars=[]
            for f in range(kfold):
                fit=ff!=f;test=~fit;assert set(labels[indices][fit]).isdisjoint(labels[indices][test])
                zz,p=fit_stack(comp,y,fit,'brier');z[test]=zz[test];pars.append(p)
            result[str(kfold)][variant]=evaluate_metrics(z,y)[0];params[str(kfold)][variant]=pars;predictions[str(kfold)][variant]=z
    return result,predictions,params,{'targets':y,'query_ids':a['ids'][indices],'candidate_ids':a['ids'][rows],'concepts':labels[indices]},diagnostics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',choices=SUBJECTS,required=True);p.add_argument('--development-confirmation',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);out=ROOT/'results/discriminant_evidence_v40'/args.subject
    stage='development_confirmation' if args.development_confirmation else 'train_diagnostic';out=out/stage
    if out.exists() and list(out.iterdir()):raise FileExistsError('Preserve completed or partial fixed study')
    out.mkdir(parents=True,exist_ok=True);guard=ROOT/'results/raw_joint_core_v2/protocol_lock.json';assert json.loads(guard.read_text())['allow_final_evaluation'] is False
    source=sha(__file__);save(out/'configuration.json',{'source_sha256':source,'primary':'pair_joint','comparators':VARIANTS,
        'partition':stage,'covariance_shrinkage':.1,'pair_aggregation':'mean marginal probit supports; shared covariance, not independent likelihoods',
        'revised_gate':'primary Top1 and mAP improve, NLL and Brier improve; majority-participant support; ECE/wrong confidence descriptive, not mandatory wins; fivefold sensitivity and matched mechanism controls',
        'data_boundary':'train diagnostics use OOF image templates but EEG encoders saw all training concepts; development checkpoint-selection bias remains',
        'final_data_used':False,'guard_sha256':sha(guard)})
    begin=time.time();v,canonical=load_features('train');v=torch.nn.functional.normalize(torch.tensor(v,dtype=torch.float32),dim=-1).numpy().astype(float)
    gs={arch:load_geometry(args.subject,arch,v,canonical) for arch in ['NICE','ATM']}
    result,predictions,params,identity,diagnostics=run_partition(gs,stage if stage=='train_diagnostic' else 'validation')
    for fold in predictions:
        for name,z in predictions[fold].items():
            if stage!='train_diagnostic':
                ref=np.load(ROOT/'results/backbone_fusion_v29'/f'{args.subject}_fold{fold}_repeat4_brier.npz')
                assert all(np.array_equal(ref[k],identity[k]) for k in ['targets','query_ids','candidate_ids'])
                if name=='strong_base':assert np.max(np.abs(z-ref['logits']))<1e-6
            np.savez_compressed(out/f'fold{fold}_{name}.npz',logits=z,**identity)
        print('DISCRIMINANT',args.subject,stage,fold,{n:{k:round(m[k],6) for k in ['top1','map','brier','nll','ece15']} for n,m in result[fold].items()},flush=True)
    assert sha(__file__)==source
    save(out/'summary.json',{'source_sha256':source,'results':result,'parameters':params,'diagnostics':diagnostics,
        'sources':{arch:g['sources'] for arch,g in gs.items()},'seconds':time.time()-begin,'final_data_used':False})

if __name__=='__main__':main()
