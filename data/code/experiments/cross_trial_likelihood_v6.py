"""Bounded redesign: learned visual-to-neural hypotheses and dependent evidence.

Fit only training concepts. A and B residual covariance is estimated jointly;
Student-t and Gaussian models are compared with equal-pooling and independent
fusion. Student-t changes confidence, not the ordering of a fixed quadratic.
"""
from pathlib import Path
import argparse,json,time,math
import numpy as np
import torch
from scipy.special import gammaln
from native_temporal_screen import Banks,sha,save
from protocol import candidate_rows,within_class_mismatch
from reliability_screen import temperature,evaluate

DIM=64

def ridge(v,x):
    vm=v.mean(0);xm=x.mean(0);a=v-vm;b=x-xm
    cov=a.T@a/len(a);regularizer=.1*torch.trace(cov)/cov.shape[0]
    w=torch.linalg.solve(cov+regularizer*torch.eye(cov.shape[0],dtype=cov.dtype,device=cov.device),a.T@b/len(a))
    return w,xm-vm@w,float(regularizer)

def quadratic(x,mu,precision):
    delta=x[:,None,:]-mu
    return np.maximum(0,np.einsum('bki,ij,bkj->bk',delta,precision,delta,optimize=True))

def student_log_from_q(q,p,nu=5.):
    return -.5*(nu+p)*np.log1p(q/nu)

def score(x,mu,cov):
    d=x.shape[1]//2;a=x[:,:d];b=x[:,d:];ma=mu[:,:,:d];mb=mu[:,:,d:]
    aa=cov[:d,:d];bb=cov[d:,d:];ba=cov[d:,:d]
    ia=np.linalg.inv(aa);ib=np.linalg.inv(bb);ij=np.linalg.inv(cov)
    qa=quadratic(a,ma,ia);qb=quadratic(b,mb,ib);q=quadratic(x,mu,ij)
    c=bb-ba@ia@ba.T;ic=np.linalg.inv(c)
    conditional_mean=mb+np.einsum('ij,bkj->bki',ba@ia,a[:,None,:]-ma)
    qc=quadratic(b,conditional_mean,ic)
    assert np.max(np.abs(q-qa-qc))<1e-5
    pooled_cov=(aa+bb+ba+ba.T)/4
    qp=quadratic((a+b)/2,(ma+mb)/2,np.linalg.inv(pooled_cov))
    gaussian=-.5*q;student=student_log_from_q(q,2*d)
    assert np.array_equal(gaussian.argmax(1),student.argmax(1))
    # Exact decomposition of the multivariate-t density, including the
    # candidate-dependent conditional scale normalization.
    nu=5.;ca=student_log_from_q(qa,d,nu)
    scale=(nu+qa)/(nu+d)
    cb=-.5*d*np.log(scale)-.5*(nu+2*d)*np.log1p(qc/(nu+qa))
    assert np.max(np.abs((ca+cb-student)-(ca+cb-student)[:,:1]))<1e-6
    return {'pooled_gaussian':-.5*qp,'independent_gaussian':-.5*(qa+qb),
            'joint_gaussian':gaussian,'joint_student':student,
            'A_only':-.5*qa,'B_only':-.5*qb,
            'conditional_B_student':cb}

def fit(data,dest):
    tr=data.train;dev=torch.device('cuda');torch.set_num_threads(4)
    emb=np.array(data.emb,dtype=np.float32)
    groups=emb[:,:2];flat=torch.tensor(groups[tr].reshape(-1,1024),device=dev,dtype=torch.float64)
    center=flat.mean(0);z=flat-center;cov=z.T@z/len(z)
    values,vectors=torch.linalg.eigh(cov);basis=vectors[:,-DIM:];scale=values[-DIM:].clamp_min(1e-8).sqrt()
    allx=(torch.tensor(groups.reshape(-1,1024),device=dev,dtype=torch.float64)-center)@basis/scale
    x=allx.reshape(len(groups),2*DIM)
    v=data.vision.to(dev,dtype=torch.float64)
    w,offset,lam=ridge(v[tr],x[tr])
    final=(v@w+offset).cpu().numpy();oof=torch.empty((len(tr),2*DIM),dtype=torch.float64,device=dev)
    # Cross-fit the residual model by training concept; no validation outcome
    # informs the mapping or covariance. The neural PCA uses training EEG only.
    classes=np.random.default_rng(2026).permutation(data.splits['train']);fold_for={c:i%5 for i,c in enumerate(classes)}
    folds=np.array([fold_for[c] for c in data.labels[tr]])
    for fold in range(5):
        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
        assert set(data.labels[tr[train]]).isdisjoint(set(data.labels[tr[test]]))
        wf,bf,_=ridge(v[tr[train]],x[tr[train]])
        oof[test]=x[tr[test]]-(v[tr[test]]@wf+bf)
    residual=oof.cpu().numpy();residual_mean=residual.mean(0);residual-=residual_mean
    empirical=residual.T@residual/len(residual)
    covariance=.9*empirical+.1*np.trace(empirical)/len(empirical)*np.eye(len(empirical))
    assert np.linalg.eigvalsh(covariance).min()>0
    np.savez_compressed(dest/'model.npz',visual_to_neural=w.cpu().numpy(),offset=offset.cpu().numpy(),
        neural_center=center.cpu().numpy(),neural_basis=basis.cpu().numpy(),neural_scale=scale.cpu().numpy(),
        residual_covariance=covariance,residual_mean=residual_mean,fit_image_ids=data.ids[tr],fit_classes=data.splits['train'],
        training_residual_fold=folds)
    return x.cpu().numpy(),final+residual_mean,covariance,lam

def calibrate(data,queries,logits,y):
    first=set(data.splits['validation'][:50]);fold=np.array([0 if data.labels[q] in first else 1 for q in queries])
    scaled=np.empty_like(logits);ts=[]
    for f in [0,1]:
        t=temperature(logits[fold!=f],y[fold!=f]);scaled[fold==f]=logits[fold==f]/t;ts.append(t)
    return evaluate(scaled,y),scaled,ts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',default='sub-01');a=p.parse_args()
    assert a.subject in ['sub-01','sub-02','sub-03'];data=Banks(a.subject)
    dest=data.base.parent.parent/'cross_trial_likelihood_v6'/a.subject
    if dest.exists() and list(dest.iterdir()):raise FileExistsError('Preserve completed or partial v6 results')
    dest.mkdir(parents=True)
    config={'source_sha256':sha(__file__),'feature_source_sha256':sha(Path(__file__).with_name('native_temporal_screen.py')),
        'calibration_helper_sha256':sha(Path(__file__).with_name('reliability_screen.py')),
        'bank':data.manifest,'neural_dimension_per_group':DIM,'ridge_trace_multiplier':.1,'covariance_shrinkage':.1,
        'training_residual_concept_folds':5,'student_df':5.,'scope':'training fits, validation-only diagnostics, final concepts sealed'}
    save(dest/'configuration.json',config);begin=time.time();x,mu,cov,lam=fit(data,dest)
    summary={'config':config,'actual_ridge_regularizer':lam,'model_sha256':sha(dest/'model.npz'),'protocols':{}}
    for cross,task in [(False,'within10'),(True,'cross10')]:
        queries=data.val;rows,y=candidate_rows(data.labels,queries,queries,cross=cross,seed=20261103)
        scores=score(x[queries],mu[rows],cov)
        v=data.vision.numpy();g=np.einsum('bd,bkd->bk',np.array(data.emb[queries,2]),v[rows]);scores['global']=g
        report={}
        for method,s in scores.items():
            if method=='conditional_B_student':continue
            calibrated,scaled,ts=calibrate(data,queries,s,y)
            report[method]={'raw':evaluate(s,y),'CF':calibrated,'temperatures':ts}
            np.savez_compressed(dest/(task+'_'+method+'.npz'),logits=s,scaled_logits=scaled,targets=y,
                               query_ids=data.ids[queries],candidate_ids=data.ids[rows])
        summary['protocols'][task]=report
        if task=='within10':
            from metrics import agreement
            summary['support']=agreement(scores['A_only'],scores['B_only'],y)
            for intervention in ['b_mismatch','all_mismatch']:
                mismatch=within_class_mismatch(data.labels,queries,20261103);changed=x[queries].copy()
                if intervention=='b_mismatch':changed[:,DIM:]=x[mismatch,DIM:]
                else:changed=x[mismatch]
                ip=score(changed,mu[rows],cov)
                summary[intervention]={m:evaluate(ip[m],y) for m in ['pooled_gaussian','independent_gaussian','joint_gaussian','joint_student']}
    summary['elapsed_seconds']=time.time()-begin;save(dest/'summary.json',summary)
    print(json.dumps({m:v['CF'] for m,v in summary['protocols']['within10'].items()},indent=2),flush=True)

if __name__=='__main__':main()
