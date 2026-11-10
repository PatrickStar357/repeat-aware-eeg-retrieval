"""Calibration, agreement and subject-level statistics. No causal interpretation."""
import numpy as np
import torch
from torch.nn import functional as F

def fit_temperature(logits,targets):
    x=torch.as_tensor(logits,dtype=torch.float64).detach()
    y=torch.as_tensor(targets,dtype=torch.long)
    log_t=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],lr=.1,max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=F.cross_entropy(x/log_t.exp().clamp(.01,100),y)
        loss.backward(); return loss
    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(.01,100))

def classification_metrics(logits,targets,temperature=1.0):
    x=np.asarray(logits,dtype=float)/temperature
    x=x-x.max(1,keepdims=True)
    p=np.exp(x); p=p/p.sum(1,keepdims=True)
    y=np.asarray(targets,dtype=int); n=len(y)
    pred=p.argmax(1); correct=pred==y; confidence=p.max(1)
    order=np.argsort(-confidence,kind='stable')
    risk=np.cumsum(~correct[order])/np.arange(1,n+1)
    bins=np.minimum((confidence*15).astype(int),14)
    ece=sum(np.mean(bins==j)*abs(confidence[bins==j].mean()-correct[bins==j].mean())
            for j in range(15) if np.any(bins==j))
    onehot=np.eye(p.shape[1])[y]
    return {'top1':float(correct.mean()),
            'brier':float(np.mean(np.sum((p-onehot)**2,axis=1))),
            'nll':float(-np.log(p[np.arange(n),y].clip(1e-15)).mean()),
            'ece15':float(ece),'aurc':float(risk.mean()),
            'wrong_confidence':float(confidence[~correct].mean()) if np.any(~correct) else float('nan')}

def agreement(score_a,score_b,targets):
    a=np.asarray(score_a).argmax(1); b=np.asarray(score_b).argmax(1)
    y=np.asarray(targets); equal=a==b
    return {'RA':float(equal.mean()),'CA':float((equal & (a==y)).mean()),
            'WA':float((equal & (a!=y)).mean())}

def paired_subject_report(ours,control,seed=2026,bootstrap=10000):
    """[subjects,seeds]; average seeds first. Exact sign-flip test, no seed pseudoreplication."""
    a=np.asarray(ours,dtype=float); b=np.asarray(control,dtype=float)
    if a.shape!=b.shape or a.ndim!=2: raise ValueError('Need matched [subjects,seeds] arrays')
    d=(a-b).mean(1); n=len(d)
    if not 2<=n<=20: raise ValueError('Exact test supported for 2..20 subjects')
    rng=np.random.default_rng(seed)
    means=d[rng.integers(n,size=(bootstrap,n))].mean(1)
    signs=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
    null=(signs*d).mean(1)
    p=np.mean(np.abs(null)>=abs(d.mean())-1e-12)
    return {'subjects':n,'mean_difference':float(d.mean()),
            'subject_bootstrap_ci95':np.quantile(means,[.025,.975]).tolist(),
            'exact_sign_flip_p':float(p)}

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float); order=np.argsort(p)
    adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0,1)
    out=np.empty_like(adjusted); out[order]=adjusted
    return out
