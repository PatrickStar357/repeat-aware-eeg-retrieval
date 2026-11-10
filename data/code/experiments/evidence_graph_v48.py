"""Candidate-pair neural evidence with repeat-noise-aware graph reconciliation.

Exploratory candidate, not an approved paper method. Diagnostic-feature
detection motivates pair contrasts; covariance models correlated comparisons;
weighted least squares reconciles a learned, possibly cyclic comparison field.
Unweighted complete-graph reconciliation is exactly Borda averaging and is
explicitly a comparator, not presented as a new algorithm.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,time,hashlib,gc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from native_temporal_screen import ROOT,sha,save,seed_all
from discriminant_evidence_v40 import load_geometry,covariance
from protocol import candidate_rows,within_class_mismatch
from load_visual_features import load_features
from reliability_repair_v15 import load_scores
from score_stacking_v24 import fit_stack
from metric_suite_v23 import evaluate_metrics

MODES=['unary','pair_mean','pair_diagonal','pair_joint']

def incidence(k,device=None,dtype=torch.float32):
    left,right=torch.triu_indices(k,k,1,device=device);b=torch.zeros((len(left),k),device=device,dtype=dtype)
    b[torch.arange(len(left),device=device),left]=1;b[torch.arange(len(left),device=device),right]=-1
    return left,right,b

def reconcile(edges,queries,b,mode):
    """edges [g,n,views,pairs]; queries [g,pairs,d]."""
    k=b.shape[1]
    if mode=='pair_mean':return torch.einsum('gnve,ek->gnvk',edges,b)/k
    empirical=queries@queries.transpose(-1,-2);variance=empirical.diagonal(dim1=-2,dim2=-1).clamp_min(1e-5)
    c=torch.diag_embed(variance) if mode=='pair_diagonal' else .9*empirical+.1*torch.diag_embed(variance)
    weighted_b=torch.linalg.solve(c,b.expand(len(c),-1,-1))
    laplacian=b.T@weighted_b
    gauge=torch.ones((k,k),device=b.device,dtype=b.dtype)/k
    rhs=torch.einsum('gek,gnve->gnvk',weighted_b,edges)
    # The gauge fixes the unidentifiable common offset, without preferring a candidate.
    score=torch.linalg.solve(laplacian+gauge,rhs.flatten(1,2).transpose(-1,-2)).transpose(-1,-2).reshape(rhs.shape)
    return score

class EvidenceHead(nn.Module):
    def __init__(self,d=128,width=32):
        super().__init__();self.diff=nn.Linear(d,width,bias=False);self.context=nn.Linear(d,width,bias=False);self.out=nn.Linear(width,d,bias=False)
        self.log_temperature=nn.Parameter(torch.zeros(()));nn.init.normal_(self.out.weight,std=.001)
    def forward(self,x,mu,mode,return_edges=False):
        # x [groups,queries,views,d], mu [groups,candidates,d]. No other query EEG
        # enters a query's score, and no test label enters any computation.
        if mode=='unary':
            center=mu.mean(1,keepdim=True);adjustment=self.out(torch.tanh(self.diff(mu))*torch.sigmoid(self.context(center)))
            template=mu+.1*adjustment
            score=torch.einsum('gnvd,gkd->gnvk',x,template)-.5*template.square().sum(-1)[:,None,None]
            return score/self.log_temperature.exp().clamp(.1,10)
        left,right,b=incidence(mu.shape[1],mu.device,mu.dtype)
        difference=mu[:,left]-mu[:,right];middle=(mu[:,left]+mu[:,right])/2
        query=difference+.1*self.out(torch.tanh(self.diff(difference))*torch.sigmoid(self.context(middle)))
        edges=torch.einsum('gnvd,ged->gnve',x,query)-(middle*query).sum(-1)[:,None,None]
        scores=reconcile(edges,query,b,mode)/self.log_temperature.exp().clamp(.1,10)
        if return_edges:
            potential=torch.einsum('gnvk,ek->gnve',scores*self.log_temperature.exp().clamp(.1,10),b)
            return scores,edges,edges-potential
        return scores

def load_data(subject,folder):
    cache=folder/'features.npz';meta=folder/'feature_sources.json'
    if cache.exists():
        record=json.loads(meta.read_text());assert record['builder_sha256']==sha(__file__)
        return dict(np.load(cache)),record
    visual,canonical=load_features('train');visual=F.normalize(torch.tensor(visual,dtype=torch.float32),dim=-1).numpy().astype(float)
    gs={a:load_geometry(subject,a,visual,canonical) for a in ['NICE','ATM']};a=gs['NICE'];allx=[];allmu=[];sources={};raw=[]
    for arch,g in gs.items():
        assert np.array_equal(g['ids'],a['ids']);allx.append(np.stack([g['x']+g['difference'],g['x']-g['difference']],axis=1))
        mu=g['mu'].copy();mu[g['tr']]=g['oof_mu'];allmu.append(mu);sources[arch]=g['sources']
        s,y,fold,_,q,c,classes,prov=load_scores(arch,subject,2026,visual,canonical);raw.append(s)
    x=np.concatenate(allx,-1);mu=np.concatenate(allmu,-1);tr=a['tr'];val=a['q'];delta=(x[:,0]-x[:,1])/2;bias=delta[tr].mean(0)
    x[:,0]-=bias;x[:,1]+=bias
    cov=covariance(delta[tr],'joint');chol=np.linalg.cholesky(cov);xw=np.linalg.solve(chol,x.reshape(-1,128).T).T.reshape(x.shape);mw=np.linalg.solve(chol,mu.T).T
    scale=np.sqrt(np.mean(xw[tr]**2));xw/=scale;mw/=scale
    train_groups=np.array([np.flatnonzero(a['labels']==cl) for cl in sorted(a['split']['train'])]);assert train_groups.shape==(1254,10)
    rows,yy=candidate_rows(a['labels'],val,val,seed=20261103);assert np.array_equal(yy,y)
    base=np.stack([raw[0]['global'],raw[1]['global'],raw[0]['repeat_difference_proxy'],raw[1]['repeat_difference_proxy']],axis=-1)
    data={'x':xw.astype(np.float32),'mu':mw.astype(np.float32),'train_groups':train_groups,'validation':val,'candidate_rows':rows,
        'targets':y,'fold':fold,'query_ids':q,'candidate_ids':c,'concepts':classes,'base_scores':base,
        'image_ids':a['ids'],'class_ids':a['labels'],'covariance':cov,'whitening_scale':np.array(scale),'group_bias':bias}
    np.savez_compressed(cache,**data);record={'builder_sha256':sha(__file__),'sources':sources,
        'fit_concepts':a['split']['train'].tolist(),'validation_concepts':a['split']['validation'].tolist(),
        'boundary':'OOF image templates for training classes; frozen encoders saw all1254 training classes; no calibration/fine-test accessed'}
    save(meta,record);return data,record

def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train(data,mode,folder,epochs):
    seed_all(20261104);model=EvidenceHead().cuda();initial=state_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    x=torch.tensor(data['x'],device='cuda');mu=torch.tensor(data['mu'],device='cuda');groups=data['train_groups'];rng=np.random.default_rng(20261104);history=[]
    start=time.time()
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(groups));losses=[]
        for offset in range(0,len(order),8):
            rows=groups[order[offset:offset+8]];ii=torch.tensor(rows,device='cuda');observed=x[ii]
            # Three views use the same R4 recordings; no additional observations.
            views=torch.cat([observed,observed.mean(2,keepdim=True)],dim=2)
            scores=model(views,mu[ii],mode);target=torch.arange(10,device='cuda')[None,:,None].expand(len(ii),10,3)
            loss=F.cross_entropy(scores.reshape(-1,10),target.reshape(-1))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite graph objective')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.time()-start};history.append(row)
        print('EVIDENCE_GRAPH_TRAIN',mode,json.dumps(row),flush=True)
    model.eval();torch.save({'model':model.state_dict(),'initial_sha256':initial,'source_sha256':sha(__file__),
        'mode':mode,'epochs':epochs,'train_image_ids':data['image_ids'][groups.ravel()].tolist()},folder/(mode+'.pt'))
    save(folder/(mode+'_training.json'),{'initial_sha256':initial,'history':history,'parameters':sum(p.numel() for p in model.parameters()),'fixed_final_epoch':True})
    del opt,x,mu;gc.collect();torch.cuda.empty_cache();return model,initial

@torch.no_grad()
def predict(model,data,mode):
    rows=data['candidate_rows'];q=data['validation'];x=data['x'];mu=data['mu'];z=[];cycle=[];a_scores=[];b_scores=[]
    for start in range(0,len(q),32):
        ix=q[start:start+32];v=torch.tensor(mu[rows[start:start+32]],device='cuda');observed=torch.tensor(x[ix],device='cuda')[:,None]
        views=torch.cat([observed,observed.mean(2,keepdim=True)],2)
        if mode=='unary':scores=model(views,v,mode);cy=torch.zeros(len(ix),device='cuda')
        else:
            scores,e,c=model(views,v,mode,True);cy=c[:,0,2].square().mean(-1)/(e[:,0,2].square().mean(-1)+1e-8)
        z.append(scores[:,0,2].cpu().numpy());a_scores.append(scores[:,0,0].cpu().numpy());b_scores.append(scores[:,0,1].cpu().numpy());cycle.append(cy.cpu().numpy())
    return np.concatenate(z),np.concatenate(a_scores),np.concatenate(b_scores),np.concatenate(cycle)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--epochs',type=int,default=12);a=p.parse_args()
    assert a.subject in ['sub-01','sub-02','sub-03'];torch.set_num_threads(1);folder=ROOT/'results/evidence_graph_v48'/a.subject
    if (folder/'summary.json').exists():raise FileExistsError('Preserve completed graph screen')
    folder.mkdir(parents=True,exist_ok=True);source=sha(__file__)
    save(folder/'configuration.json',{'source_sha256':source,'primary':'pair_joint','controls':MODES,'epochs':a.epochs,'seed':20261104,
        'same_network_parameters':12289,'same_training_views':'A2,B2,andtheirpooledR4; no extra EEG','source_covariance':'training-only repeat differences',
        'training_template_boundary':'concept-cross-fitted visual-to-neural templates; encoders frozen, trainedon1254 concepts',
        'claim_boundary':'candidate method under novelty/performance review; no final-data or CVPR readiness claim','final_data_used':False})
    data,provenance=load_data(a.subject,folder);results={};parameters={};diagnostics={};initials={};begin=time.time()
    for mode in MODES:
        model,initial=train(data,mode,folder,a.epochs);initials[mode]=initial;local,sa,sb,cycle=predict(model,data,mode)
        np.savez_compressed(folder/(mode+'_readout.npz'),pooled=local,A=sa,B=sb,cycle_ratio=cycle,targets=data['targets'])
        for k in [2,5]:
            fold=data['fold'] if k==2 else np.searchsorted(np.unique(data['concepts']),data['concepts'])//20
            z=np.empty_like(local);pars=[];comp=[data['base_scores'][...,j] for j in range(4)]+[local]
            for f in range(k):
                fit=fold!=f;zz,pa=fit_stack(comp,data['targets'],fit,'brier');z[~fit]=zz[~fit];pars.append(pa)
            results.setdefault(str(k),{})[mode]=evaluate_metrics(z,data['targets'])[0];parameters.setdefault(str(k),{})[mode]=pars
            np.savez_compressed(folder/f'fold{k}_{mode}.npz',logits=z,targets=data['targets'],query_ids=data['query_ids'],candidate_ids=data['candidate_ids'],concepts=data['concepts'])
        diagnostics[mode]={'standalone':evaluate_metrics(local,data['targets'])[0],'mean_cycle_ratio':float(cycle.mean()),
            'A_B_top1_agreement':float((sa.argmax(1)==sb.argmax(1)).mean()),'A_accuracy':float((sa.argmax(1)==data['targets']).mean()),'B_accuracy':float((sb.argmax(1)==data['targets']).mean())}
        print('GRAPH_SCREEN',a.subject,mode,{f:{m:round(results[f][mode][m],6) for m in ['top1','map','brier','nll','ece15']} for f in results},flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(set(initials.values()))==1 and sha(__file__)==source
    save(folder/'summary.json',{'source_sha256':source,'results':results,'parameters':parameters,'diagnostics':diagnostics,'initial_hashes':initials,
        'provenance':provenance,'seconds':time.time()-begin,'final_data_used':False})

if __name__=='__main__':main()
