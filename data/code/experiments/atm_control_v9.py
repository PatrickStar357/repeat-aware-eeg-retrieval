"""ATM-S/RN50 controlled adaptation and fixed noise-readout transfer.

Author class definitions are extracted without executing their training CLI.
Architecture semantics, including the author's subject-token / :63 slice,
are preserved. Objective, RN50 targets, R4 pooling and split are controlled
to match the NICE experiment: this is not published-protocol reproduction.
"""
from pathlib import Path
import argparse,ast,math,json,time,gc
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from validate_foundation import ROOT,candidate_metrics
from protocol import category_split,repeat_groups,candidate_rows
from load_visual_features import load_features
from native_temporal_screen import sha,save,seed_all
from cross_trial_likelihood_v6 import fit,score,quadratic
from compare_noise_controls import crossfit_components
from reliability_screen import evaluate

def author_classes():
    upstream=ROOT/'code/upstream/ATM'
    selections=[('models/subject_layers/Embed.py',{'PositionalEmbedding','FixedEmbedding','TemporalEmbedding','TimeFeatureEmbedding','SubjectEmbedding','DataEmbedding'}),
        ('models/subject_layers/SelfAttention_Family.py',{'FullAttention','AttentionLayer'}),
        ('models/subject_layers/Transformer_EncDec.py',{'Encoder','EncoderLayer'}),
        ('models/loss.py',{'ClipLoss'}),
        ('Retrieval/ATMS_retrieval.py',{'Config','iTransformer','PatchEmbedding','ResidualAdd','FlattenHead','Enc_eeg','Proj_eeg','ATMS'})]
    ns={'torch':torch,'nn':nn,'np':np,'math':math,'sqrt':math.sqrt,'Tensor':Tensor,'Rearrange':Rearrange,'F':F}
    hashes={}
    for relative,names in selections:
        source=upstream/relative;tree=ast.parse(source.read_text(encoding='utf8'))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
        assert {n.name for n in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
        hashes[relative]=sha(source)
    return ns,hashes

class Pool:
    def __init__(self,subject):
        self.subject=subject;self.folder=ROOT/'processed'/subject
        identity=dict(np.load(self.folder/'train_identities.npz'));pre=dict(np.load(self.folder/'preprocessing.npz'))
        self.full_labels=identity['class_id'].astype(str);self.splits=category_split(self.full_labels)
        assert set(pre['preprocessing_fit_classes'])==set(self.splits['train'])
        self.original=np.flatnonzero(np.isin(self.full_labels,np.r_[self.splits['train'],self.splits['validation']]))
        self.labels=self.full_labels[self.original];self.ids=identity['image_id'][self.original]
        self.train=np.flatnonzero(np.isin(self.labels,self.splits['train']));self.val=np.flatnonzero(np.isin(self.labels,self.splits['validation']))
        self.eeg=np.load(self.folder/'train_eeg.npy',mmap_mode='r');self.pairs=[repeat_groups(s) for s in identity['session_id'][self.original]]
        for s in identity['session_id'][self.original]:
            a,b=repeat_groups(s);assert set(s[a]).isdisjoint(set(s[b]))
        visual,ids=load_features('train');assert np.array_equal(ids[self.original],self.ids)
        self.vision=F.normalize(torch.tensor(visual[self.original]),dim=-1)
    def means(self,indices,three=False):
        x=np.array(self.eeg[self.original[indices]],dtype=np.float32)
        if not three:return x.mean(1)
        pairs=[self.pairs[i] for i in indices]
        a=np.stack([r[p[0]].mean(0) for r,p in zip(x,pairs)]);b=np.stack([r[p[1]].mean(0) for r,p in zip(x,pairs)])
        return np.stack([a,b,x.mean(1)],1)

@torch.no_grad()
def encode(model,data,indices,batch=24):
    model.eval();zs=[]
    for start in range(0,len(indices),batch):
        x=torch.tensor(data.means(indices[start:start+batch]),device='cuda');subjects=torch.ones(len(x),device='cuda',dtype=torch.long)
        with torch.autocast('cuda'):z=model(x,subjects)
        zs.append(z.float().cpu())
    return F.normalize(torch.cat(zs),dim=-1)

def train(data,base,seed,epochs):
    dest=base/'baseline'
    if (dest/'summary.json').exists():return
    if dest.exists() and list(dest.iterdir()):raise ValueError('Inspect incomplete ATM training')
    dest.mkdir(parents=True,exist_ok=True);ns,source_hashes=author_classes();seed_all(seed)
    model=ns['ATMS']().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    initial=sha_state(model);history=[];rng=np.random.default_rng(seed);best=-1.;begin=time.time()
    for epoch in range(epochs):
        model.train();order=rng.permutation(data.train);losses=[];start=time.time()
        for offset in range(0,len(order),24):
            ix=order[offset:offset+24]
            if len(ix)<2:continue
            x=torch.tensor(data.means(ix),device='cuda');v=data.vision[ix].cuda();subject=torch.ones(len(x),device='cuda',dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                z=F.normalize(model(x,subject),dim=-1);logits=model.logit_scale.exp().clamp(max=100)*z@v.T;y=torch.arange(len(x),device='cuda')
                loss=(F.cross_entropy(logits,y)+F.cross_entropy(logits.T,y))/2
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        z=encode(model,data,data.val);metrics,pred=candidate_metrics(z,data.vision,data.labels,data.val)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),**metrics,'seconds':time.time()-start};history.append(row)
        print('ATM',data.subject,json.dumps(row),flush=True)
        if row['within10']>best:
            best=row['within10'];torch.save({'model':model.state_dict(),'encoder_fit_classes':data.splits['train'].tolist(),
                'source_hashes':source_hashes,'trainer_sha256':sha(__file__),'seed':seed,'epoch':epoch+1,
                'pooling':'raw_input_mean','R':4,'subject_id':1,'projection_dim':1024},dest/'model.pt')
            np.savez_compressed(dest/'validation_predictions.npz',**pred,query_indices=data.original[data.val],image_id=data.ids[data.val])
        save(dest/'history.json',history)
    result={'source_hashes':source_hashes,'trainer_sha256':sha(__file__),'checkpoint_sha256':sha(dest/'model.pt'),
        'initial_state_sha256':initial,'seed':seed,'epochs':epochs,'batch_size':24,'R':4,'lr':.0002,'weight_decay':.01,
        'declared_parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.time()-begin,
        'scope':'controlled RN50 raw-input R4, image-only normalized symmetric contrastive loss; not author protocol',
        'architecture_note':'Original DataEmbedding appends a subject token and iTransformer keeps the first 63 outputs, including that token; preserved exactly.'}
    save(dest/'summary.json',result);del model,opt;gc.collect();torch.cuda.empty_cache()

def sha_state(model):
    import hashlib
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def export_bank(data,base):
    bank=base/'banks';bank.mkdir(exist_ok=True);ckpath=base/'baseline/model.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=True);ns,hashes=author_classes()
    assert ck['source_hashes']==hashes and ck['trainer_sha256']==sha(__file__)
    assert set(ck['encoder_fit_classes'])==set(data.splits['train'])
    data.manifest={'checkpoint_sha256':sha(ckpath),'preprocessing_sha256':sha(data.folder/'preprocessing.npz'),
        'architecture':'author ATMS forward, controlled RN50 objective','scope':'training and validation only','precision':'FP32 full-model inference'}
    if not (bank/'manifest.json').exists():
        if list(bank.iterdir()):raise ValueError('Inspect incomplete ATM bank')
        model=ns['ATMS']().cuda();model.load_state_dict(ck['model']);model.eval()
        e=np.lib.format.open_memmap(bank/'embeddings.npy',mode='w+',dtype='float32',shape=(len(data.ids),3,1024))
        with torch.no_grad():
            for start in range(0,len(data.ids),24):
                ix=np.arange(start,min(start+24,len(data.ids)));x=torch.tensor(data.means(ix,three=True).reshape(-1,63,250),device='cuda')
                z=F.normalize(model(x,torch.ones(len(x),device='cuda',dtype=torch.long)),dim=-1)
                e[ix]=z.reshape(len(ix),3,1024).cpu().numpy()
        e.flush();del e,model;torch.cuda.empty_cache();save(bank/'manifest.json',data.manifest)
    assert json.loads((bank/'manifest.json').read_text())==data.manifest
    data.emb=np.load(bank/'embeddings.npy',mmap_mode='r')

def evaluate_head(data,base):
    if (base/'summary.json').exists():return
    statistics=base/'statistics';statistics.mkdir(exist_ok=True)
    if list(statistics.iterdir()):raise ValueError('Inspect incomplete ATM statistics')
    x,mu,cov,lam=fit(data,statistics);summary={'manifest':data.manifest,'source_sha256':sha(__file__),'ridge_regularizer':lam,'protocols':{}}
    for cross,task in [(False,'within10'),(True,'cross10')]:
        q=data.val;rows,y=candidate_rows(data.labels,q,q,cross=cross,seed=20261103);scores=score(x[q],mu[rows],cov);d=64
        cm=(cov[:d,:d]+cov[d:,d:]+cov[:d,d:]+cov[d:,:d])/4
        m=(x[q,:d]+x[q,d:])/2;candidate=(mu[rows][:,:,:d]+mu[rows][:,:,d:])/2
        scores['isotropic']=-.5*quadratic(m,candidate,np.eye(d))
        scores['diagonal_noise']=-.5*quadratic(m,candidate,np.diag(1/np.diag(cm)))
        scores['raw4']=np.einsum('bd,bkd->bk',np.array(data.emb[q,2]),data.vision.numpy()[rows])
        first=set(data.splits['validation'][:50]);fold=np.array([0 if data.labels[i] in first else 1 for i in q]);metrics={}
        for name,comp in [('raw4',[scores['raw4']]),('raw4+isotropic',[scores['raw4'],scores['isotropic']]),
                ('raw4+diagonal_noise',[scores['raw4'],scores['diagonal_noise']]),('raw4+noise',[scores['raw4'],scores['pooled_gaussian']])]:
            z,params=crossfit_components(comp,y,fold);metrics[name]={'metrics':evaluate(z,y),'parameters':params}
            np.savez_compressed(base/(task+'_'+name.replace('+','_plus_')+'.npz'),logits=z,targets=y,query_ids=data.ids[q],candidate_ids=data.ids[rows])
        summary['protocols'][task]=metrics
    save(base/'summary.json',summary);print('ATM_TRANSFER',json.dumps({k:v['metrics'] for k,v in summary['protocols']['within10'].items()}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',default='sub-01');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--probe',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    if a.probe:
        ns,hashes=author_classes();seed_all(2026);model=ns['ATMS']().eval();x=torch.randn(3,63,250)
        y=model(x,torch.ones(3,dtype=torch.long));assert y.shape==(3,1024) and torch.isfinite(y).all()
        y.square().mean().backward();assert model.encoder.encoder.attn_layers[0].attention.query_projection.weight.grad.norm()>0
        print(json.dumps({'CPU_forward_backward':'passed','shape':list(y.shape),'parameters':sum(p.numel() for p in model.parameters()),'source_hashes':hashes},indent=2));return
    assert a.subject in ['sub-01','sub-02','sub-03']
    # Leave room for the two already running small NICE replications. This
    # caps tensor allocation, not the driver's process-context overhead.
    torch.cuda.set_per_process_memory_fraction(.22)
    base=ROOT/'results/atm_noise_control_v9'/a.subject/f'seed{a.seed}';base.mkdir(parents=True,exist_ok=True)
    config={'trainer_sha256':sha(__file__),'statistical_sha256':sha(Path(__file__).with_name('cross_trial_likelihood_v6.py')),
        'calibration_sha256':sha(Path(__file__).with_name('compare_noise_controls.py')),'subject':a.subject,'seed':a.seed,'epochs':a.epochs,
        'R':4,'vision':'frozen RN50','split':'1254/100/100/200','fit_held_out_EEG':False,'scope':'validation-only independent architecture control'}
    cp=base/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:save(cp,config)
    data=Pool(a.subject);train(data,base,a.seed,a.epochs);export_bank(data,base);evaluate_head(data,base)

if __name__=='__main__':main()
