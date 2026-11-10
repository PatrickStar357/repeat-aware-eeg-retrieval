"""Fixed-budget ten-participant backbones, independent of readout selection.

Twenty epochs, batch64, seed2028, fixed final checkpoint; no validation-based
epoch selection. Training-only normalization and RN50 image targets. The full
run recipe differs from the earlier batch24 development pilots and is recorded.
GPU-resident input means avoid repeated disk reads. Compact latent/global
banks replace this run's temporary three-embedding arrays after verified fit.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,gc,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from native_temporal_screen import ROOT,sha,save,seed_all
from validate_foundation import official_classes
from atm_control_v9 import author_classes
from protocol import category_split,repeat_groups
from load_visual_features import load_features
from cross_trial_likelihood_v6 import fit

class Nice(nn.Module):
    def __init__(self):
        super().__init__();ns,_=official_classes();self.encoder=ns['Enc_eeg']();self.projector=ns['Proj_eeg'](proj_dim=1024);self.logit_scale=nn.Parameter(torch.tensor(np.log(1/.07),dtype=torch.float32))
    def forward(self,x):return self.projector(self.encoder(x[:,None]))

def forward(model,x,arch):
    return model(x) if arch=='NICE' else model(x,torch.ones(len(x),device=x.device,dtype=torch.long))

class Data:
    def __init__(self,subject):
        self.subject=subject;self.folder=ROOT/'processed'/subject
        ids=np.load(self.folder/'train_identities.npz');pre=np.load(self.folder/'preprocessing.npz');self.splits=category_split(ids['class_id'])
        assert set(pre['preprocessing_fit_classes'])==set(self.splits['train'])
        self.original=np.flatnonzero(np.isin(ids['class_id'],np.r_[self.splits['train'],self.splits['validation']]))
        self.ids=ids['image_id'][self.original];self.labels=ids['class_id'][self.original].astype(str)
        self.train=np.flatnonzero(np.isin(self.labels,self.splits['train']));self.val=np.flatnonzero(np.isin(self.labels,self.splits['validation']))
        v,canonical=load_features('train');assert np.array_equal(canonical[self.original],self.ids)
        self.vision=F.normalize(torch.tensor(v[self.original],dtype=torch.float32),dim=-1)
        self.pairs=[repeat_groups(s) for s in ids['session_id'][self.original]]
        for ss,p in zip(ids['session_id'][self.original],self.pairs):assert set(ss[p[0]]).isdisjoint(ss[p[1]])
        self.raw=np.load(self.folder/'train_eeg.npy',mmap_mode='r');self.raw_means=torch.empty((len(self.ids),63,250),device='cuda')
        for start in range(0,len(self.ids),64):
            indices=np.arange(start,min(start+64,len(self.ids)));x=np.asarray(self.raw[self.original[indices]],dtype=np.float32)
            self.raw_means[indices]=torch.tensor(x.mean(1),device='cuda')
        self.gpu_vision=self.vision.cuda()
    def means3(self,indices):
        x=np.asarray(self.raw[self.original[indices]],dtype=np.float32);p=[self.pairs[i] for i in indices]
        a=np.stack([r[s[0]].mean(0) for r,s in zip(x,p)]);b=np.stack([r[s[1]].mean(0) for r,s in zip(x,p)])
        return np.stack([a,b,x.mean(1)],axis=1)

def train(data,arch,base,seed,epochs,batch):
    if (base/'manifest.json').exists():
        manifest=json.loads((base/'manifest.json').read_text());assert manifest['source_sha256']==sha(__file__)
        assert sha(base/'compact.npz')==manifest['compact_sha256']
        print('BACKBONE_ALREADY_COMPLETE',data.subject,arch,flush=True);return
    if (base/'compact.npz').exists():raise FileExistsError('Review completed compact bank without its final manifest')
    base.mkdir(parents=True,exist_ok=True);checkpoint=base/'model.pt';torch.cuda.reset_peak_memory_stats()
    ns,atm_hashes=author_classes();_,nice_hash=official_classes();source_hash=nice_hash if arch=='NICE' else atm_hashes
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert ck['trainer_sha256']==sha(__file__) and ck['seed']==seed and ck['epochs']==epochs and ck['batch']==batch
    else:
        # All caches are ready before RNG reset and model construction.
        seed_all(seed);model=(Nice() if arch=='NICE' else ns['ATMS']()).cuda()
        initial=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for _,t in sorted(model.state_dict().items()))).hexdigest()
        opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');rng=np.random.default_rng(seed);history=[];begin=time.time()
        for epoch in range(epochs):
            model.train();order=rng.permutation(data.train);losses=[];epoch_begin=time.time()
            for start in range(0,len(order),batch):
                ii=order[start:start+batch]
                if len(ii)<2:continue
                ix=torch.tensor(ii,device='cuda');x=data.raw_means[ix];v=data.gpu_vision[ix];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    z=F.normalize(forward(model,x,arch),dim=-1);scores=model.logit_scale.exp().clamp(max=100)*z@v.T;y=torch.arange(len(ix),device='cuda')
                    loss=.5*(F.cross_entropy(scores,y)+F.cross_entropy(scores.T,y))
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.time()-epoch_begin};history.append(row)
            save(base/'history.json',history)
            print('FULL_TRAIN',data.subject,arch,json.dumps(row),flush=True)
        ck={'model':model.state_dict(),'arch':arch,'seed':seed,'epochs':epochs,'batch':batch,'trainer_sha256':sha(__file__),
            'encoder_fit_classes':data.splits['train'].tolist(),'source_hash':source_hash,'checkpoint_selection':'fixed final epoch',
            'preprocessing_sha256':sha(data.folder/'preprocessing.npz'),'initial_state_sha256':initial}
        torch.save(ck,checkpoint);save(base/'training.json',{'trainer_sha256':sha(__file__),'epochs':epochs,'batch':batch,
            'parameters':sum(p.numel() for p in model.parameters()),'initial_sha256':initial,'seconds':time.time()-begin,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,'checkpoint_selection':'fixed final epoch, no development choice'})
        del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    ns,_=author_classes();model=Nice() if arch=='NICE' else ns['ATMS']();model.load_state_dict(ck['model']);model.cuda().eval()
    temporary=base/'temporary_embeddings.npy'
    if temporary.exists():raise FileExistsError('Review existing incomplete temporary embedding export')
    bank=np.lib.format.open_memmap(temporary,mode='w+',dtype='float32',shape=(len(data.ids),3,1024))
    with torch.no_grad():
        for start in range(0,len(data.ids),32):
            ii=np.arange(start,min(start+32,len(data.ids)));x=torch.tensor(data.means3(ii).reshape(-1,63,250),device='cuda')
            if arch=='NICE':
                # Match the development feature bank's native-token precision.
                t=model.encoder[0](x[:,None]).half().float();z=F.normalize(model.projector(t.flatten(1)),dim=-1)
            else:z=F.normalize(forward(model,x,arch),dim=-1)
            bank[ii]=z.reshape(len(ii),3,1024).cpu().numpy()
    bank.flush();del bank,model;gc.collect();torch.cuda.empty_cache()
    data.emb=np.load(temporary,mmap_mode='r');x,mu,cov,lam=fit(data,base)
    raw4=np.asarray(data.emb[:,2],dtype=np.float32).copy()
    np.savez_compressed(base/'compact.npz',group_latents=x.reshape(-1,2,64),global_embeddings=raw4,image_ids=data.ids,class_ids=data.labels,original_indices=data.original)
    check=np.load(base/'compact.npz');assert check['group_latents'].shape==(len(data.ids),2,64) and np.array_equal(check['image_ids'],data.ids)
    assert np.isfinite(check['group_latents']).all() and np.isfinite(check['global_embeddings']).all();check.close()
    data.emb._mmap.close();data.emb=None;gc.collect()
    assert temporary.resolve().parent==base.resolve() and temporary.name=='temporary_embeddings.npy';temporary.unlink()
    save(base/'manifest.json',{'source_sha256':sha(__file__),'arch':arch,'subject':data.subject,'seed':seed,'epochs':epochs,'batch':batch,
        'checkpoint_sha256':sha(checkpoint),'mapping_sha256':sha(base/'model.npz'),'compact_sha256':sha(base/'compact.npz'),
        'preprocessing_sha256':sha(data.folder/'preprocessing.npz'),'fit_classes':data.splits['train'].tolist(),
        'bank_scope':'training1254 and development100 only; calibration/fine-test not encoded',
        'temporary_embeddings_removed':True,'ridge_regularizer':lam})
    print('BACKBONE_COMPLETE',data.subject,arch,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);p.add_argument('--seed',type=int,default=2028);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=64);a=p.parse_args()
    torch.set_num_threads(2);root=ROOT/'results/all_subject_backbones_v42';root.mkdir(exist_ok=True)
    config={'source_sha256':sha(__file__),'seed':a.seed,'epochs':a.epochs,'batch':a.batch,'subjects':[f'sub-{i:02}' for i in range(1,11)],
        'encoders':['NICE','ATM'],'optimizer':'AdamW lr2e-4 wd0.01','R':4,'vision':'frozen RN501024','checkpoint':'fixed final epoch',
        'purpose':'Method-independent backbone preparation; readout/contrast mechanism selected separately',
        'formal_calibration_and_test_access':False,'recipe_change_from_pilots':'batch64 instead of24, fixed final epoch instead of best development epoch'}
    cp=root/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:save(cp,config)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    for subject in a.subjects:
        if subject not in config['subjects']:raise ValueError(subject)
        folder=ROOT/'processed'/subject
        while not (folder/'preprocessing.npz').exists():
            print('WAITING_FOR_PREPROCESSING',subject,flush=True);time.sleep(20)
        if subject not in ['sub-01','sub-02','sub-03']:
            while not (folder/'stream_manifest.json').exists():time.sleep(5)
        base=root/subject/f'seed{a.seed}'
        if all((base/arch/'manifest.json').exists() for arch in ['NICE','ATM']):continue
        print('BACKBONE_DATA_LOADING',subject,flush=True);data=Data(subject)
        for arch in ['NICE','ATM']:
            train(data,arch,base/arch,a.seed,a.epochs,a.batch)
        del data;gc.collect();torch.cuda.empty_cache()
        save(root/'progress.json',{'completed_subject':subject,'completed':[s for s in config['subjects'] if all((root/s/f'seed{a.seed}'/arch/'manifest.json').exists() for arch in ['NICE','ATM'])]})
    print('ALL_REQUESTED_BACKBONES_COMPLETE',flush=True)

if __name__=='__main__':main()
