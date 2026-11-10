"""Bounded mechanism repair: pretrained temporal tokens and frozen visual readout.

Development only. Native NICE tokens integrate the channels: 36 overlapping
temporal positions (300-ms receptive field, 20-ms stride), not scalp locations.
Uniform masks exactly recover the same frozen projector's group scores.
"""
from pathlib import Path
import argparse,hashlib,json,time,random,gc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from validate_foundation import ROOT,official_classes
from protocol import category_split,repeat_groups,candidate_rows,within_class_mismatch
from load_visual_features import load_features
from metrics import classification_metrics,agreement

MODES=['uniform','cross','two_view','same_pool','mlp']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False),encoding='utf8')
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

class Banks:
    def __init__(self,subject):
        self.subject=subject;self.folder=ROOT/'processed'/subject
        ids=dict(np.load(self.folder/'train_identities.npz'));pre=dict(np.load(self.folder/'preprocessing.npz'))
        self.splits=category_split(ids['class_id']);assert set(pre['preprocessing_fit_classes'])==set(self.splits['train'])
        self.original=np.flatnonzero(np.isin(ids['class_id'],np.r_[self.splits['train'],self.splits['validation']]))
        self.ids=ids['image_id'][self.original];self.labels=ids['class_id'][self.original].astype(str)
        self.train=np.flatnonzero(np.isin(self.labels,self.splits['train']));self.val=np.flatnonzero(np.isin(self.labels,self.splits['validation']))
        self.pairs=[repeat_groups(s) for s in ids['session_id'][self.original]]
        self.checkpoint=ROOT/'results/input_average_screen_v1'/subject/'seed2026/input_average.pt'
        self.ck=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        assert self.ck['pooling']=='raw_input_mean' and set(self.ck['encoder_fit_classes'])==set(self.splits['train'])
        self.base=ROOT/'results/native_temporal_screen_v3'/subject;self.cache=self.base/'banks';self.cache.mkdir(parents=True,exist_ok=True)
        self.manifest={'bank_version':'native36_masked_projector_v1','checkpoint_sha256':sha(self.checkpoint),
            'preprocessing_sha256':sha(self.folder/'preprocessing.npz'),
            'image_ids_sha256':hashlib.sha256('\n'.join(self.ids).encode()).hexdigest(),
            'scope':'training and validation concepts only','token_shape':[len(self.ids),3,36,40]}
        if not (self.cache/'manifest.json').exists():self.export()
        assert json.loads((self.cache/'manifest.json').read_text())==self.manifest
        self.tokens=np.load(self.cache/'tokens.npy',mmap_mode='r');self.emb=np.load(self.cache/'embeddings.npy',mmap_mode='r')
        vision,canonical=load_features('train');assert np.array_equal(canonical[self.original],self.ids)
        self.vision=F.normalize(torch.tensor(vision[self.original]),dim=-1)
    def projector(self):
        ns,_=official_classes();p=ns['Proj_eeg'](proj_dim=1024);p.load_state_dict(self.ck['projector'])
        return p.requires_grad_(False).eval()
    @torch.no_grad()
    def export(self):
        if list(self.cache.iterdir()):raise ValueError('Inspect partial native cache before rebuilding')
        ns,_=official_classes();enc=ns['Enc_eeg']().cuda();enc.load_state_dict(self.ck['encoder']);enc.eval()
        proj=self.projector().cuda();eeg=np.load(self.folder/'train_eeg.npy',mmap_mode='r')
        tokens=np.lib.format.open_memmap(self.cache/'tokens.npy',mode='w+',dtype='float16',shape=(len(self.ids),3,36,40))
        emb=np.lib.format.open_memmap(self.cache/'embeddings.npy',mode='w+',dtype='float32',shape=(len(self.ids),3,1024))
        for start in range(0,len(self.ids),32):
            sl=slice(start,start+32);x=np.array(eeg[self.original[sl]],dtype=np.float32)
            pairs=self.pairs[sl];a=np.stack([r[p[0]].mean(0) for r,p in zip(x,pairs)]);b=np.stack([r[p[1]].mean(0) for r,p in zip(x,pairs)])
            means=np.stack([a,b,x.mean(1)],axis=1)
            t=enc[0](torch.tensor(means.reshape(-1,1,63,250),device='cuda'))
            assert t.shape[1:]==(36,40)
            t=t.half().float();z=F.normalize(proj(t.flatten(1)),dim=-1)
            tokens[sl]=t.reshape(len(x),3,36,40).cpu().numpy();emb[sl]=z.reshape(len(x),3,1024).cpu().numpy()
        tokens.flush();emb.flush();del tokens,emb,enc,proj;torch.cuda.empty_cache()
        save(self.cache/'manifest.json',self.manifest);print('NATIVE_BANKS_EXPORTED',self.subject,flush=True)
    def batch(self,queries,rows,b_mismatch=None,all_mismatch=None):
        q=queries if all_mismatch is None else all_mismatch
        t=torch.tensor(np.array(self.tokens[q],dtype=np.float32),device='cuda')
        emb=torch.tensor(np.array(self.emb[q],dtype=np.float32),device='cuda')
        if b_mismatch is not None:t[:,1]=torch.tensor(np.array(self.tokens[b_mismatch,1],dtype=np.float32),device='cuda')
        v=self.vision[rows].cuda();g=torch.einsum('bd,bkd->bk',emb[:,2],v)
        return t,emb,v,g

class NativeReadout(nn.Module):
    def __init__(self,data,mode):
        super().__init__();self.mode=mode;self.projector=data.projector()
        self.query=nn.Linear(1024,80,bias=False);nn.init.zeros_(self.query.weight)
        self.fusion=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,1))
        if mode=='uniform':self.query.requires_grad_(False)
        if mode=='mlp':
            self.query.requires_grad_(False);self.fusion.requires_grad_(False)
            self.mlp=nn.Sequential(nn.Linear(4097,20),nn.GELU(),nn.Linear(20,1))
    def train(self,mode=True):super().train(mode);self.projector.eval();return self
    def select(self,t,v):
        if self.mode=='uniform':return torch.full((len(t),v.shape[1],2,36),1/36,device=t.device)
        q=self.query(v-v.mean(1,keepdim=True)).reshape(len(t),v.shape[1],2,40)
        return torch.softmax(torch.einsum('bkmd,bld->bkml',q,t)/(40**.5),dim=-1)
    def read(self,alpha,t,v):
        weighted=36*alpha[...,None]*t[:,None,None,:,:]
        z=self.projector(weighted.reshape(-1,1440)).reshape(len(t),v.shape[1],2,1024)
        return F.cosine_similarity(z,v[:,:,None,:],dim=-1)
    def forward(self,t,emb,v,g):
        if self.mode=='mlp':
            a=emb[:,0,None].expand(-1,v.shape[1],-1);b=emb[:,1,None].expand_as(a)
            return {'logits':g+self.mlp(torch.cat([a,b,(a-b).abs(),v,g[...,None]],-1)).squeeze(-1)}
        a,b=t[:,0],t[:,1]
        if self.mode=='same_pool':a=b=t[:,2]
        alpha=self.select(a,v);aa=self.read(alpha,a,v)
        beta=self.select(b,v) if self.mode=='two_view' else alpha;bb=self.read(beta,b,v)
        evidence=torch.cat([aa,bb,(aa-bb).abs(),g[...,None]],dim=-1)
        return {'logits':g+self.fusion(evidence).squeeze(-1),'score_a':aa.mean(-1),'score_b':bb.mean(-1),'alpha':alpha}

@torch.no_grad()
def predict(model,data,queries,rows,y,batch=32,intervention=None):
    model.eval();out=[];aa=[];bb=[];gg=[]
    mismatch=within_class_mismatch(data.labels,queries,20261103) if intervention else None
    for start in range(0,len(queries),batch):
        sl=slice(start,start+batch)
        args=data.batch(queries[sl],rows[sl],b_mismatch=mismatch[sl] if intervention=='b_mismatch' else None,
            all_mismatch=mismatch[sl] if intervention=='all_mismatch' else None)
        p=model(*args);out.append(p['logits'].cpu().numpy());gg.append(args[-1].cpu().numpy())
        if 'score_a' in p:aa.append(p['score_a'].cpu().numpy());bb.append(p['score_b'].cpu().numpy())
    r={'logits':np.concatenate(out),'global_logits':np.concatenate(gg),'targets':y,'query_ids':data.ids[queries],'candidate_ids':data.ids[rows]}
    if aa:r.update(score_a=np.concatenate(aa),score_b=np.concatenate(bb))
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',default='sub-01');p.add_argument('--epochs',type=int,default=20)
    a=p.parse_args();torch.set_num_threads(4);data=Banks(a.subject)
    config={'seed':2026,'epochs':a.epochs,'patience':5,'batch_size':32,'lr':.001,'weight_decay':.0001,
        'source_sha256':sha(__file__),'bank':data.manifest,'scope':'validation-only bounded native-temporal mechanism repair'}
    path=data.base/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:save(path,config)
    vr,vy=candidate_rows(data.labels,data.val,data.val,seed=20261103)
    for mode in MODES:
        dest=data.base/mode
        if (dest/'summary.json').exists():continue
        if dest.exists() and list(dest.iterdir()):raise ValueError('Inspect partial run '+str(dest))
        dest.mkdir(exist_ok=True);seed_all(2026);model=NativeReadout(data,mode).cuda()
        initial=hashlib.sha256()
        for n,v in sorted(model.state_dict().items()):initial.update(n.encode());initial.update(v.detach().cpu().numpy().tobytes())
        initial_sha256=initial.hexdigest()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
        checks={};t,emb,v,g=data.batch(data.val[:8],vr[:8]);model.eval()
        with torch.no_grad():
            alpha=torch.full((8,10,2,36),1/36,device='cuda')
            uniform=model.read(alpha,t[:,0],v).mean(-1);direct=torch.einsum('bd,bkd->bk',emb[:,0],v)
            error=float((uniform-direct).abs().max());assert error<1e-5
            checks['uniform_readout_max_error']=error
        if mode=='cross':
            result=model(t,emb,v,g);alter=t.clone();alter[:,1]=alter.roll(1,0)[:,1]
            other=model(alter,emb,v,g);assert torch.equal(result['alpha'],other['alpha'])
            loss=F.cross_entropy(result['logits'],torch.tensor(vy[:8],device='cuda'));loss.backward()
            assert float(model.query.weight.grad.norm())>0;model.zero_grad()
            checks['A_selection_B_independent']=True;checks['query_gradient_nonzero']=True
            assert all(p.grad is None for p in model.projector.parameters())
        seed_all(2026);best=-1.;bad=0;history=[];begin=time.time()
        for epoch in range(a.epochs):
            rows,y=candidate_rows(data.labels,data.train,data.train,seed=2026+epoch)
            cr,cy=candidate_rows(data.labels,data.train,data.train,cross=True,seed=2026+epoch)
            rng=np.random.default_rng(2026+epoch);mix=rng.permutation(len(y))[:len(y)//2];rows[mix]=cr[mix];y[mix]=cy[mix]
            order=rng.permutation(len(y));model.train();losses=[]
            for start in range(0,len(order),32):
                ix=order[start:start+32];result=model(*data.batch(data.train[ix],rows[ix]))
                loss=F.cross_entropy(result['logits'],torch.tensor(y[ix],device='cuda'))
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
            pred=predict(model,data,data.val,vr,vy);acc=classification_metrics(pred['logits'],vy)['top1']
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_top1':acc};history.append(row);print(mode,json.dumps(row),flush=True)
            if acc>best:
                best=acc;bad=0;torch.save({'model':model.state_dict(),'config':config,'mode':mode,'epoch':epoch+1,'initial_state_sha256':initial_sha256},dest/'model.pt')
            else:bad+=1
            save(dest/'history.json',history)
            if bad>=5:break
        model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True)['model'])
        summary={'subject':a.subject,'mode':mode,'seed':2026,'config':config,'checks':checks,'elapsed_seconds':time.time()-begin,
            'initial_state_sha256':initial_sha256,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'protocols':{}}
        for cross,name in [(False,'within10'),(True,'cross10')]:
            rows,y=candidate_rows(data.labels,data.val,data.val,cross=cross,seed=20261103)
            pred=predict(model,data,data.val,rows,y);np.savez_compressed(dest/(name+'.npz'),**pred)
            summary['protocols'][name]={'model':classification_metrics(pred['logits'],y),'global':classification_metrics(pred['global_logits'],y)}
            if 'score_a' in pred:summary['protocols'][name].update(agreement(pred['score_a'],pred['score_b'],y))
            if mode=='cross' and name=='within10':
                for intervention in ['b_mismatch','all_mismatch']:
                    ip=predict(model,data,data.val,rows,y,intervention=intervention);np.savez_compressed(dest/(intervention+'.npz'),**ip)
                    summary['protocols'][intervention]=classification_metrics(ip['logits'],y)
        save(dest/'summary.json',summary);del model,optimizer;gc.collect();torch.cuda.empty_cache()
        print('NATIVE_MODE_COMPLETE',mode,summary['protocols']['within10']['model']['top1'],flush=True)
if __name__=='__main__':main()
