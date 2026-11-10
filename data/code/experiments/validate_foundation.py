"""Real-data prerequisite audit and a strictly split NICE-family baseline pilot.

The architecture is extracted from the pinned official NICE source (class
definitions only), avoiding its GPU/environment/argument parsing side effects.
RN50 output dimension and train-only z-scoring are controlled adaptations;
this is not the authors' published ViT/whitened checkpoint protocol.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sqlite3,time,random,sys,re
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from protocol import category_split,candidate_rows,repeat_groups
from load_visual_features import load_features
ROOT=Path(__file__).resolve().parents[2]

def official_classes():
    source=ROOT/'code/upstream/NICE-EEG/nice_stand.py'
    names={'PatchEmbedding','ResidualAdd','FlattenHead','Enc_eeg','Proj_eeg'}
    tree=ast.parse(source.read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
    if len(nodes)!=5:raise RuntimeError('Pinned NICE class interface changed')
    ns={'nn':nn,'Tensor':Tensor,'Rearrange':Rearrange}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    return ns,hashlib.sha256(source.read_bytes()).hexdigest()

def audit(subject,run_name=''):
    d=ROOT/'processed'/subject
    identity=np.load(d/'train_identities.npz');p=np.load(d/'preprocessing.npz')
    labels=identity['class_id'].astype(str);splits=category_split(labels)
    assert set(p['preprocessing_fit_classes'])==set(splits['train'])
    assert len(set(identity['image_id']))==16540
    assert len(set(identity['recording_id'].ravel()))==66160
    assert np.all(np.unique(labels,return_counts=True)[1]==10)
    for i in range(len(labels)):
        a,b=repeat_groups(identity['session_id'][i])
        assert not set(identity['session_id'][i,a])&set(identity['session_id'][i,b])
    eeg=np.load(d/'train_eeg.npy',mmap_mode='r')
    train_idx=np.flatnonzero(np.isin(labels,splits['train']))
    sums=np.zeros(63);squares=np.zeros(63);count=0
    for start in range(0,len(train_idx),32):
        x=np.asarray(eeg[train_idx[start:start+32]],dtype=np.float32)
        assert np.isfinite(x).all()
        sums+=x.sum(axis=(0,1,3),dtype=np.float64)
        squares+=(x.astype(np.float64)**2).sum(axis=(0,1,3));count+=x.shape[0]*4*250
    m=sums/count;sd=np.sqrt(squares/count-m*m)
    assert abs(m).max()<1e-3 and abs(sd-1).max()<1e-3
    dest=ROOT/'results/foundation'/subject
    if run_name:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',run_name):raise ValueError('run-name must be a simple directory name')
        dest=dest/run_name
    dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'fixed_protocol.npz',**splits,split_seed=20261102)
    result={'subject':subject,'shape':list(eeg.shape),'dtype':str(eeg.dtype),
            'partitions':{k:len(v) for k,v in splits.items()},'unique_images':len(labels),
            'unique_repetitions':66160,'session_disjoint_A_B_images':len(labels),
            'normalized_train_mean_abs_max':float(abs(m).max()),
            'normalized_train_std_error_max':float(abs(sd-1).max()),
            'preprocessing_sha256':hashlib.sha256((d/'preprocessing.npz').read_bytes()).hexdigest()}
    return d,dest,identity,labels,splits,result

@torch.no_grad()
def encode(enc,proj,eeg,indices,device,batch=32):
    enc.eval();proj.eval();all_z=[]
    for start in range(0,len(indices),batch):
        x=torch.tensor(np.array(eeg[indices[start:start+batch]],dtype=np.float32),device=device)
        with torch.autocast('cuda',enabled=device=='cuda'):
            z=proj(enc(x.reshape(-1,1,63,250)))
        all_z.append(z.float().reshape(len(x),4,-1).mean(1).cpu())
    return F.normalize(torch.cat(all_z),dim=-1)

def candidate_metrics(z,visual,labels,indices):
    out={};pred={}
    for cross,name in [(False,'within10'),(True,'cross10')]:
        rows,y=candidate_rows(labels,indices,indices,10,cross,20261103)
        score=(z[:,None,:]*visual[rows]).sum(-1)
        a=score.argmax(1).numpy();correct=a==y
        out[name]=float(correct.mean());pred[name+'_correct']=correct
        pred[name+'_candidate_rows']=rows;pred[name+'_target']=y
        pred[name+'_scores']=score.numpy()
    return out,pred

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject',default='sub-01');ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--audit-only',action='store_true');ap.add_argument('--run-name',default='');args=ap.parse_args()
    if args.epochs<1 or args.batch_size<2:raise ValueError('Need positive epochs and batch size >=2')
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    d,dest,identity,labels,splits,report=audit(args.subject,getattr(args,'run_name',''))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report['environment']={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':device,
        'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None}
    if args.audit_only:
        (dest/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));return
    if (dest/'nice_rn50_strict_best.pt').exists():raise FileExistsError('Run already exists; choose a new --run-name to preserve results')
    if device!='cuda':raise RuntimeError('CUDA prerequisite not satisfied')
    # Execute a real CUDA forward/backward before any training.
    probe=torch.randn(128,128,device=device,requires_grad=True);(probe@probe.T).square().mean().backward()
    assert torch.isfinite(probe.grad).all();del probe;torch.cuda.empty_cache()
    ns,source_hash=official_classes();enc=ns['Enc_eeg']().to(device);proj=ns['Proj_eeg'](proj_dim=1024).to(device)
    visual,ids=load_features('train');assert np.array_equal(ids,identity['image_id'])
    visual=F.normalize(torch.tensor(visual,dtype=torch.float32),dim=-1)
    train_idx=np.flatnonzero(np.isin(labels,splits['train']));val_idx=np.flatnonzero(np.isin(labels,splits['validation']))
    eeg=np.load(d/'train_eeg.npy',mmap_mode='r')
    log_scale=nn.Parameter(torch.tensor(np.log(1/.07),dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(list(enc.parameters())+list(proj.parameters())+[log_scale],lr=2e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');history=[];best=-1.;start_time=time.time()
    rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        enc.train();proj.train();order=rng.permutation(train_idx);losses=[];begin=time.time()
        for start in range(0,len(order),args.batch_size):
            idx=order[start:start+args.batch_size]
            if len(idx)<2:continue
            x=torch.tensor(np.array(eeg[idx],dtype=np.float32),device=device)
            v=visual[idx].to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                z=proj(enc(x.reshape(-1,1,63,250))).reshape(len(x),4,-1).mean(1)
                z=F.normalize(z,dim=-1);logits=log_scale.exp().clamp(max=100)*z@v.T
                y=torch.arange(len(idx),device=device)
                loss=(F.cross_entropy(logits,y)+F.cross_entropy(logits.T,y))/2
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        z=encode(enc,proj,eeg,val_idx,device)
        metrics,preds=candidate_metrics(z,visual,labels,val_idx)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.time()-begin,**metrics};history.append(row)
        print(json.dumps(row),flush=True)
        # Validation-only selection; no final-test or official-test accuracy is queried here.
        if metrics['within10']>best:
            best=metrics['within10'];best_row=row
            torch.save({'encoder':enc.state_dict(),'projector':proj.state_dict(),
                'encoder_fit_classes':splits['train'].tolist(),'source_sha256':source_hash,
                'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'seed':args.seed,'validation_selection':row,'projection_dim':1024},dest/'nice_rn50_strict_best.pt')
            np.savez(dest/'validation_predictions.npz',**preds,query_indices=val_idx,image_id=ids[val_idx])
        (dest/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    report.update({'protocol':'new category-disjoint RN50 adapted NICE; R4 mean of per-repetition embeddings',
        'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,'best_validation':best_row,
        'optimizer':{'name':'AdamW','lr':2e-4,'weight_decay':.01,'mixed_precision':True},
        'source_sha256':source_hash,'elapsed_seconds':time.time()-start_time,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,
        'parameter_count':sum(p.numel() for p in enc.parameters())+sum(p.numel() for p in proj.parameters()),
        'official_replication_status':'not a published-protocol replication',
        'fine_test_status':'not evaluated; reserved for locked full experiment',
        'calibration_status':'not used in training or selection'})
    (dest/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
