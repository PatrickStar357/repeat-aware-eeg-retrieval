"""Compact frozen-feature banks for closeout, independent of method fitting.

Includes calibration100, fine-test200, and the official200 R4 gallery. Existing
raw/processed data and frozen checkpoints are never changed. The bank is about
35MB/subject/seed, avoiding duplicate EEG arrays or image downloads.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,gc,time
import numpy as np
import torch
from torch.nn import functional as F
from native_temporal_screen import ROOT,sha,save
from protocol import category_split,repeat_groups,candidate_rows
from load_visual_features import load_features
from train_all_backbones_v42 import Nice,forward
from atm_control_v9 import author_classes

def model_root(subject,seed):
    parent='all_subject_backbones_v42' if seed==2028 else 'seed_replication_v61'
    return ROOT/'results'/parent/subject/f'seed{seed}'

def vision():
    result={}
    for task in ['train','test']:
        v,ids=load_features(task);v=F.normalize(torch.tensor(v,dtype=torch.float32),dim=-1).numpy().astype(float);result[task]=(v,ids)
    return result

def load_training(subject,seed,v):
    base=model_root(subject,seed);banks={a:dict(np.load(base/a/'compact.npz')) for a in ['NICE','ATM']};models={a:dict(np.load(base/a/'model.npz')) for a in banks}
    ident=np.load(ROOT/'processed'/subject/'train_identities.npz');splits=category_split(ident['class_id']);a=banks['NICE'];tr=np.flatnonzero(np.isin(a['class_ids'],splits['train']))
    assert len(tr)==12540 and set(models['NICE']['fit_classes'])==set(splits['train'])
    np.testing.assert_array_equal(a['image_ids'],v['train'][1][a['original_indices']])
    return banks,models,tr,splits

@torch.no_grad()
def encode(subject,seed,v):
    folder=ROOT/'processed'/subject;identity=np.load(folder/'train_identities.npz');splits=category_split(identity['class_id'])
    cal=np.flatnonzero(np.isin(identity['class_id'],splits['calibration']));test=np.flatnonzero(np.isin(identity['class_id'],splits['fine_test']));selected=np.r_[cal,test]
    np.testing.assert_array_equal(identity['image_id'][selected],v['train'][1][selected]);assert len(cal)==1000 and len(test)==2000
    official=np.load(folder/'test_R4_identities.npz');np.testing.assert_array_equal(official['image_id'],v['test'][1])
    paths=[folder/'train_eeg.npy',folder/'test_R4_eeg.npy'];raw=[np.load(p,mmap_mode='r') for p in paths]
    indices=[selected,np.arange(200)];sessions=[identity['session_id'][selected],official['session_id']]
    base=model_root(subject,seed);ns,_=author_classes();data={};cost={}
    for arch in ['NICE','ATM']:
        ck=torch.load(base/arch/'model.pt',map_location='cpu',weights_only=True);m=np.load(base/arch/'model.npz');model=Nice() if arch=='NICE' else ns['ATMS']();model.load_state_dict(ck['model']);model.cuda().eval()
        latent=[];glob=[];start_time=time.time()
        for source,ix,ss in zip(raw,indices,sessions):
            pairs=[repeat_groups(s) for s in ss]
            for start in range(0,len(ix),32):
                rr=np.asarray(source[ix[start:start+32]],dtype=np.float32);pp=pairs[start:start+32]
                a=np.stack([r[p[0]].mean(0) for r,p in zip(rr,pp)]);b=np.stack([r[p[1]].mean(0) for r,p in zip(rr,pp)])
                x=torch.tensor(np.stack([a,b,rr.mean(1)],axis=1).reshape(-1,63,250),device='cuda')
                if arch=='NICE':t=model.encoder[0](x[:,None]).half().float();z=F.normalize(model.projector(t.flatten(1)),dim=-1)
                else:z=F.normalize(forward(model,x,arch),dim=-1)
                z=z.reshape(len(rr),3,1024).cpu().numpy();glob.append(z[:,2]);latent.append(((z[:,:2].astype(float).reshape(-1,1024)-m['neural_center'])@m['neural_basis']/m['neural_scale']).reshape(-1,2,64))
        data[arch+'_latent']=np.concatenate(latent);data[arch+'_global']=np.concatenate(glob);cost[arch]=time.time()-start_time
        del model,ck;gc.collect();torch.cuda.empty_cache()
    data.update(image_ids=np.r_[identity['image_id'][selected],official['image_id']],class_ids=np.r_[identity['class_id'][selected],official['class_id']],original_indices=np.r_[selected,np.full(200,-1)],visual=np.r_[v['train'][0][selected],v['test'][0]].astype('float32'))
    return data,cost

def bank(subject,seed,v):
    folder=ROOT/'results/evaluation_banks_v62'/subject;folder.mkdir(parents=True,exist_ok=True);path=folder/f'seed{seed}.npz';meta=path.with_suffix('.json');base=model_root(subject,seed)
    sources={a:sha(base/a/'model.pt') for a in ['NICE','ATM']}
    if meta.exists():
        record=json.loads(meta.read_text());assert record['builder_sha256']==sha(__file__) and record['checkpoints']==sources and sha(path)==record['bank_sha256'];return dict(np.load(path))
    if path.exists():raise FileExistsError('Review incomplete evaluation bank')
    data,cost=encode(subject,seed,v);np.savez_compressed(path,**data)
    save(meta,{'builder_sha256':sha(__file__),'checkpoints':sources,'bank_sha256':sha(path),'preprocessing_sha256':sha(ROOT/'processed'/subject/'preprocessing.npz'),
        'partition_rows':{'calibration':[0,1000],'fine_test':[1000,3000],'official200':[3000,3200]},'source_inference_seconds':cost,'not_method_selection':True})
    print('COMPACT_EVALUATION_BANK',subject,seed,round(path.stat().st_size/1024**2,1),'MiB',flush=True);return data

def tasks(data):
    labels=data['class_ids'];qcal=np.arange(1000);fine=np.arange(1000,3000);out={}
    for cross,name in [(False,'within10'),(True,'cross10')]:
        rc,yc=candidate_rows(labels,qcal,qcal,cross=cross,seed=20261103);rt,yt=candidate_rows(labels,fine,fine,cross=cross,seed=20261103)
        out[name]={'queries':np.r_[qcal,fine],'rows':np.r_[rc,rt],'targets':np.r_[yc,yt],'calibration_n':1000,'test_n':2000}
    out['official200']={'queries':np.arange(3000,3200),'rows':np.tile(np.arange(3000,3200),(200,1)),'targets':np.arange(200),'calibration_n':0,'test_n':200}
    return out
