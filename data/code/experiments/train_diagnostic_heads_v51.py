"""Fixed-recipe heads for all participants; train/development EEG only.

The selected head is exactly v48 unary. Isotropic and raw-diagonal geometry
are matched-parameter controls, not candidates selected on final outcomes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,gc,time
import numpy as np
import torch
from torch.nn import functional as F
from native_temporal_screen import ROOT,sha,save
from protocol import category_split
from load_visual_features import load_features
from cross_trial_likelihood_v6 import ridge
from discriminant_evidence_v40 import covariance
from evidence_graph_v48 import train

MODES=['adapter_joint','adapter_diagonal','adapter_isotropic']

def load_training(subject,visual,canonical):
    base=ROOT/'results/all_subject_backbones_v42'/subject/'seed2028'
    banks={a:dict(np.load(base/a/'compact.npz')) for a in ['NICE','ATM']}
    models={a:dict(np.load(base/a/'model.npz')) for a in banks}
    a=banks['NICE'];ids=a['image_ids'];labels=a['class_ids'].astype(str);original=a['original_indices']
    np.testing.assert_array_equal(ids,canonical[original]);np.testing.assert_array_equal(ids,banks['ATM']['image_ids'])
    all_labels=np.load(ROOT/'processed'/subject/'train_identities.npz')['class_id'];splits=category_split(all_labels)
    tr=np.flatnonzero(np.isin(labels,splits['train']));assert len(tr)==12540
    classes=np.random.default_rng(2026).permutation(splits['train']);which={c:i%5 for i,c in enumerate(classes)}
    fold=np.array([which[c] for c in labels[tr]]);xx=[];mm=[]
    for arch in ['NICE','ATM']:
        m=models[arch];x=banks[arch]['group_latents'].astype(float)
        assert set(m['fit_classes'])==set(splits['train'])
        mu=(visual[original]@m['visual_to_neural']+m['offset']+m['residual_mean']).reshape(-1,2,64).mean(1)
        oof=np.empty((len(tr),64));vt=torch.tensor(visual[original][tr],dtype=torch.float64,device='cuda')
        xt=torch.tensor(x[tr].mean(1),dtype=torch.float64,device='cuda')
        for f in range(5):
            fit=fold!=f;wf,bf,_=ridge(vt[fit],xt[fit]);oof[~fit]=(vt[~fit]@wf+bf).cpu().numpy()
        mu[tr]=oof+m['residual_mean'].reshape(2,64).mean(0)
        xx.append(x);mm.append(mu);del vt,xt
    x=np.concatenate(xx,-1);mu=np.concatenate(mm,-1);delta=(x[:,0]-x[:,1])/2;bias=delta[tr].mean(0)
    x[:,0]-=bias;x[:,1]+=bias;c=covariance(delta[tr],'joint')
    groups=np.array([np.flatnonzero(labels==cl) for cl in sorted(splits['train'])]);assert groups.shape==(1254,10)
    return {'x_raw':x,'mu_raw':mu,'covariance':c,'group_bias':bias,'train_groups':groups,'image_ids':ids,'class_ids':labels,'tr':tr},base

def transform(x,mu,c,scale):
    chol=np.linalg.cholesky(c)
    a=np.linalg.solve(chol,np.asarray(x).reshape(-1,128).T).T.reshape(x.shape)/scale
    b=np.linalg.solve(chol,np.asarray(mu).reshape(-1,128).T).T.reshape(mu.shape)/scale
    return a.astype('float32'),b.astype('float32')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',default=[f'sub-{i:02}' for i in range(1,11)]);args=p.parse_args()
    lock=ROOT/'results/diagnostic_line_lock_v50.json';record=json.loads(lock.read_text());assert record['allow_final_evaluation'] and not record['allow_final_data_for_method_selection']
    root=ROOT/'results/diagnostic_heads_v51';root.mkdir(exist_ok=True);source=sha(__file__);torch.set_num_threads(1)
    configuration={'source_sha256':source,'method_source_sha256':sha(ROOT/'code/experiments/evidence_graph_v48.py'),'lock_sha256':sha(lock),'modes':MODES,'epochs':12,'seed':20261104,'head_fit':'1254 training concepts, OOF visual templates; frozen encoders saw these1254 concepts','final_data_used':False}
    path=root/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==configuration
    else:save(path,configuration)
    visual,canonical=load_features('train');visual=F.normalize(torch.tensor(visual,dtype=torch.float32),dim=-1).numpy().astype(float)
    for subject in args.subjects:
        folder=root/subject
        if (folder/'manifest.json').exists():continue
        folder.mkdir(exist_ok=True);data,base=load_training(subject,visual,canonical);geometry={};initials=[]
        for mode in MODES:
            c=data['covariance'].copy()
            if mode=='adapter_diagonal':c=np.diag(np.diag(c))
            if mode=='adapter_isotropic':c=np.eye(128)*np.trace(c)/128
            x,mu=transform(data['x_raw'],data['mu_raw'],c,1.);scale=float(np.sqrt(np.mean(x[data['tr']]**2)))
            prepared={'x':x/scale,'mu':mu/scale,'train_groups':data['train_groups'],'image_ids':data['image_ids']}
            sub=folder/mode;sub.mkdir(exist_ok=True)
            if (sub/'unary.pt').exists():raise FileExistsError('Review partial head run before resuming')
            model,initial=train(prepared,'unary',sub,12);initials.append(initial)
            geometry[mode+'_covariance']=c;geometry[mode+'_scale']=np.array(scale)
            del model,prepared,x,mu;gc.collect();torch.cuda.empty_cache()
            print('HEAD_COMPLETE',subject,mode,flush=True)
        assert len(set(initials))==1
        geometry['group_bias']=data['group_bias'];geometry['fit_image_ids']=data['image_ids'][data['tr']]
        np.savez_compressed(folder/'geometry.npz',**geometry)
        save(folder/'manifest.json',{'source_sha256':source,'lock_sha256':sha(lock),'method_source_sha256':configuration['method_source_sha256'],
            'geometry_sha256':sha(folder/'geometry.npz'),'head_sha256':{m:sha(folder/m/'unary.pt') for m in MODES},'initial_sha256':initials[0],
            'mapping_sha256':{a:sha(base/a/'model.npz') for a in ['NICE','ATM']},'encoder_sha256':{a:sha(base/a/'model.pt') for a in ['NICE','ATM']},
            'final_data_used':False})
        del data;gc.collect();torch.cuda.empty_cache()
        save(root/'progress.json',{'completed':[s for s in [f'sub-{i:02}' for i in range(1,11)] if (root/s/'manifest.json').exists()]})
    assert sha(__file__)==source;print('ALL_REQUESTED_HEADS_COMPLETE',flush=True)

if __name__=='__main__':main()
