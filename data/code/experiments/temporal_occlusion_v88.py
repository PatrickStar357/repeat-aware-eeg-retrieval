"""Fixed-window perturbation of the final scorer; ten people, seed2028.

Zeroing is in training-normalized EEG coordinates. This is sensitivity of a
fixed model, not a causal claim about cognition or a separately trained decoder.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,gc,time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax
from native_temporal_screen import ROOT,sha,save
from evaluation_banks_v62 import vision,bank,load_training,model_root,tasks
from compact_evidence_audit_v86 import geometry
from closeout_experiments_v64 import GalleryMLP
from train_all_backbones_v42 import Nice,forward
from atm_control_v9 import author_classes
from protocol import repeat_groups
from evaluate_diagnostic_v52 import apply_stack
from metric_suite_v23 import evaluate_metrics

@torch.no_grad()
def main():
 torch.set_num_threads(1);out=ROOT/'results/temporal_occlusion_v88';out.mkdir(exist_ok=True);v=vision();ns,_=author_classes();windows=[('unmasked',None,None)]+[(f'{a}-{a+200}',a//4,(a+200)//4) for a in range(0,1000,200)]
 save(out/'configuration.json',{'source_sha256':sha(__file__),'seed':2028,'subjects':10,'windows_ms':windows,'no_retraining_or_recalibration':True,'interpretation':'Frozen-model sensitivity under zeroing in normalized coordinates; not causal functional localization.'})
 for i in range(1,11):
  s=f'sub-{i:02}';dest=out/s;dest.mkdir(exist_ok=True)
  if (dest/'summary.json').exists():continue
  data=bank(s,2028,v);train,maps,tr,_=load_training(s,2028,v);x,mu,prec,_=geometry(data,train,maps,tr);task=tasks(data)['within10'];q=task['queries'][1000:];rows=task['rows'][1000:];y=task['targets'][1000:];ids=np.load(ROOT/'processed'/s/'train_identities.npz');selected=data['original_indices'][q];raw=np.load(ROOT/'processed'/s/'train_eeg.npy',mmap_mode='r');pairs=[repeat_groups(z) for z in ids['session_id'][selected]];anchors={};models={}
  for a in ['NICE','ATM']:
   ck=torch.load(ROOT/f'results/closeout_experiments_v64/{s}/seed2028/nearg/{a}/mlp.pt',map_location='cpu',weights_only=True);mlp=GalleryMLP().cuda();mlp.load_state_dict(ck['model']);mlp.eval();anchors[a]=np.concatenate([F.normalize(mlp(torch.tensor(data['visual'][j:j+256],dtype=torch.float32,device='cuda')),dim=-1).cpu().numpy() for j in range(0,3200,256)]);del mlp,ck
   ck=torch.load(model_root(s,2028)/a/'model.pt',map_location='cpu',weights_only=True);m=Nice() if a=='NICE' else ns['ATMS']();m.load_state_dict(ck['model']);models[a]=m.cuda().eval()
  pa=json.loads((ROOT/f'results/compact_audit_v86/{s}/seed2028/summary.json').read_text())['parameters']['within10']['compact'];records={};start=time.time()
  for name,lo,hi in windows:
   lat={a:[] for a in models};gl={a:[] for a in models}
   for j in range(0,len(q),32):
    rr=np.asarray(raw[selected[j:j+32]],dtype=np.float32).copy()
    if lo is not None:rr[:,:,:,lo:hi]=0
    p=pairs[j:j+32];aa=np.stack([r[z[0]].mean(0) for r,z in zip(rr,p)]);bb=np.stack([r[z[1]].mean(0) for r,z in zip(rr,p)]);xx=torch.tensor(np.stack([aa,bb,rr.mean(1)],1).reshape(-1,63,250),device='cuda')
    for a,m in models.items():
     if a=='NICE':tt=m.encoder[0](xx[:,None]).half().float();z=F.normalize(m.projector(tt.flatten(1)),dim=-1)
     else:z=F.normalize(forward(m,xx,a),dim=-1)
     z=z.reshape(len(rr),3,1024).cpu().numpy();gl[a].append(z[:,2]);mp=maps[a];lat[a].append(((z[:,:2].astype(float).reshape(-1,1024)-mp['neural_center'])@mp['neural_basis']/mp['neural_scale']).reshape(-1,2,64).mean(1))
   latent=np.concatenate([np.concatenate(lat[a]) for a in models],-1);diff=latent[:,None]-mu[rows];scores=[-.5*np.einsum('nkd,de,nke->nk',diff,prec['compact'],diff,optimize=True)]
   for a in models:
    zg=np.concatenate(gl[a]);scores.append(.25*np.einsum('nd,nkd->nk',zg,data['visual'][rows])+.75*np.einsum('nd,nkd->nk',zg,anchors[a][rows]))
   logits=apply_stack(scores,pa);records[name]=evaluate_metrics(logits,y)[0];np.savez_compressed(dest/(name+'.npz'),logits=logits,targets=y,query_ids=data['image_ids'][q])
   if name=='unmasked':
    base=np.load(ROOT/f'results/compact_audit_v86/{s}/seed2028/within10_compact.npz');np.testing.assert_allclose(logits,base['logits'],atol=3e-5,rtol=1e-5)
   print('OCCLUSION',s,name,round(records[name]['top1'],4),flush=True)
  save(dest/'summary.json',{'source_sha256':sha(__file__),'results':records,'elapsed_seconds':time.time()-start});del models,train,data,raw;gc.collect();torch.cuda.empty_cache()
 save(out/'summary.json',{'source_sha256':sha(__file__),'subjects':{f'sub-{i:02}':json.loads((out/f'sub-{i:02}/summary.json').read_text())['results'] for i in range(1,11)},'complete':True})
 print('TEMPORAL_OCCLUSION_COMPLETE',flush=True)
if __name__=='__main__':main()
