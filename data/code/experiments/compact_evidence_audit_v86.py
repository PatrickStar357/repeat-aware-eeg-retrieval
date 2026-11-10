"""Bounded compact-scorer/source audit on the same frozen recordings/models.

The compact three-score variant was already evaluated in the complete matrix.
This run adds matched NLL/source/single-encoder controls, not a parameter search.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,gc,time
import numpy as np
import torch
from native_temporal_screen import ROOT,sha,save
from evaluation_banks_v62 import vision,bank,load_training,tasks
from final_choice_screen_v75 import mlp_scores
from discriminant_evidence_v40 import covariance
from score_stacking_v24 import fit_stack
from evaluate_diagnostic_v52 import apply_stack
from closeout_experiments_v64 import fusion_controls
from metric_suite_v23 import evaluate_metrics

SUBJECTS=[f'sub-{i:02}' for i in range(1,11)];SEEDS=[2028,2029,2030]

def geometry(data,training,models,tr):
 xx=[];mu=[];dd=[];means=[];res=np.zeros((128,128));cc=[]
 for j,a in enumerate(['NICE','ATM']):
  m=models[a];x=training[a]['group_latents'][tr];dd.append((x[:,0]-x[:,1])/2);means.append(x.mean(1));xx.append(data[a+'_latent'].mean(1));mu.append((data['visual'].astype(float)@m['visual_to_neural']+m['offset']+m['residual_mean']).reshape(-1,2,64).mean(1))
  c=m['residual_covariance'];res[j*64:(j+1)*64,j*64:(j+1)*64]=(c[:64,:64]+c[64:,64:]+c[:64,64:]+c[64:,:64])/4
  cc.append(covariance(dd[-1],'joint'))
 repeat=covariance(np.concatenate(dd,-1),'block');perm=np.r_[np.random.default_rng(20261104).permutation(64),64+np.random.default_rng(20261105).permutation(64)]
 covs={'compact':repeat,'compact_diagonal':np.diag(np.diag(repeat)),'compact_isotropic':np.eye(128),'compact_total':covariance(np.concatenate(means,-1),'block'),'compact_residual':res,'compact_permuted':repeat[perm][:,perm]}
 return np.concatenate(xx,-1),np.concatenate(mu,-1),{k:np.linalg.inv(c) for k,c in covs.items()},cc

def main():
 torch.set_num_threads(1);root=ROOT/'results/compact_audit_v86';root.mkdir(exist_ok=True)
 cfg={'source_sha256':sha(__file__),'subjects':SUBJECTS,'seeds':SEEDS,'mainline_candidate':'existing hybrid_block_mlp, three shared NLL coefficients','fixed_variants':['compact','anchor_nll','compact_diagonal','compact_isotropic','compact_total','compact_residual','compact_permuted','compact_NICE','compact_ATM','compact_Brier','compact_equal_temperature','compact_equal_uncalibrated'],'no_backbone_or_gallery_training':True,'no_new_hyperparameter_search':True,'history':'Latest user asks to meet most-first/rest-second core performance and study16 reference papers. All earlier test access remains disclosed.'}
 lock=root/'configuration.json'
 if lock.exists():assert json.loads(lock.read_text())==cfg
 else:save(lock,cfg)
 v=vision()
 for se in SEEDS:
  for s in SUBJECTS:
   folder=root/s/f'seed{se}';folder.mkdir(parents=True,exist_ok=True)
   if (folder/'summary.json').exists():continue
   data=bank(s,se,v);training,models,tr,split=load_training(s,se,v);x,mu,precision,single=geometry(data,training,models,tr);records={};parameters={};eqparams={}
   for taskname,task in tasks(data).items():
    q=task['queries'];rows=task['rows'];n=task['calibration_n'];y=task['targets'];diff=x[q,None]-mu[rows];anchor=mlp_scores(data,task,s,se)
    comp={k:[-.5*np.einsum('nkd,de,nke->nk',diff,p,diff,optimize=True)]+anchor for k,p in precision.items()};comp['anchor_nll']=anchor
    for j,a in enumerate(['NICE','ATM']):
     z=diff[:,:,j*64:(j+1)*64];local=-.5*np.einsum('nkd,de,nke->nk',z,np.linalg.inv(single[j]),z,optimize=True);comp['compact_'+a]=[local,anchor[j]]
    comp['compact_Brier']=comp['compact'];records[taskname]={};parameters[taskname]={}
    for method,scores in comp.items():
     objective='brier' if method=='compact_Brier' else 'nll'
     if n:_,pa=fit_stack([z[:n] for z in scores],y[:n],np.ones(n,bool),objective)
     else:pa=parameters['cross10'][method]
     z=apply_stack([z[n:] for z in scores],pa);parameters[taskname][method]=pa;records[taskname][method]=evaluate_metrics(z,y[n:])[0]
     np.savez_compressed(folder/f'{taskname}_{method}.npz',logits=z,targets=y[n:],query_ids=data['image_ids'][q[n:]],candidate_ids=data['image_ids'][rows[n:]],concepts=data['class_ids'][q[n:]])
     if method=='compact':
      old=np.load(ROOT/f'results/final_hybrid_v76/{s}/seed{se}/{taskname}_hybrid_block_mlp.npz');np.testing.assert_allclose(z,old['logits'],atol=2e-5,rtol=1e-6)
    equal,pa=fusion_controls(comp['compact'],y,n,eqparams['cross10'] if n==0 else None);eqparams[taskname]=pa
    for old,method in [('block_equal_temperature','compact_equal_temperature'),('block_equal_uncalibrated','compact_equal_uncalibrated')]:
     z=equal[old];records[taskname][method]=evaluate_metrics(z,y[n:])[0];np.savez_compressed(folder/f'{taskname}_{method}.npz',logits=z,targets=y[n:])
    print('COMPACT_AUDIT',s,se,taskname,{m:round(records[taskname][m]['top1'],4) for m in ['compact','anchor_nll','compact_diagonal','compact_total']},flush=True)
   save(folder/'summary.json',{'configuration_sha256':sha(lock),'results':records,'parameters':parameters,'equal_parameters':eqparams})
   del data,training,models;gc.collect();torch.cuda.empty_cache()
   done=[{'subject':a,'seed':b} for b in SEEDS for a in SUBJECTS if (root/a/f'seed{b}/summary.json').exists()];save(root/'progress.json',{'completed':done,'expected':30})
 print('COMPACT_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
