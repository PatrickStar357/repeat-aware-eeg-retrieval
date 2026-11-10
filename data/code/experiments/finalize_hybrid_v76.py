"""Fixed final small-screen winner and mechanism controls; no new search."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,datetime,gc,time
import numpy as np
import torch
from native_temporal_screen import ROOT,save,sha
from evaluation_banks_v62 import vision,bank,load_training,tasks
from closeout_experiments_v64 import raw_components,fusion_controls
from final_choice_screen_v75 import mlp_scores,build_components
from score_stacking_v24 import fit_stack
from evaluate_diagnostic_v52 import apply_stack
from metric_suite_v23 import evaluate_metrics

SUBJECTS=[f'sub-{i:02}' for i in range(1,11)];SEEDS=[2028,2029,2030]
SPECS={'hybrid_nll':('full','nll'),'hybrid_no_block':('no_block','nll'),'hybrid_no_repeat':('no_repeat','nll'),'hybrid_block_mlp':('block_mlp','nll'),'hybrid_brier':('full','brier')}

def sets(metric,mlp):return {'full':metric+mlp,'no_block':metric[:4]+mlp,'no_repeat':metric[:2]+mlp,'block_mlp':[metric[4]]+mlp}

def develop_controls(v,root):
 for s in SUBJECTS[:3]:
  dest=root/s/'development_controls.json'
  if dest.exists():continue
  comp,y,labels=build_components(s,v);allsets=sets(comp['metric'],comp['mlp']);unique=sorted(set(labels[:1000]));order=np.random.default_rng(20261105).permutation(unique);lookup={c:i%5 for i,c in enumerate(order)};fold=np.array([lookup[c] for c in labels[:1000]]);results={}
  for name,(kind,obj) in SPECS.items():
   cc=allsets[kind];pred=np.empty((1000,10))
   for f in range(5):
    fit=fold!=f;test=~fit;_,pa=fit_stack([x[:1000] for x in cc],y[:1000],fit,obj);pred[test]=apply_stack([x[:1000][test] for x in cc],pa)
   _,pa=fit_stack([x[:1000] for x in cc],y[:1000],np.ones(1000,bool),obj);dev=apply_stack([x[1000:] for x in cc],pa)
   results[name]={'calibration_oof':evaluate_metrics(pred,y[:1000])[0],'development':evaluate_metrics(dev,y[1000:])[0]}
  save(dest,{'results':results,'selection_unchanged':True});print('DEVELOPMENT_ABLATION',s,{m:round(z['development']['top1'],4) for m,z in results.items()},flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--subjects',nargs='+',default=SUBJECTS);p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);a=p.parse_args();torch.set_num_threads(1)
 root=ROOT/'results/final_hybrid_v76';root.mkdir(exist_ok=True);decision=ROOT/'results/final_choice_screen_v75/decision.json';sel=json.loads(decision.read_text());assert sel['selected']=='hybrid_nll'
 cfg={'source_sha256':sha(__file__),'selection_sha256':sha(decision),'selected':'hybrid_nll','candidate_set':'No additional candidates. Other variants are fixed mechanism controls.','frozen_at_utc':None,'components':7,'MLP_source':'Previously trained v64 GalleryMLP, 20 fixed epochs; no new backbone/head training','extra_parameters_per_encoder':8393728,'gallery_MLP_precomputed':True,'calibration':'100 concepts, nonnegative NLL stack; official200 transfers cross10 coefficients','fallback_guard':'Keep selected if full within10 mean Top1 is not lower than incumbent, Brier/NLL are not higher, mAP drop <=.001, ECE increase<=.005 and error-confidence increase<=.01. Otherwise restore incumbent without further search.','comparison_boundary':'Existing test galleries previously inspected; a fixed post-exploratory evaluation, not an independent confirmation.'}
 cp=root/'configuration.json'
 if cp.exists():assert json.loads(cp.read_text())==cfg
 else:save(cp,cfg)
 v=vision();develop_controls(v,root)
 for seed in a.seeds:
  for s in a.subjects:
   out=root/s/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
   if (out/'summary.json').exists():continue
   data=bank(s,seed,v);training,models,tr,splits=load_training(s,seed,v);records={};params={};eqparams={}
   for taskname,task in tasks(data).items():
    y=task['targets'];n=task['calibration_n'];q=task['queries'];rows=task['rows'];metric=raw_components(data,training,models,tr,task)['block_distance'];mlp=mlp_scores(data,task,s,seed);cc=sets(metric,mlp);records[taskname]={};params[taskname]={}
    for method,(kind,obj) in SPECS.items():
     if n:_,pa=fit_stack([x[:n] for x in cc[kind]],y[:n],np.ones(n,bool),obj)
     else:pa=params['cross10'][method]
     z=apply_stack([x[n:] for x in cc[kind]],pa);params[taskname][method]=pa;records[taskname][method]=evaluate_metrics(z,y[n:])[0]
     np.savez_compressed(out/f'{taskname}_{method}.npz',logits=z,targets=y[n:],query_ids=data['image_ids'][q[n:]],candidate_ids=data['image_ids'][rows[n:]],concepts=data['class_ids'][q[n:]])
    fz,fp=fusion_controls(cc['full'],y,n,eqparams['cross10'] if n==0 else None);eqparams[taskname]=fp
    for key in ['block_equal_temperature','block_equal_uncalibrated']:
     method=key.replace('block','hybrid');z=fz[key];records[taskname][method]=evaluate_metrics(z,y[n:])[0];np.savez_compressed(out/f'{taskname}_{method}.npz',logits=z,targets=y[n:])
    print('FIXED_HYBRID_TASK',s,seed,taskname,{m:round(records[taskname][m]['top1'],4) for m in ['hybrid_nll','hybrid_no_block','hybrid_no_repeat']},flush=True)
   save(out/'summary.json',{'configuration_sha256':sha(cp),'results':records,'parameters':params,'equal_fusion_parameters':eqparams,'backbone_sources':{x:sha(ROOT/f'results/closeout_experiments_v64/{s}/seed{seed}/nearg/{x}/mlp.pt') for x in ['NICE','ATM']}})
   del data,training,models;gc.collect();torch.cuda.empty_cache()
   save(root/'progress.json',{'completed':[{'subject':s0,'seed':se} for se in SEEDS for s0 in SUBJECTS if (root/s0/f'seed{se}/summary.json').exists()],'expected':30})
 print('FIXED_HYBRID_ALL_REQUESTED_COMPLETE',flush=True)

if __name__=='__main__':main()
