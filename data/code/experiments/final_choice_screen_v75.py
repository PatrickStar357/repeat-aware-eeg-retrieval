"""One bounded author-authorized final choice on calibration/development data.

Four fixed candidates, three participants, seed2028. No final-test outcomes
are read to select the candidate. Earlier test inspection is acknowledged.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,gc,time,datetime
import numpy as np
import torch
from torch.nn import functional as F
from native_temporal_screen import ROOT,sha,save
from evaluation_banks_v62 import vision,load_training
from closeout_experiments_v64 import GalleryMLP,raw_components
from protocol import candidate_rows
from score_stacking_v24 import fit_stack
from evaluate_diagnostic_v52 import apply_stack
from metric_suite_v23 import evaluate_metrics

METHODS=['incumbent','metric_nll','hybrid_brier','hybrid_nll','MLP_reference']
SUBJECTS=['sub-01','sub-02','sub-03']
SPECS={'incumbent':('metric','brier'),'metric_nll':('metric','nll'),'hybrid_brier':('hybrid','brier'),'hybrid_nll':('hybrid','nll'),'MLP_reference':('mlp','brier')}

def screening_data(s,v):
 training,models,tr,splits=load_training(s,2028,v)
 cache=np.load(ROOT/f'results/evaluation_banks_v62/{s}/seed2028.npz')
 original=training['NICE']['original_indices'];dev=np.flatnonzero(np.isin(training['NICE']['class_ids'],splits['validation']))
 assert len(dev)==1000
 data={}
 for arch in ['NICE','ATM']:
  data[arch+'_latent']=np.r_[cache[arch+'_latent'][:1000],training[arch]['group_latents'][dev]]
  data[arch+'_global']=np.r_[cache[arch+'_global'][:1000],training[arch]['global_embeddings'][dev]]
 data.update(image_ids=np.r_[cache['image_ids'][:1000],training['NICE']['image_ids'][dev]],class_ids=np.r_[cache['class_ids'][:1000],training['NICE']['class_ids'][dev]],visual=np.r_[cache['visual'][:1000],v['train'][0][original[dev]].astype('float32')])
 assert set(data['class_ids'][:1000])==set(splits['calibration']) and set(data['class_ids'][1000:])==set(splits['validation'])
 assert set(data['class_ids']).isdisjoint(splits['train']) and set(data['class_ids']).isdisjoint(splits['fine_test'])
 return data,training,models,tr

@torch.no_grad()
def mlp_scores(data,task,s,seed=2028):
 out=[];q=task['queries'];rows=task['rows'];v=data['visual'].astype(float)
 for arch in ['NICE','ATM']:
  path=ROOT/f'results/closeout_experiments_v64/{s}/seed{seed}/nearg/{arch}/mlp.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
  assert ck['source_sha256']==sha(ROOT/'code/experiments/closeout_experiments_v64.py') and ck['backbone_seed']==seed
  m=GalleryMLP().cuda();m.load_state_dict(ck['model']);m.eval()
  hyp=np.concatenate([F.normalize(m(torch.tensor(v[i:i+256],dtype=torch.float32,device='cuda')),dim=-1).cpu().numpy() for i in range(0,len(v),256)])
  x=data[arch+'_global'];glob=np.einsum('nd,nkd->nk',x[q],v[rows]);anchor=np.einsum('nd,nkd->nk',x[q],hyp[rows]);out.append(.25*glob+.75*anchor)
  del m,ck;torch.cuda.empty_cache()
 return out

def build_components(s,v):
 data,training,models,tr=screening_data(s,v);labels=data['class_ids'];q=np.arange(2000)
 rc,yc=candidate_rows(labels,np.arange(1000),np.arange(1000),seed=20261103);rd,yd=candidate_rows(labels,np.arange(1000,2000),np.arange(1000,2000),seed=20261103)
 task={'queries':q,'rows':np.r_[rc,rd]};metric=raw_components(data,training,models,tr,task)['block_distance'];mlp=mlp_scores(data,task,s)
 return {'metric':metric,'mlp':mlp,'hybrid':metric+mlp},np.r_[yc,yd],labels

def run_case(s,v,out):
 components,y,labels=build_components(s,v);unique=sorted(set(labels[:1000]));order=np.random.default_rng(20261105).permutation(unique);lookup={c:i%5 for i,c in enumerate(order)};fold=np.array([lookup[c] for c in labels[:1000]])
 result={};parameters={}
 for method,(kind,objective) in SPECS.items():
  comp=components[kind];z=np.empty((1000,10));foldparams=[]
  for f in range(5):
   fit=fold!=f;test=~fit;assert set(labels[:1000][fit]).isdisjoint(labels[:1000][test])
   _,pa=fit_stack([a[:1000] for a in comp],y[:1000],fit,objective);z[test]=apply_stack([a[:1000][test] for a in comp],pa);foldparams.append(pa)
  _,pa=fit_stack([a[:1000] for a in comp],y[:1000],np.ones(1000,bool),objective);dev=apply_stack([a[1000:] for a in comp],pa)
  result[method]={'calibration_oof':evaluate_metrics(z,y[:1000])[0],'development':evaluate_metrics(dev,y[1000:])[0]};parameters[method]={'folds':foldparams,'all_calibration':pa}
  np.savez_compressed(out/(method+'.npz'),calibration_oof=z,development=dev,targets_cal=y[:1000],targets_dev=y[1000:],fold=fold)
 save(out/'summary.json',{'source_sha256':sha(__file__),'results':result,'parameters':parameters,'no_final_test_used':True,'roles':'calibration100 concept-OOF5 and development100, sub01-03 seed2028','raw_R4_unchanged':True})
 print('FINAL_SCREEN',s,{m:{split:{k:round(z[k],5) for k in ['top1','brier','nll','ece15','wrong_confidence']} for split,z in value.items()} for m,value in result.items()},flush=True)
 return result

def main():
 torch.set_num_threads(1);root=ROOT/'results/final_choice_screen_v75';root.mkdir(exist_ok=True)
 config={'source_sha256':sha(__file__),'authorized_by':'User explicitly requests one final small-scale improvement check and fallback if it fails; 2026-11-05','subjects':SUBJECTS,'seed':2028,'candidates':SPECS,'selection_uses':['calibration100 five concept folds','development100'],'already_seen_test_acknowledged':True,'test_for_selection':False,'fixed_gate':{'cal_oof_top1_gain_min':0.0,'dev_top1_gain_min':.001,'dev_brier_increase_max':.0002,'dev_nll_increase_max':.005,'dev_map_drop_max':.001,'dev_ece_increase_max':.005,'dev_wrong_confidence_increase_max':.01,'dev_aurc_increase_max':.003,'participants_with_dev_accuracy_gain_min':2,'hybrid_dev_top1_gain_min':.003},'choice_rule':'Among eligible candidates choose largest mean of calibration-OOF and development Top1; ties lower development Brier. Otherwise incumbent. Exactly one bounded screen, no tuned temperatures/architectures.'}
 lock=root/'configuration.json'
 if lock.exists():assert json.loads(lock.read_text())==config
 else:save(lock,config)
 if not (root/'prior_mainline.json').exists():save(root/'prior_mainline.json',json.loads((ROOT/'results/current_mainline.json').read_text(encoding='utf8')))
 v=vision();records={}
 for s in SUBJECTS:
  folder=root/s;folder.mkdir(exist_ok=True)
  if (folder/'summary.json').exists():
   d=json.loads((folder/'summary.json').read_text());assert d['source_sha256']==sha(__file__);records[s]=d['results']
  else:records[s]=run_case(s,v,folder)
  gc.collect();torch.cuda.empty_cache()
 aggregate={m:{split:{k:float(np.mean([records[s][m][split][k] for s in SUBJECTS])) for k in ['top1','top5','map','brier','nll','ece15','wrong_confidence','aurc']} for split in ['calibration_oof','development']} for m in METHODS};eligible=[];gates={}
 for m in METHODS[1:-1]:
  delta={split:{k:aggregate[m][split][k]-aggregate['incumbent'][split][k] for k in aggregate[m][split]} for split in aggregate[m]};z=delta['development'];wins=sum(records[s][m]['development']['top1']>records[s]['incumbent']['development']['top1'] for s in SUBJECTS)
  checks={'cal_accuracy':delta['calibration_oof']['top1']>=-1e-12,'dev_accuracy':z['top1']>=.001-1e-12,'brier':z['brier']<=.0002,'nll':z['nll']<=.005,'map':z['map']>=-.001,'ece':z['ece15']<=.005,'wrong_confidence':z['wrong_confidence']<=.01,'aurc':z['aurc']<=.003,'participant_accuracy':wins>=2,'extra_MLP_cost_justified':not m.startswith('hybrid') or z['top1']>=.003-1e-12}
  gates[m]={'checks':checks,'deltas':delta,'eligible':all(checks.values())}
  if all(checks.values()):eligible.append(m)
 selected=max(eligible,key=lambda m:((aggregate[m]['calibration_oof']['top1']+aggregate[m]['development']['top1'])/2,-aggregate[m]['development']['brier'])) if eligible else 'incumbent'
 save(root/'decision.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'configuration_sha256':sha(lock),'source_sha256':sha(__file__),'records':records,'aggregate':aggregate,'gates':gates,'selected':selected,'selection_finalized_before_full_evaluation':True,'no_further_candidate_search':True,'reporting_boundary':'Exploratory improvement after earlier test inspection; new selection uses calibration and development only, not a new independent test.'})
 print('FINAL_SMALL_SCREEN_DECISION',selected,json.dumps({'aggregate':aggregate,'gates':gates},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
