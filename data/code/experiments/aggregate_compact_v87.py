"""Freeze the previously evaluated compact scorer and all matched controls."""
from pathlib import Path
import json,copy
import numpy as np
from aggregate_closeout_v67 import paired,ROOT,RESULTS,SUBJECTS,SEEDS,TASKS,sha

def main():
 old=json.loads((RESULTS/'final_choice_v77/summary.json').read_text(encoding='utf8'));r=copy.deepcopy(old['records']);root=RESULTS/'compact_audit_v86';cfg=root/'configuration.json'
 for s in SUBJECTS:
  for se in SEEDS:
   p=root/s/f'seed{se}/summary.json';z=json.loads(p.read_text());assert z['configuration_sha256']==sha(cfg)
   for t in TASKS:
    r[s][str(se)][t].update({k:v for k,v in z['results'][t].items() if k!='compact'})
    r[s][str(se)][t]['hybrid_block_mlp']=z['results'][t]['compact']
 methods=sorted(r[SUBJECTS[0]]['2028']['within10']);a={};pp={};bs={};st={}
 for t in TASKS:
  a[t]={};pp[t]={};bs[t]={}
  for m in methods:
   a[t][m]={};pp[t][m]={};bs[t][m]={}
   for k in r[SUBJECTS[0]]['2028'][t][m]:
    x=[[r[s][str(se)][t][m][k] for se in SEEDS] for s in SUBJECTS]
    if any(v is None for row in x for v in row):continue
    x=np.array(x,float);p=x.mean(1);a[t][m][k]={'mean':float(p.mean()),'participant_sd':float(p.std(ddof=1))};pp[t][m][k]=p.tolist();bs[t][m][k]=x.mean(0).tolist()
  st[t]={}
  for m in ['block_distance','dual_NEARG_MLP','anchor_nll','compact_diagonal','compact_isotropic','compact_total','compact_residual','compact_permuted','compact_NICE','compact_ATM','hybrid_nll']:
   st[t][m]={k:paired(pp[t]['hybrid_block_mlp'][k],pp[t][m][k]) for k in ['top1','top5','map','brier','nll','ece15','wrong_confidence','aurc']}
   prev=0
   for i,k in enumerate(sorted(['top1','brier'],key=lambda k:st[t][m][k]['exact_signflip_p'])):
    prev=max(prev,min(1,(2-i)*st[t][m][k]['exact_signflip_p']));st[t][m][k]['holm_primary_p']=prev
 rank={}
 for t in TASKS:
  rank[t]={}
  for k in ['top1','top5','map','brier','nll','aurc','ece15','wrong_confidence']:
   rev=k in ['top1','top5','map'];ms=sorted(methods,key=lambda m:a[t][m][k]['mean'],reverse=rev);rank[t][k]={'rank':ms.index('hybrid_block_mlp')+1,'best3':ms[:3]}
 out=RESULTS/'compact_final_v87';out.mkdir(exist_ok=True)
 d={**old,'records':r,'aggregate':a,'participant':pp,'by_seed':bs,'methods':methods,'final_comparisons':st,'rank_all_configurations':rank,'final_method':'hybrid_block_mlp','final_decision_locked':True,'no_more_search':True,'source_sha256':sha(Path(__file__)),'audit_configuration_sha256':sha(cfg),'selection_source':'Three-score simplification already evaluated as hybrid_block_mlp; calibration concept-OOF and development support checked for subjects01-03 seed2028. Final test was already inspected; reported as exploratory, not independent confirmation. No extra backbone training.'}
 (out/'summary.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
 for t in TASKS:
  print(t,'ranks',rank[t])
  for m in ['NICE_global','ATM_global','global2','dual_NEARG_MLP','anchor_nll','block_distance','compact_diagonal','compact_total','compact_residual','hybrid_nll','hybrid_block_mlp']:
   print(m,{k:round(a[t][m][k]['mean'],6) for k in ['top1','top5','map','brier','nll','ece15','wrong_confidence','aurc']})
 print('LOCKED',len(methods),'fixed configurations')
if __name__=='__main__':main()
