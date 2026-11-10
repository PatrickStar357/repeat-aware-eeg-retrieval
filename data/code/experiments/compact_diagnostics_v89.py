"""Cheap fixed-prediction identity and confidence checks, all 30 final runs."""
from pathlib import Path
import json
import numpy as np
from scipy.special import softmax
from aggregate_closeout_v67 import ROOT,RESULTS,SUBJECTS,SEEDS

def main():
 root=RESULTS/'compact_final_v87';d=json.loads((root/'summary.json').read_text());records={};rng=np.random.default_rng(20261105)
 for s in SUBJECTS:
  records[s]={}
  for seed in SEEDS:
   folder=RESULTS/f'compact_audit_v86/{s}/seed{seed}';f=np.load(folder/'within10_compact.npz');y=f['targets'];p=softmax(f['logits'],axis=1);pred=p.argmax(1);ok=pred==y;groups=[np.flatnonzero(f['concepts']==c) for c in np.unique(f['concepts'])];permuted=[]
   for _ in range(100):
    permuted_ids=f['query_ids'].copy()
    for ix in groups:permuted_ids[ix]=rng.permutation(permuted_ids[ix])
    target=(f['candidate_ids']==permuted_ids[:,None]).argmax(1);permuted.append(float((pred==target).mean()))
   wrong_logits=f['logits'][np.arange(len(y))[:,None],np.array([[j for j in range(10) if j!=truth] for truth in y])];absent=softmax(wrong_logits,axis=1).max(1);common={}
   for m in ['block_distance','dual_NEARG_MLP','anchor_nll']:
    source='compact_audit_v86' if m=='anchor_nll' else 'closeout_experiments_v64'
    b=np.load(RESULTS/f'{source}/{s}/seed{seed}/within10_{m}.npz');np.testing.assert_array_equal(b['targets'],y);bp=softmax(b['logits'],axis=1);bo=bp.argmax(1)==y
    common[m]={}
    for key,mask in [('both_wrong',~bo&~ok),('corrected',~bo&ok),('regressed',bo&~ok),('both_correct',bo&ok)]:
     common[m][key]={'n':int(mask.sum()),'reference_confidence':float(bp.max(1)[mask].mean()) if mask.any() else None,'final_confidence':float(p.max(1)[mask].mean()) if mask.any() else None}
   records[s][str(seed)]={'permuted_top1_mean':float(np.mean(permuted)),'absent_maxp_mean':float(absent.mean()),'absent_above_07':float((absent>.7).mean()),'common_errors':common}
 agg={k:float(np.mean([records[s][str(seed)][k] for s in SUBJECTS for seed in SEEDS])) for k in ['permuted_top1_mean','absent_maxp_mean','absent_above_07']}
 out={'records':records,'aggregate':agg,'scope':'Frozen final predictions; within-concept image-identity permutation100 and K9 absence. No open-set or anatomical recurrence claim.'};(root/'diagnostics.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('FINAL_DIAGNOSTICS',agg)
if __name__=='__main__':main()
