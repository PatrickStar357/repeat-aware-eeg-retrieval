"""Recompute the main matched comparison from frozen logits, without training.

Run in the artifact folder: python reproduce_frozen.py
Requires NumPy. It does not load EEG recordings, stimulus photographs or models.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np

ROOT=Path(__file__).resolve().parent
SUMMARY=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
MODELS={'ours':'hybrid_block_mlp','anchor_nll':'anchor_nll'}

def metrics(z,y):
    z=np.asarray(z,dtype=float);y=np.asarray(y,dtype=int)
    assert np.isfinite(z).all() and z.ndim==2 and len(z)==len(y)
    order=np.argsort(-z,axis=1,kind='stable')
    rank=(order==y[:,None]).argmax(1)+1
    x=z-z.max(1,keepdims=True);p=np.exp(x);p/=p.sum(1,keepdims=True)
    c=p.max(1);correct=rank==1;qo=np.argsort(-c,kind='stable')
    oh=np.eye(z.shape[1])[y]
    b=np.minimum((c*15).astype(int),14)
    result={'top1':float(correct.mean()),'top5':float((rank<=min(5,z.shape[1])).mean()),
            'map':float((1/rank).mean()),'brier':float(np.square(p-oh).sum(1).mean()),
            'nll':float(-np.log(p[np.arange(len(y)),y].clip(1e-15)).mean()),
            'aurc':float((np.cumsum(~correct[qo])/np.arange(1,len(y)+1)).mean()),
            'ece15':float(sum(np.mean(b==i)*abs(c[b==i].mean()-correct[b==i].mean())for i in range(15)if (b==i).any())),
            'wrong_confidence':float(c[~correct].mean())}
    for coverage in [25,50,75]:result[f'risk_at_{coverage}']=float((~correct[qo[:int(len(y)*coverage/100)]]).mean())
    return result,rank

def main():
    all_rows=[];errors=[];count=0;max_error=0.
    for task in ['within10','cross10','official200']:
        for name,key in MODELS.items():
            participants=[]
            for subject in range(1,11):
                runs=[]
                for seed in [2028,2029,2030]:
                    path=ROOT/'predictions'/f'sub-{subject:02}'/f'seed{seed}'/f'{task}_{name}.npz'
                    with np.load(path,allow_pickle=False)as z:
                        m,_=metrics(z['logits'],z['targets']);runs.append(m)
                    count+=1
                participants.append({k:float(np.mean([r[k]for r in runs]))for k in runs[0]})
            for metric in participants[0]:
                observed=float(np.mean([p[metric]for p in participants]))
                expected=float(SUMMARY['aggregate'][task][key][metric]['mean'])
                error=abs(observed-expected);max_error=max(max_error,error)
                if error>1e-9:errors.append({'task':task,'model':name,'metric':metric,'expected':expected,'actual':observed})
                all_rows.append({'task':task,'model':name,'metric':metric,'recomputed':observed,'recorded':expected,'absolute_error':error})
    cases=json.loads((ROOT/'results/qualitative_cases_v151.json').read_text(encoding='utf-8'))['cases']
    for name,key in MODELS.items():
        with np.load(ROOT/'predictions/sub-01/seed2028'/f'within10_{name}.npz',allow_pickle=False)as z:
            _,ranks=metrics(z['logits'],z['targets'])
            for c in cases:assert int(ranks[c['query_index']])==int(c['models'][key]['target_rank'])
    report={'frozen_prediction_files':count,'aggregate_metrics_checked':len(all_rows),'max_absolute_error':max_error,
            'six_case_ranks_for_main_pair':'passed','units':'Three seeds averaged within each of ten participants',
            'summary_sha256':hashlib.sha256((ROOT/'results/summary.json').read_bytes()).hexdigest(),
            'errors':errors,'rows':all_rows,'status':'passed'if not errors else'failed',
            'scope':'Reproduction from frozen predictions, not independent retraining or validation of the development protocol.'}
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT.parent/'outputs'/'frozen_reproduction_check.json');args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items()if k not in ['rows','errors']},indent=2))
    assert not errors,errors[:3]

if __name__=='__main__':main()
