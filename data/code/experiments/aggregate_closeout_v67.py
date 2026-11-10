"""Participant-level aggregation of the locked 30-run closeout matrix."""
from pathlib import Path
import json,itertools,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'data/results'
SUBJECTS=[f'sub-{i:02}' for i in range(1,11)]
SEEDS=[2028,2029,2030]
TASKS=['within10','cross10','official200']

def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def paired(a,b):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(20261104)
    ci=np.quantile(d[rng.integers(len(d),size=(20000,len(d)))].mean(1),[.025,.975])
    null=(np.array(list(itertools.product([-1,1],repeat=len(d))))*d).mean(1)
    return {'difference':float(d.mean()),'bootstrap95CI':ci.tolist(),'exact_signflip_p':float((abs(null)>=abs(d.mean())-1e-12).mean()),'individual':d.tolist(),'n_participants':len(d)}

def main():
    root=RESULTS/'closeout_experiments_v64';config=root/'configuration.json';cfg=read(config)
    records={};hashes={}
    for s in SUBJECTS:
        records[s]={}
        for seed in SEEDS:
            p=root/s/f'seed{seed}/summary.json';d=read(p)
            assert d['configuration_sha256']==sha(config) and d['method_unchanged']
            assert set(d['results'])==set(TASKS)
            records[s][str(seed)]=d['results'];hashes[str(p.relative_to(ROOT))]=sha(p)
    methods=sorted(records[SUBJECTS[0]]['2028']['within10'])
    assert len(methods)==27
    participant={};aggregate={};by_seed={};stats={}
    for task in TASKS:
        participant[task]={};aggregate[task]={};by_seed[task]={}
        for m in methods:
            metrics=list(records[SUBJECTS[0]]['2028'][task][m]);participant[task][m]={};aggregate[task][m]={};by_seed[task][m]={}
            for metric in metrics:
                vals=[[records[s][str(seed)][task][m][metric] for seed in SEEDS] for s in SUBJECTS]
                if any(v is None for row in vals for v in row): continue
                v=np.array(vals,float);assert np.isfinite(v).all()
                p=v.mean(1);participant[task][m][metric]=p.tolist()
                aggregate[task][m][metric]={'mean':float(p.mean()),'participant_sd':float(p.std(ddof=1))}
                by_seed[task][m][metric]=v.mean(0).tolist()
            assert abs(aggregate[task][m]['map']['mean']-aggregate[task][m]['mrr']['mean'])<1e-12
        stats[task]={k:paired(participant[task]['block_distance'][k],participant[task]['repeat4'][k]) for k in ['top1','top5','map','brier','nll','ece15','wrong_confidence','aurc']}
        previous=0
        for i,k in enumerate(sorted(['top1','brier'],key=lambda k:stats[task][k]['exact_signflip_p'])):
            previous=max(previous,min(1,(2-i)*stats[task][k]['exact_signflip_p']));stats[task][k]['holm_primary_p']=previous
    cost=read(RESULTS/'cost_benchmark_v66/summary.json');assert len(cost['records'])==24
    difficulty={str(k):{m:{metric:float(np.mean([read(root/s/'seed2028/summary.json')['difficulty'][str(k)][m][metric] for s in SUBJECTS])) for metric in ['top1','map','brier','nll']} for m in ['repeat4','block_distance']} for k in [2,5]}
    out={'complete_subjects':10,'seeds':SEEDS,'complete_runs':30,'methods':methods,'tasks':TASKS,'unit':'participant; average seeds within participant before uncertainty/statistical tests','records':records,'participant':participant,'aggregate':aggregate,'by_seed':by_seed,'paired_block_minus_repeat':stats,'difficulty_seed2028':difficulty,'cost':cost,'configuration':cfg,'hashes':hashes,'configuration_sha256':sha(config),'source_sha256':sha(Path(__file__))}
    dest=RESULTS/'submission_closeout_v67';dest.mkdir(exist_ok=True);(dest/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    for t in TASKS:
        print(t)
        for m in ['global2','repeat4','block_distance','NICE_block','ATM_block','dual_NEARG_ridge','dual_NEARG_MLP','dual_SATTC','block_nll','block_equal_temperature','block_equal_probability']:
            print(m,{k:round(aggregate[t][m][k]['mean'],6) for k in ['top1','top5','map','brier','nll','ece15','wrong_confidence','aurc']})
        print('paired',stats[t]['top1'],stats[t]['brier'])
    print('SEEDS',[{k:by_seed['within10'][m][k] for k in ['top1','brier']} for m in ['repeat4','block_distance']])

if __name__=='__main__':main()
