"""Matched three-participant reliability audit, development concepts only.

Two-fold concept-separated temperature fitting is a development diagnostic:
the EEG checkpoints were already selected using these validation concepts.
It must not be called an unbiased final calibration evaluation.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from metrics import classification_metrics,agreement
from protocol import category_split
from validate_foundation import ROOT

SUBJECTS=['sub-01','sub-02','sub-03']
MODES=['global','input_average','mlp','two_view','cross']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def temperature(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y);r=np.arange(len(y))
    def loss(logt):
        z=x/np.exp(logt)
        return float(np.mean(logsumexp(z,axis=1)-z[r,y]))
    result=minimize_scalar(loss,bounds=(np.log(1e-4),np.log(100)),method='bounded',options={'xatol':1e-8})
    assert result.success
    return float(np.exp(result.x))

def evaluate(x,y):
    result=classification_metrics(x,y)
    p=np.exp(x-logsumexp(x,axis=1,keepdims=True));pred=x.argmax(1);conf=p.max(1)
    order=np.argsort(-conf,kind='stable');correct=pred==y
    for coverage in [.25,.5,.75]:
        result['risk_at_'+str(int(coverage*100))]=float((~correct[order[:int(len(y)*coverage)]]).mean())
    result['mean_confidence']=float(conf.mean())
    return result

def load(subject,mode,task):
    base=ROOT/'results/raw_joint_core_v2'/subject/'seed2026'
    if mode=='input_average':
        path=ROOT/'results/input_average_screen_v1'/subject/'seed2026/validation_predictions.npz'
        p=dict(np.load(path));return p[task+'_scores'],p[task+'_target'],p['query_indices'],p[task+'_candidate_rows'],path,None
    path=base/('cross' if mode=='global' else mode)/(task+'_validation.npz')
    p=dict(np.load(path))
    return p['global_logits'] if mode=='global' else p['logits'],p['targets'],p['query_indices'],p['candidate_rows'],path,p

def main():
    out=ROOT/'results/reliability_screen_v1';out.mkdir(exist_ok=True)
    records={};sources={}
    for subject in SUBJECTS:
        labels=np.load(ROOT/'processed'/subject/'train_identities.npz')['class_id'].astype(str)
        split=category_split(labels);fold_classes=[set(split['validation'][:50]),set(split['validation'][50:])]
        records[subject]={}
        for task in ['within10','cross10']:
            records[subject][task]={};reference=None
            for mode in MODES:
                x,y,queries,rows,path,p=load(subject,mode,task)
                if reference is None:reference=(y,queries,rows)
                else:assert all(np.array_equal(a,b) for a,b in zip(reference,(y,queries,rows)))
                fold=np.array([0 if labels[q] in fold_classes[0] else 1 for q in queries]);scaled=np.empty_like(x,dtype=float);temps=[]
                for testfold in [0,1]:
                    fit=fold!=testfold;test=~fit;assert set(labels[queries[fit]]).isdisjoint(labels[queries[test]])
                    t=temperature(x[fit],y[fit]);temps.append(t);scaled[test]=x[test]/t
                raw=evaluate(x,y);cal=evaluate(scaled,y)
                assert raw['top1']==cal['top1']
                value={'raw':raw,'cross_fitted_temperature':cal,'temperature_for_evaluation_fold':temps}
                if p is not None and mode in ['cross','two_view']:
                    a,b=p['score_a'],p['score_b'];value['support']=agreement(a,b,y)
                    ac=a.argmax(1)==y;bc=b.argmax(1)==y
                    value['support']['B_correct_given_A_correct']=float(bc[ac].mean()) if ac.any() else None
                    value['support']['A_correct']=float(ac.mean());value['support']['B_correct']=float(bc.mean())
                records[subject][task][mode]=value;sources[str(path.relative_to(ROOT))]=sha(path)
                np.savez_compressed(out/(subject+'_'+task+'_'+mode+'.npz'),logits=x,scaled_logits=scaled,
                    targets=y,query_indices=queries,candidate_rows=rows,fold=fold)
    aggregate={}
    for task in ['within10','cross10']:
        aggregate[task]={}
        for mode in MODES:
            aggregate[task][mode]={}
            for phase in ['raw','cross_fitted_temperature']:
                keys=records[SUBJECTS[0]][task][mode][phase]
                aggregate[task][mode][phase]={k:float(np.mean([records[s][task][mode][phase][k] for s in SUBJECTS])) for k in keys}
    report={'scope':'checkpoint-selected validation only; 3 participants x 1 seed; cross-fitted temperature is diagnostic, not final calibration',
        'source_sha256':sha(__file__),'sources':sources,'calibration_or_final_fine_test_accessed':False,
        'temperature_bounds':[1e-4,100],'records':records,'aggregate':aggregate,
        'attention_overlap_as_evidence_reproducibility':False,
        'notes':['Brier is a proper probability score combining calibration and predictive resolution, not a pure calibration error.',
                 'Wrong-answer confidence alone can be made low by uniform predictions; interpret with NLL/Brier and selective risk.',
                 'A larger mismatch drop shows sensitivity to B, not unique benefit or greater reliability.',
                 'Temperature-fitting folds are disjoint by concept; model selection already used all validation concepts.']}
    (out/'reliability.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    rows=[]
    for mode in MODES:
        raw=aggregate['within10'][mode]['raw'];v=aggregate['within10'][mode]['cross_fitted_temperature']
        rows.append(f"| {mode} | {100*v['top1']:.1f}% | {raw['brier']:.4f} | {v['brier']:.4f} | {v['nll']:.4f} | {100*v['ece15']:.2f}% | {100*v['wrong_confidence']:.2f}% | {v['aurc']:.4f} | {100*v['risk_at_50']:.1f}% |")
    folder=ROOT.parent/'outputs/实验设计与论文写作'
    text='''# 准确性与可靠性：统一开发诊断

范围：sub-01至03、seed2026、已按验证集选取的模型。以下只用于开发决策；两折按概念分离拟合温度，但模型选取已使用全部验证概念，不能称为独立最终校准结果。正式100校准类和200细粒度测试类未读取。

## 指标定义的修正

- 复制A注意力到B后再量位置重合，必然重合，不能作为线索复现证据。改报B上的目标支持、正确一致CA、错误一致WA，以及A正确时B仍正确的比例；MLP无相应注意力指标，记为不适用。
- Brier同时受校准与辨别能力影响，不是纯校准误差；同时报告NLL、ECE和风险—覆盖指标。
- 错误置信度低并不自动更可靠，输出近均匀分布也能做到。所有方法都应有相同温度校准机会。
- 错配B下降大说明依赖B，不能单独证明复核有效；需与双视图/普通融合的下降比较。
- softmax(z/T)中增大T更平滑；融合权重属于模型参数，应在训练集学习。

## 三被试同类10选1均值

raw为原始分数，CF表示两折概念分离温度拟合后的开发诊断；所有低优指标均越低越好。

| 方法 | Top-1 | raw Brier | CF Brier | CF NLL | CF ECE | CF错误置信度 | CF AURC | 保留50%时错误率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n'
    text+='\n校准前后Top-1逐方法一致。完整逐被试、同类/跨类及支持指标保存在data/results/reliability_screen_v1/reliability.json。\n'
    (folder/'准确性与可靠性诊断及模型改版.md').write_text(text,encoding='utf8')
    print(json.dumps(aggregate['within10'],indent=2))

if __name__=='__main__':main()
