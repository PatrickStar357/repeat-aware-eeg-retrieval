"""Explicit single-positive retrieval and probabilistic reliability metrics.

Keep existing locked metrics.py unchanged. One image is relevant per query:
full-gallery AP=reciprocal rank, hence mAP=MRR. AURC instead ranks queries
by confidence and is not an image-ranking statistic.
"""
import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata
from reliability_repair_v15 import extra_metrics

def evaluate_metrics(logits,targets):
    x=np.asarray(logits,dtype=float);y=np.asarray(targets,dtype=int)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError('Invalid score matrix')
    if np.any(y<0) or np.any(y>=x.shape[1]):raise ValueError('Every closed-set query needs exactly one relevant candidate')
    order=np.argsort(-x,axis=1,kind='stable');rank=(order==y[:,None]).argmax(1)+1
    p=softmax(x,axis=1);confidence=p.max(1);correct=rank==1
    result=extra_metrics(x,y)
    result.update(top5=float((rank<=min(5,x.shape[1])).mean()),
        map=float(np.mean(1/rank)),mrr=float(np.mean(1/rank)),
        mean_rank=float(rank.mean()),median_rank=float(np.median(rank)),
        correct_confidence=float(confidence[correct].mean()) if correct.any() else None,
        wrong_confidence=float(confidence[~correct].mean()) if (~correct).any() else None)
    result['confidence_gap']=result['correct_confidence']-result['wrong_confidence'] if correct.any() and (~correct).any() else None
    b=np.minimum((confidence*15).astype(int),14)
    result['mce15']=float(max(abs(confidence[b==j].mean()-correct[b==j].mean()) for j in range(15) if (b==j).any()))
    if correct.any() and (~correct).any():
        ranks=rankdata(confidence,method='average');npos=correct.sum();nneg=(~correct).sum()
        result['correctness_auroc']=float((ranks[correct].sum()-npos*(npos+1)/2)/(npos*nneg))
    else:result['correctness_auroc']=None
    if x.shape[1]>10:result['top10']=float((rank<=10).mean())
    assert result['top1']<=result['top5']<=1 and result['map']==result['mrr']
    assert np.array_equal(order[:,0],x.argmax(1))
    return result,rank
