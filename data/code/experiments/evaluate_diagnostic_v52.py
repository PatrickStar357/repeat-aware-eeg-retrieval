"""Locked held-out all-participant evaluation. No tuning from final outcomes.

Calibration labels are supplied only to fit_stack on the calibration concepts.
Test labels are supplied only to metrics after scores and weights are frozen.
Both tasks share EEG, trial counts and galleries across methods.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,gc,time,itertools
import numpy as np
import torch
from torch.nn import functional as F
from native_temporal_screen import ROOT,sha,save
from protocol import category_split,repeat_groups,candidate_rows
from load_visual_features import load_features
from train_all_backbones_v42 import Nice,forward
from atm_control_v9 import author_classes
from evidence_graph_v48 import EvidenceHead
from train_diagnostic_heads_v51 import MODES,transform
from variance_source_controls_v13 import shrunk_variance
from discriminant_evidence_v40 import covariance
from score_stacking_v24 import fit_stack
from metric_suite_v23 import evaluate_metrics

METHODS=['NICE_global','ATM_global','global2','isotropic4','repeat4','block_distance','joint_distance']+MODES

def apply_stack(components,parameters):
    h=np.stack([s-s.mean(1,keepdims=True) for s in components],axis=-1)/np.array(parameters['fit_scales'])
    return h@np.array(parameters['weights'])

def encode_reserved(subject,indices,labels,ids):
    folder=ROOT/'processed'/subject;raw=np.load(folder/'train_eeg.npy',mmap_mode='r');metadata=np.load(folder/'train_identities.npz')
    base=ROOT/'results/all_subject_backbones_v42'/subject/'seed2028';out={};ns,_=author_classes()
    pairs=[repeat_groups(s) for s in metadata['session_id'][indices]]
    for arch in ['NICE','ATM']:
        checkpoint=torch.load(base/arch/'model.pt',map_location='cpu',weights_only=True)
        model=Nice() if arch=='NICE' else ns['ATMS']();model.load_state_dict(checkpoint['model']);model.cuda().eval()
        mapping=dict(np.load(base/arch/'model.npz'));groups=[];global4=[];begin=time.time();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0,len(indices),32):
                rr=np.asarray(raw[indices[start:start+32]],dtype=np.float32);pp=pairs[start:start+32]
                a=np.stack([r[p[0]].mean(0) for r,p in zip(rr,pp)]);b=np.stack([r[p[1]].mean(0) for r,p in zip(rr,pp)])
                x=torch.tensor(np.stack([a,b,rr.mean(1)],axis=1).reshape(-1,63,250),device='cuda')
                if arch=='NICE':
                    token=model.encoder[0](x[:,None]).half().float();z=F.normalize(model.projector(token.flatten(1)),dim=-1)
                else:z=F.normalize(forward(model,x,arch),dim=-1)
                z=z.reshape(len(rr),3,1024).cpu().numpy();global4.append(z[:,2])
                latent=((z[:,:2].astype(float).reshape(-1,1024)-mapping['neural_center'])@mapping['neural_basis']/mapping['neural_scale']).reshape(-1,2,64)
                groups.append(latent)
        out[arch]={'x':np.concatenate(groups),'global':np.concatenate(global4),'mapping':mapping,
            'encoding_seconds':time.time()-begin,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2,
            'parameters':sum(p.numel() for p in model.parameters())}
        del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    return out

def load_heads(subject):
    base=ROOT/'results/diagnostic_heads_v51'/subject;manifest=json.loads((base/'manifest.json').read_text());geometry=dict(np.load(base/'geometry.npz'));heads={}
    assert sha(base/'geometry.npz')==manifest['geometry_sha256']
    for mode in MODES:
        file=base/mode/'unary.pt';assert sha(file)==manifest['head_sha256'][mode]
        checkpoint=torch.load(file,map_location='cpu',weights_only=True);model=EvidenceHead().cuda();model.load_state_dict(checkpoint['model']);model.eval();heads[mode]=model
    return heads,geometry,manifest

@torch.no_grad()
def head_scores(model,x,mu,rows,queries,c,scale,bias):
    x=x.copy();x[:,0]-=bias;x[:,1]+=bias;x,mu=transform(x,mu,c,scale)
    pooled=[];aa=[];bb=[];begin=time.time();torch.cuda.synchronize()
    for start in range(0,len(queries),64):
        q=queries[start:start+64];v=torch.tensor(mu[rows[start:start+64]],device='cuda');xx=torch.tensor(x[q],device='cuda')[:,None]
        views=torch.cat([xx,xx.mean(2,keepdim=True)],dim=2);z=model(views,v,'unary')[:,0]
        aa.append(z[:,0].cpu().numpy());bb.append(z[:,1].cpu().numpy());pooled.append(z[:,2].cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(pooled),np.concatenate(aa),np.concatenate(bb),time.time()-begin

def components(encoded,train,visual,rows,q,heads,geometry):
    global_scores=[];isotropic=[];repeat=[];x=[];mu=[];delta=[]
    for arch in ['NICE','ATM']:
        m=encoded[arch]['mapping'];xx=encoded[arch]['x'];pred=(visual@m['visual_to_neural']+m['offset']+m['residual_mean']).reshape(-1,2,64).mean(1)
        dd=(train[arch]['group_latents'][:,0]-train[arch]['group_latents'][:,1])/2
        local=xx[q].mean(1)[:,None]-pred[rows]
        global_scores.append(np.einsum('nd,nkd->nk',encoded[arch]['global'][q],visual[rows]))
        isotropic.append(-.5*np.sum(local**2,axis=-1));repeat.append(-.5*np.sum(local**2/shrunk_variance(dd),axis=-1))
        x.append(xx);mu.append(pred);delta.append(dd)
    x=np.concatenate(x,-1);mu=np.concatenate(mu,-1);dd=np.concatenate(delta,-1);difference=x[q].mean(1)[:,None]-mu[rows]
    base=global_scores+repeat;out={'NICE_global':[global_scores[0]],'ATM_global':[global_scores[1]],'global2':global_scores,
        'isotropic4':global_scores+isotropic,'repeat4':base};diagnostic={};timing={}
    for kind in ['block','joint']:
        score=-.5*np.einsum('nkd,de,nke->nk',difference,np.linalg.inv(covariance(dd,kind)),difference,optimize=True)
        out[kind+'_distance']=base+[score]
    for mode in MODES:
        z,a,b,t=head_scores(heads[mode],x,mu,rows,q,geometry[mode+'_covariance'],float(geometry[mode+'_scale']),geometry['group_bias'])
        out[mode]=base+[z];timing[mode]=t
        diagnostic[mode]={'A':a,'B':b,'pooled':z}
    return out,diagnostic,timing

def statistics(records,task='within10'):
    subsets={'all10':sorted(records),'additional04_10':[s for s in sorted(records) if int(s[-2:])>=4]};out={};rng=np.random.default_rng(20261104)
    for name,subjects in subsets.items():
        out[name]={};raw_p=[]
        for metric in ['top1','brier','map','nll','ece15','wrong_confidence','aurc']:
            d=np.array([records[s][task]['adapter_joint'][metric]-records[s][task]['repeat4'][metric] for s in subjects]);n=len(d)
            means=d[rng.integers(n,size=(20000,n))].mean(1)
            signs=np.array(list(itertools.product([-1,1],repeat=n)));null=(signs*d).mean(1)
            p=float((np.abs(null)>=abs(d.mean())-1e-12).mean())
            out[name][metric]={'n_participants':n,'mean_difference':float(d.mean()),'bootstrap95CI':np.quantile(means,[.025,.975]).tolist(),
                'exact_two_sided_signflip_p':p,'individual_differences':d.tolist(),'secondary':metric not in ['top1','brier']}
            if metric in ['top1','brier']:raw_p.append((metric,p))
        order=sorted(raw_p,key=lambda x:x[1]);running=0
        for i,(m,p) in enumerate(order):running=max(running,min(1.,(2-i)*p));out[name][m]['holm_primary_p']=running
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',default=[f'sub-{i:02}' for i in range(1,11)]);args=p.parse_args()
    root=ROOT/'results/diagnostic_final_v52';root.mkdir(exist_ok=True);lock=ROOT/'results/diagnostic_line_lock_v50.json';record=json.loads(lock.read_text())
    assert record['allow_final_evaluation'] and not record['allow_final_data_for_method_selection'];torch.set_num_threads(1)
    cfg={'source_sha256':sha(__file__),'lock_sha256':sha(lock),'head_configuration_sha256':sha(ROOT/'results/diagnostic_heads_v51/configuration.json'),
        'methods':METHODS,'tasks':['within10','cross10'],'calibration':'100 calibration concepts only; all scaling and fusion fitted here',
        'test':'200 fine-test concepts;2000 queries; same R4 across methods','no_final_method_selection':True,'original_guard_unchanged':True}
    cp=root/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:save(cp,cfg)
    v,canonical=load_features('train');v=F.normalize(torch.tensor(v,dtype=torch.float32),dim=-1).numpy().astype(float);records={}
    for subject in args.subjects:
        folder=root/subject
        if (folder/'summary.json').exists():records[subject]=json.loads((folder/'summary.json').read_text())['results'];continue
        folder.mkdir(exist_ok=True);identity=np.load(ROOT/'processed'/subject/'train_identities.npz');splits=category_split(identity['class_id'])
        assert all(set(splits[a]).isdisjoint(splits[b]) for a,b in itertools.combinations(splits,2))
        cal=np.flatnonzero(np.isin(identity['class_id'],splits['calibration']));test=np.flatnonzero(np.isin(identity['class_id'],splits['fine_test']));indices=np.r_[cal,test]
        ids=identity['image_id'][indices];labels=identity['class_id'][indices].astype(str);np.testing.assert_array_equal(ids,canonical[indices]);assert len(cal)==1000 and len(test)==2000
        base=ROOT/'results/all_subject_backbones_v42'/subject/'seed2028';train={}
        for arch in ['NICE','ATM']:
            bank=dict(np.load(base/arch/'compact.npz'));train[arch]={'group_latents':bank['group_latents'][np.isin(bank['class_ids'],splits['train'])]}
        print('FINAL_ENCODE_START',subject,flush=True);encoded=encode_reserved(subject,indices,labels,ids);heads,geometry,manifest=load_heads(subject);result={};params={};repro={};time_record={}
        for cross,task in [(False,'within10'),(True,'cross10')]:
            qcal=np.arange(1000);qtest=np.arange(1000,3000)
            rc,yc=candidate_rows(labels,qcal,qcal,cross=cross,seed=20261103);rt,yt=candidate_rows(labels,qtest,qtest,cross=cross,seed=20261103)
            rows=np.r_[rc,rt];q=np.r_[qcal,qtest];comp,diagnosis,timing=components(encoded,train,v[indices],rows,q,heads,geometry);result[task]={};params[task]={};repro[task]={}
            for method in METHODS:
                # Crucially: test targets never enter fit_stack, or its scales.
                _,pa=fit_stack([a[:1000] for a in comp[method]],yc,np.ones(1000,dtype=bool),'brier')
                z=apply_stack([a[1000:] for a in comp[method]],pa);result[task][method]=evaluate_metrics(z,yt)[0];params[task][method]=pa
                np.savez_compressed(folder/(task+'_'+method+'.npz'),logits=z,targets=yt,query_ids=ids[qtest],candidate_ids=ids[rt],concepts=labels[qtest])
            for mode,d in diagnosis.items():
                a=d['A'][1000:];b=d['B'][1000:];ya=a.argmax(1);yb=b.argmax(1)
                margin_a=a[np.arange(2000),yt]-np.max(np.where(np.eye(10)[yt].astype(bool),-np.inf,a),axis=1)
                margin_b=b[np.arange(2000),yt]-np.max(np.where(np.eye(10)[yt].astype(bool),-np.inf,b),axis=1)
                repro[task][mode]={'split_half_prediction_agreement':float((ya==yb).mean()),'both_correct':float(((ya==yt)&(yb==yt)).mean()),
                    'both_agree_wrong':float(((ya==yb)&(ya!=yt)).mean()),'target_margin_correlation':float(np.corrcoef(margin_a,margin_b)[0,1]),
                    'boundary':'Predictions and margins of session-disjoint R2 groups; not attention/topographic replication or statistically independent witnesses.'}
                np.savez_compressed(folder/(task+'_'+mode+'_split_halves.npz'),A=a,B=b,pooled=d['pooled'][1000:],targets=yt)
            time_record[task]=timing
            print('FINAL_TASK',subject,task,{m:{k:round(result[task][m][k],6) for k in ['top1','map','brier','nll','ece15']} for m in ['repeat4','adapter_isotropic','adapter_diagonal','adapter_joint']},flush=True)
        save(folder/'summary.json',{'configuration_sha256':sha(cp),'head_manifest_sha256':sha(ROOT/'results/diagnostic_heads_v51'/subject/'manifest.json'),
            'results':result,'calibration_parameters':params,'split_half_diagnostics':repro,'head_timing_seconds':time_record,
            'backbone_cost':{a:{k:encoded[a][k] for k in ['encoding_seconds','peak_GPU_MiB','parameters']} for a in encoded},
            'final_method_unchanged':True,'test_used_only_for_metrics':True})
        records[subject]=result;save(root/'progress.json',{'completed':sorted(records),'method_frozen':True})
        del encoded,train,heads;gc.collect();torch.cuda.empty_cache()
    all_subjects=sorted(records);aggregate={task:{m:{k:float(np.mean([records[s][task][m][k] for s in all_subjects])) if records[all_subjects[0]][task][m][k] is not None else None for k in records[all_subjects[0]][task][m]} for m in METHODS} for task in ['within10','cross10']}
    save(root/'summary.json',{'records':records,'aggregate':aggregate,'paired_statistics':statistics(records),'configuration_sha256':sha(cp),'complete_n':len(records),
        'method_selection_before_reserved_access':True,'source_sha256':sha(__file__)})
    assert sha(__file__)==cfg['source_sha256'];print('FINAL_EVALUATION_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
