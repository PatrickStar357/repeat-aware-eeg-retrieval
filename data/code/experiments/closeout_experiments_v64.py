"""Finite closeout experiment matrix for the already fixed light mainline.

Three fixed backbone seeds, all ten participants. Adds complete single-encoder
and fusion controls, NEAR Gallery Anchoring adaptations, and pinned official
SATTC postprocessing with explicit unlabeled-batch access. No test-driven
method/parameter replacement. Original runs/checkpoints are preserved.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,gc,itertools,contextlib,io
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax
from native_temporal_screen import ROOT,sha,save,seed_all
from evaluation_banks_v62 import model_root,vision,bank,load_training,tasks
from protocol import candidate_rows
from cross_trial_likelihood_v6 import ridge
from discriminant_evidence_v40 import covariance
from variance_source_controls_v13 import shrunk_variance
from score_stacking_v24 import fit_stack
from evaluate_diagnostic_v52 import apply_stack
from reliability_screen import temperature
from metric_suite_v23 import evaluate_metrics
from sattc_adapter_v63 import author_functions,score as sattc_score,COMMIT,SOURCE

SUBJECTS=[f'sub-{i:02}' for i in range(1,11)]
SEEDS=[2028,2029,2030]
CORE=['NICE_global','ATM_global','global2','isotropic4','repeat4','block_distance','NICE_repeat','NICE_block','ATM_repeat','ATM_block','block_only','no_global','block_nll','block_equal_temperature','block_equal_probability','block_equal_uncalibrated','shrink_005','shrink_020']
EXTERNAL=['NICE_NEARG_ridge','ATM_NEARG_ridge','dual_NEARG_ridge','NICE_NEARG_MLP','ATM_NEARG_MLP','dual_NEARG_MLP','NICE_SATTC','ATM_SATTC','dual_SATTC']

class GalleryMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,2048),nn.GELU(),nn.Linear(2048,1024))
    def forward(self,x):return self.net(x)

def near_gallery(training,models,tr,v,original,folder,seed):
    """Paper Eq6 Gallery Anchoring only, beta=.75; no Query Anchoring claim."""
    result={};targets=v['train'][0][original];fit_visual=torch.tensor(targets[tr],dtype=torch.float64,device='cuda')
    for arch in ['NICE','ATM']:
        out=folder/arch;out.mkdir(parents=True,exist_ok=True);y=np.asarray(training[arch]['global_embeddings'][tr],dtype=float)
        ridgefile=out/'ridge.npz';mlpfile=out/'mlp.pt'
        if not ridgefile.exists():
            w,b,lam=ridge(fit_visual,torch.tensor(y,dtype=torch.float64,device='cuda'));np.savez_compressed(ridgefile,weight=w.cpu().numpy(),offset=b.cpu().numpy(),regularizer=lam,fit_ids=training[arch]['image_ids'][tr])
        rr=np.load(ridgefile);result[arch+'_ridge']=(rr['weight'],rr['offset'])
        if mlpfile.exists():
            ck=torch.load(mlpfile,map_location='cpu',weights_only=True);assert ck['source_sha256']==sha(__file__) and ck['backbone_seed']==seed
            m=GalleryMLP().cuda();m.load_state_dict(ck['model'])
        else:
            seed_all(20261105+seed);m=GalleryMLP().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);vx=fit_visual.float();yy=F.normalize(torch.tensor(y,dtype=torch.float32,device='cuda'),dim=-1);rng=np.random.default_rng(20261105+seed);history=[]
            for epoch in range(20):
                order=rng.permutation(len(tr));losses=[];m.train()
                for start in range(0,len(order),256):
                    ix=torch.tensor(order[start:start+256],device='cuda');z=F.normalize(m(vx[ix]),dim=-1);loss=(1-(z*yy[ix]).sum(-1)).mean()
                    assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
            torch.save({'model':m.state_dict(),'source_sha256':sha(__file__),'backbone_seed':seed,'head_seed':20261105+seed,'epochs':20,'fit_ids':training[arch]['image_ids'][tr].tolist()},mlpfile)
            save(out/'training.json',{'history':history,'parameters':sum(p.numel() for p in m.parameters()),'architecture':'1024-2048-2048-1024 GELU','objective':'cosine regression to training R4 neural embedding','epochs':20,'batch':256,'lr':.001,'wd':.0001,'fixed_beta':.75,'method_boundary':'Paper Gallery Anchoring adaptation; not full NEAR/HCF/Noise2Noise reproduction'})
            del opt,vx,yy
        result[arch+'_MLP']=m.eval()
    del fit_visual;return result

def raw_components(data,training,models,tr,task):
    q=task['queries'];rows=task['rows'];v=data['visual'].astype(float);g=[];iso=[];rep=[];block=[];xx=[];mm=[];dd=[]
    for arch in ['NICE','ATM']:
        m=models[arch];x=data[arch+'_latent'];mu=(v@m['visual_to_neural']+m['offset']+m['residual_mean']).reshape(-1,2,64).mean(1)
        delta=(training[arch]['group_latents'][tr,0]-training[arch]['group_latents'][tr,1])/2;difference=x[q].mean(1)[:,None]-mu[rows]
        g.append(np.einsum('nd,nkd->nk',data[arch+'_global'][q],v[rows]));iso.append(-.5*np.sum(difference**2,-1));rep.append(-.5*np.sum(difference**2/shrunk_variance(delta),-1))
        c=covariance(delta,'joint');block.append(-.5*np.einsum('nkd,de,nke->nk',difference,np.linalg.inv(c),difference,optimize=True));xx.append(x[q].mean(1));mm.append(mu[rows]);dd.append(delta)
    base=g+rep;added=block[0]+block[1];out={'NICE_global':[g[0]],'ATM_global':[g[1]],'global2':g,'isotropic4':g+iso,'repeat4':base,'block_distance':base+[added],
        'NICE_repeat':[g[0],rep[0]],'NICE_block':[g[0],rep[0],block[0]],'ATM_repeat':[g[1],rep[1]],'ATM_block':[g[1],rep[1],block[1]],'block_only':[added],'no_global':rep+[added]}
    difference=np.concatenate(xx,-1)[:,None]-np.concatenate(mm,-1);z=np.concatenate(dd,-1);z-=z.mean(0);emp=z.T@z/len(z);emp[:64,64:]=0;emp[64:,:64]=0
    for lam,name in [(.05,'shrink_005'),(.2,'shrink_020')]:
        c=(1-lam)*emp+lam*np.diag(np.diag(emp));extra=-.5*np.einsum('nkd,de,nke->nk',difference,np.linalg.inv(c),difference,optimize=True);out[name]=base+[extra]
    return out

@torch.no_grad()
def external_components(data,task,predictors,sattc_raw):
    q=task['queries'];rows=task['rows'];v=data['visual'].astype(float);out={};mixed={}
    for arch in ['NICE','ATM']:
        x=data[arch+'_global'];glob=np.einsum('nd,nkd->nk',x[q],v[rows])
        for kind in ['ridge','MLP']:
            predictor=predictors[arch+'_'+kind]
            if kind=='ridge':hyp=v@predictor[0]+predictor[1]
            else:
                hyp=np.concatenate([predictor(torch.tensor(v[i:i+256],dtype=torch.float32,device='cuda')).cpu().numpy() for i in range(0,len(v),256)])
            hyp=hyp/np.linalg.norm(hyp,axis=1,keepdims=True).clip(1e-12);local=np.einsum('nd,nkd->nk',x[q],hyp[rows]);mixed[arch+'_'+kind]=.25*glob+.75*local
            out[arch+'_NEARG_'+kind]=[mixed[arch+'_'+kind]]
        out[arch+'_SATTC']=[sattc_raw[arch]]
    for kind in ['ridge','MLP']:out['dual_NEARG_'+kind]=[mixed['NICE_'+kind],mixed['ATM_'+kind]]
    out['dual_SATTC']=[sattc_raw['NICE'],sattc_raw['ATM']];return out

def sattc_task(data,task,ns):
    q=task['queries'];rows=task['rows'];n=task['calibration_n'];result={}
    pools=[np.arange(n),np.arange(n,len(q))] if n else [np.arange(len(q))]
    for arch in ['NICE','ATM']:
        z=np.empty(rows.shape,dtype=float)
        for positions in pools:
            query=q[positions];pool=np.unique(rows[positions]);index={int(x):i for i,x in enumerate(pool)}
            s=sattc_score(data[arch+'_global'][query],data['visual'][pool],ns)
            take=np.array([[index[int(j)] for j in row] for row in rows[positions]])
            z[positions]=np.take_along_axis(s,take,axis=1)
        result[arch]=z
    return result

def fit_and_apply(comp,y,n,objective='brier',parameters=None):
    if parameters is None:
        _,parameters=fit_stack([s[:n] for s in comp],y[:n],np.ones(n,dtype=bool),objective)
    return apply_stack([s[n:] for s in comp],parameters),parameters

def fusion_controls(comp,y,n,frozen=None):
    h=np.stack([s-s.mean(1,keepdims=True) for s in comp],-1);scale=h[:n].std((0,1)).clip(1e-8) if frozen is None else np.array(frozen['scale']);h/=scale;z=h.mean(-1)
    t=temperature(z[:n],y[:n]) if frozen is None else frozen['temperature'];raw=z[n:];out={'block_equal_temperature':z[n:]/t,'block_equal_uncalibrated':raw}
    tt=[temperature(s[:n],y[:n]) for s in comp] if frozen is None else frozen['component_temperatures'];prob=sum(softmax(s/t0,axis=1) for s,t0 in zip(comp,tt))/len(comp)
    out['block_equal_probability']=np.log(prob[n:].clip(1e-12));return out,{'scale':scale.tolist(),'temperature':t,'component_temperatures':tt}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--subjects',nargs='+',default=SUBJECTS);a=p.parse_args();torch.set_num_threads(1)
    root=ROOT/'results/closeout_experiments_v64';root.mkdir(exist_ok=True)
    config={'source_sha256':sha(__file__),'mainline':'block_distance unchanged','subjects':SUBJECTS,'seeds':SEEDS,'main_comparator':'repeat4',
        'matrix':CORE+EXTERNAL,'tasks':['within10','cross10','official200'],'calibration':'100 concepts only; official transfers cross10 parameters',
        'NEARG':'Reimplementation of Gallery Anchoring only, ridge and two-hidden-layer2048 MLP, beta0.75; no full NEAR/Query Anchoring/LOSO claim',
        'SATTC':{'repository':'https://github.com/QunjieHuang/SATTC-CVPR2026','commit':COMMIT,'numerical_source_sha256':sha(SOURCE),'scope':'Official label-free postprocessor on adapted within-subject encoders, complete unlabeled query/gallery pool; restrict lists to shared K10 candidates afterward. Extra unlabeled batch context, reported separately from inductive methods.'},
        'external_fit_epochs':20,'no_checkpoint_selection':True,'no_test_method_replacement':True,'decision_status':'Closing an already selected method; all new variants are controls.'}
    path=root/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:save(path,config)
    v=vision();ns=author_functions()
    for seed in a.seeds:
        for subject in a.subjects:
            folder=root/subject/f'seed{seed}';resultfile=folder/'summary.json'
            if resultfile.exists():continue
            while not all((model_root(subject,seed)/m/'manifest.json').exists() for m in ['NICE','ATM']):
                print('WAIT_BACKBONE',subject,seed,flush=True);time.sleep(20)
            folder.mkdir(parents=True,exist_ok=True);print('CLOSEOUT_START',subject,seed,flush=True);data=bank(subject,seed,v);training,models,tr,splits=load_training(subject,seed,v)
            predictors=near_gallery(training,models,tr,v,training['NICE']['original_indices'],folder/'nearg',seed);result={};parameters={};fusion_parameters={}
            for taskname,task in tasks(data).items():
                comp=raw_components(data,training,models,tr,task);sattc=sattc_task(data,task,ns);comp.update(external_components(data,task,predictors,sattc))
                y=task['targets'];n=task['calibration_n'];result[taskname]={};parameters[taskname]={}
                for method,scores in comp.items():
                    z,pa=fit_and_apply(scores,y,n,parameters=parameters['cross10'][method] if n==0 else None)
                    result[taskname][method]=evaluate_metrics(z,y[n:])[0];parameters[taskname][method]=pa
                    np.savez_compressed(folder/(taskname+'_'+method+'.npz'),logits=z,targets=y[n:],query_ids=data['image_ids'][task['queries'][n:]],candidate_ids=data['image_ids'][task['rows'][n:]],concepts=data['class_ids'][task['queries'][n:]])
                    if seed==2028 and method in ['NICE_global','ATM_global','global2','isotropic4','repeat4','block_distance']:
                        if taskname=='official200':reference=ROOT/'results/official200_v54'/subject/(method+'.npz')
                        else:reference=ROOT/'results/diagnostic_final_v52'/subject/(taskname+'_'+method+'.npz')
                        np.testing.assert_allclose(z,np.load(reference)['logits'],atol=2e-5,rtol=1e-6)
                z,pa=fit_and_apply(comp['block_distance'],y,n,'nll',parameters=parameters['cross10']['block_nll'] if n==0 else None)
                result[taskname]['block_nll']=evaluate_metrics(z,y[n:])[0];parameters[taskname]['block_nll']=pa
                np.savez_compressed(folder/(taskname+'_block_nll.npz'),logits=z,targets=y[n:])
                fz,fp=fusion_controls(comp['block_distance'],y,n,fusion_parameters['cross10'] if n==0 else None);fusion_parameters[taskname]=fp
                for method,z in fz.items():result[taskname][method]=evaluate_metrics(z,y[n:])[0];np.savez_compressed(folder/(taskname+'_'+method+'.npz'),logits=z,targets=y[n:])
                print('CLOSEOUT_TASK',subject,seed,taskname,{m:round(result[taskname][m]['top1'],4) for m in ['repeat4','block_distance','dual_NEARG_MLP','dual_SATTC']},flush=True)
            # Candidate-count stress test is descriptive and fixed in advance.
            difficulty={}
            if seed==2028:
                labels=data['class_ids'];q=np.arange(3000)
                for k in [2,5]:
                    rc,yc=candidate_rows(labels,np.arange(1000),np.arange(1000),n=k,seed=20261103);rt,yt=candidate_rows(labels,np.arange(1000,3000),np.arange(1000,3000),n=k,seed=20261103)
                    t={'queries':q,'rows':np.r_[rc,rt]};cc=raw_components(data,training,models,tr,t);difficulty[str(k)]={}
                    for m in ['repeat4','block_distance']:
                        z,pa=fit_and_apply(cc[m],np.r_[yc,yt],1000);difficulty[str(k)][m]=evaluate_metrics(z,yt)[0]
            save(resultfile,{'configuration_sha256':sha(path),'results':result,'calibration_parameters':parameters,'fusion_parameters':fusion_parameters,'difficulty':difficulty,
                'backbone_sources':{e:sha(model_root(subject,seed)/e/'manifest.json') for e in ['NICE','ATM']},'method_unchanged':True,'SATTC_extra_unlabeled_context':True})
            del data,training,models,predictors;gc.collect();torch.cuda.empty_cache()
            completed=[{'subject':s,'seed':se} for se in SEEDS for s in SUBJECTS if (root/s/f'seed{se}/summary.json').exists()];save(root/'progress.json',{'completed':completed,'expected':30})
    print('CLOSEOUT_REQUESTED_EXPERIMENTS_COMPLETE',flush=True)

if __name__=='__main__':main()
