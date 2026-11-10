"""Actual idle-GPU latency and memory benchmark on real cached EEG inputs.

Waits for both training/evaluation workers, then measures batch1/32 at K10/200.
Wall-clock query time includes group averaging, H2D transfer, frozen encoders,
latent projection, candidate matching, fusion and softmax. It excludes EEG
acquisition, disk loading, common RN50 extraction and offline gallery fitting.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,gc,platform,subprocess
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax
from native_temporal_screen import ROOT,sha,save
from evaluation_banks_v62 import bank,vision,load_training,model_root
from closeout_experiments_v64 import GalleryMLP
from train_all_backbones_v42 import Nice,forward
from atm_control_v9 import author_classes
from protocol import repeat_groups
from variance_source_controls_v13 import shrunk_variance
from discriminant_evidence_v40 import covariance
from evaluate_diagnostic_v52 import apply_stack

METHODS=['block_distance','compact']

def wait_for_workers():
    root=ROOT/'results/final_hybrid_v76';notified=False
    while not all((root/f'sub-{i:02}'/f'seed{s}/summary.json').exists() for i in range(1,11) for s in [2028,2029,2030]):
        if not notified:print('COST_WAIT_FOR_ALL_TRAINING_AND_EVALUATION',flush=True);notified=True
        time.sleep(20)
    # Let the final evaluation process release CUDA objects/context.
    time.sleep(3)

def main():
    wait_for_workers();torch.set_num_threads(1);folder=ROOT/'results/cost_benchmark_v89'
    if (folder/'summary.json').exists():raise FileExistsError('Preserve completed timings')
    folder.mkdir(exist_ok=True);v=vision();subject='sub-01';seed=2028;data=bank(subject,seed,v);training,mapping,tr,splits=load_training(subject,seed,v)
    all_ids=np.load(ROOT/'processed'/subject/'train_identities.npz');queries=np.arange(1000,1032);selected=data['original_indices'][queries];raw=np.load(ROOT/'processed'/subject/'train_eeg.npy',mmap_mode='r');eeg=np.asarray(raw[selected],dtype=np.float32)
    pairs=[repeat_groups(s) for s in all_ids['session_id'][selected]];templates={};variances={};precisions={}
    for arch in ['NICE','ATM']:
        m=mapping[arch];templates[arch]=(data['visual'].astype(float)@m['visual_to_neural']+m['offset']+m['residual_mean']).reshape(-1,2,64).mean(1)
        x=training[arch]['group_latents'][tr];delta=(x[:,0]-x[:,1])/2;variances[arch]=shrunk_variance(delta);precisions[arch]=np.linalg.inv(covariance(delta,'joint'))
    anchors={};offline_begin=time.perf_counter()
    for arch in ['NICE','ATM']:
        ck=torch.load(ROOT/f'results/closeout_experiments_v64/{subject}/seed2028/nearg/{arch}/mlp.pt',map_location='cpu',weights_only=True)
        mlp=GalleryMLP().cuda();mlp.load_state_dict(ck['model']);mlp.eval()
        with torch.no_grad():
            anchors[arch]=np.concatenate([F.normalize(mlp(torch.tensor(data['visual'][i:i+256],dtype=torch.float32,device='cuda')),dim=-1).cpu().numpy() for i in range(0,len(data['visual']),256)])
        del mlp,ck;gc.collect();torch.cuda.empty_cache()
    gallery_precompute_seconds=time.perf_counter()-offline_begin
    cal=json.loads((ROOT/'results/closeout_experiments_v64'/subject/'seed2028/summary.json').read_text())['calibration_parameters']
    hybridcal=json.loads((ROOT/'results/compact_audit_v86'/subject/'seed2028/summary.json').read_text())['parameters'];ns,_=author_classes();records=[]
    hardware={'CPU':platform.processor(),'platform':platform.platform(),'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'CPU_threads':1}
    try:hardware['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    except Exception:pass
    for method in METHODS:
        arches=['NICE'] if method.startswith('NICE') else ['ATM'] if method.startswith('ATM') else ['NICE','ATM'];models={}
        for arch in arches:
            ck=torch.load(model_root(subject,seed)/arch/'model.pt',map_location='cpu',weights_only=True);m=Nice() if arch=='NICE' else ns['ATMS']();m.load_state_dict(ck['model']);models[arch]=m.cuda().eval()
        for batch in [1,32]:
            for k in [10,200]:
                if k==10:rows=np.flatnonzero(data['class_ids']==data['class_ids'][1000])[:10]
                else:rows=np.arange(3000,3200)
                gallery=data['visual'][rows].astype(float);mu={a:templates[a][rows] for a in arches};anchor_gallery={a:anchors[a][rows] for a in arches};pa=(hybridcal if method=='compact' else cal)['within10' if k==10 else 'official200'][method]
                def matching(encoded):
                    gg=[];rr=[];bb=[];anchor_scores=[]
                    for a in arches:
                        z=encoded[a];gg.append(z[:,-1]@gallery.T)
                        if method=='compact':anchor_scores.append(.25*gg[-1]+.75*(z[:,-1]@anchor_gallery[a].T))
                        if method not in ['NICE_global','ATM_global']:
                            m=mapping[a];latent=((z[:,:2].reshape(-1,1024).astype(float)-m['neural_center'])@m['neural_basis']/m['neural_scale']).reshape(batch,2,64).mean(1)
                            diff=latent[:,None]-mu[a];rr.append(-.5*np.sum(diff**2/variances[a],axis=-1));bb.append(-.5*np.einsum('nkd,de,nke->nk',diff,precisions[a],diff,optimize=True))
                    if method in ['NICE_global','ATM_global']:components=gg
                    elif method=='repeat4':components=gg+rr
                    elif method=='compact':components=[sum(bb)]+anchor_scores
                    else:components=gg+rr+[sum(bb)]
                    return softmax(apply_stack(components,pa),axis=1)
                @torch.no_grad()
                def query():
                    rr=eeg[:batch];needs_local=method not in ['NICE_global','ATM_global']
                    if needs_local:
                        p=pairs[:batch];a=np.stack([r[s[0]].mean(0) for r,s in zip(rr,p)]);b=np.stack([r[s[1]].mean(0) for r,s in zip(rr,p)]);means=np.stack([a,b,rr.mean(1)],axis=1)
                    else:means=rr.mean(1)[:,None]
                    x=torch.tensor(means.reshape(-1,63,250),device='cuda');encoded={}
                    for a,m in models.items():
                        if a=='NICE':t=m.encoder[0](x[:,None]).half().float();z=F.normalize(m.projector(t.flatten(1)),dim=-1)
                        else:z=F.normalize(forward(m,x,a),dim=-1)
                        encoded[a]=z.reshape(batch,3 if needs_local else 1,1024).cpu().numpy()
                    return matching(encoded)
                for _ in range(10):query()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base_allocated=torch.cuda.memory_allocated();wall=[]
                for _ in range(100):
                    start=time.perf_counter();result=query();torch.cuda.synchronize();wall.append((time.perf_counter()-start)*1000)
                assert result.shape==(batch,k) and np.allclose(result.sum(1),1)
                row={'method':method,'batch':batch,'K':k,'warmup':10,'repetitions':100,'median_ms_per_batch':float(np.median(wall)),'p95_ms_per_batch':float(np.quantile(wall,.95)),
                    'median_ms_per_query':float(np.median(wall)/batch),'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2,'working_GPU_MiB':(torch.cuda.max_memory_allocated()-base_allocated)/1024**2,
                    'encoder_parameters':sum(p.numel() for m in models.values() for p in m.parameters()),'EEG_encoder_forwards_per_query':len(arches)*(1 if method in ['NICE_global','ATM_global'] else 3),
                    'image_to_neural_parameters':0 if method in ['NICE_global','ATM_global'] else sum(mapping[a]['visual_to_neural'].size+mapping[a]['offset'].size for a in arches),
                    'fusion_parameters':len(pa['weights']),'gallery_anchor_parameters':16787456 if method=='compact' else 0,'MLP_forwards_at_query_time':0}
                records.append(row);print('MEASURED_COST',json.dumps(row),flush=True)
        del models,ck,m;gc.collect();torch.cuda.empty_cache()
    save(folder/'summary.json',{'source_sha256':sha(__file__),'hardware':hardware,'records':records,'exclusive_after_training':True,'gallery_MLP_precompute_seconds_3200_images_two_encoders':gallery_precompute_seconds,
        'boundary':'Real normalized EEG inputs; cached visual/neural candidate templates; CPU-GPU query wall-clock, not acquisition/disk/feature-extraction time; measured on subject01 seed2028 weights.'})
    print('COST_BENCHMARK_COMPLETE',flush=True)

if __name__=='__main__':main()
