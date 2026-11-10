"""Fixed extra-seed replication; reuse the frozen v42 trainer unchanged."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,gc,argparse,time
import torch
from native_temporal_screen import ROOT,sha,save
from train_all_backbones_v42 import Data,train

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',nargs='+',type=int,default=[2029,2030]);p.add_argument('--subjects',nargs='+',default=[f'sub-{i:02}' for i in range(1,11)]);a=p.parse_args()
    torch.set_num_threads(2);root=ROOT/'results/seed_replication_v61';root.mkdir(exist_ok=True)
    config={'source_sha256':sha(__file__),'trainer_sha256':sha(ROOT/'code/experiments/train_all_backbones_v42.py'),
        'seeds':[2029,2030],'reference_seed':2028,'epochs':20,'batch':64,'architecture':['NICE','ATM'],
        'checkpoint':'fixed final epoch','fit':'1254 training concepts only','purpose':'Repeatability check of a fixed block_distance method; not method selection'}
    path=root/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:save(path,config)
    for subject in a.subjects:
        for seed in a.seeds:
            assert seed in config['seeds'];base=root/subject/f'seed{seed}'
            if all((base/m/'manifest.json').exists() for m in ['NICE','ATM']):continue
            print('REPLICATION_START',subject,seed,flush=True);data=Data(subject)
            for arch in ['NICE','ATM']:train(data,arch,base/arch,seed,20,64)
            del data;gc.collect();torch.cuda.empty_cache()
            save(root/'progress.json',{'completed':[{'subject':s,'seed':t} for s in [f'sub-{i:02}' for i in range(1,11)] for t in config['seeds'] if all((root/s/f'seed{t}'/m/'manifest.json').exists() for m in ['NICE','ATM'])]})
    print('SEED_REPLICATION_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
