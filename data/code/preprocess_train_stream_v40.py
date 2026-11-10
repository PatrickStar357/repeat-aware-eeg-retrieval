"""Two-pass THINGS-EEG2 training-pool preprocessing without float32 disk copies.

Uses the exact existing recording_epochs implementation and concept split.
Raw BDF sources are never modified. Official R80 test epochs are not duplicated.
Outputs include all 1,654 training-pool concepts, with statistics fitted solely
on 1,254 training concepts; reserved concepts are not evaluated.
"""
import argparse,json,sqlite3,shutil,hashlib,time
from pathlib import Path
import numpy as np
from preprocess_eeg import ROOT,recording_epochs
from experiments.protocol import category_split

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(subject):
    dest=ROOT/'processed'/subject
    if (dest/'stream_manifest.json').exists():
        manifest=json.loads((dest/'stream_manifest.json').read_text())
        assert manifest['source_sha256']==sha(__file__)
        print('STREAM_ALREADY_COMPLETE',subject,flush=True);return
    if dest.exists():raise FileExistsError('Preserve existing/partial processed subject: '+subject)
    shape=(16540,4,63,250);required=int(np.prod(shape))*2+3*1024**3
    if shutil.disk_usage(ROOT).free<required:raise RuntimeError('Need one float16 subject plus 3 GiB reserve')
    db=sqlite3.connect(ROOT/'metadata/data_index.sqlite');db.row_factory=sqlite3.Row
    stimuli=db.execute("SELECT image_id,class_id FROM stimuli WHERE task='train' ORDER BY image").fetchall()
    labels=np.array([r['class_id'] for r in stimuli]);split=category_split(labels);training=set(split['train'])
    assert training=={r[0] for r in db.execute("SELECT class_id FROM splits WHERE partition='train'")}
    sources=db.execute("SELECT DISTINCT source FROM trials WHERE subject=? AND task='train' AND selected=1 ORDER BY source",(subject,)).fetchall()
    if not sources:raise ValueError('Missing indexed raw subject')
    begin=time.time();total=np.zeros(63);squares=np.zeros(63);n=0;channels=None
    def epochs(source):
        rows=[dict(r) for r in db.execute("SELECT * FROM trials WHERE source=? AND task='train' AND selected=1 ORDER BY sample",(source,))]
        path=ROOT/source.replace('_events.tsv','_eeg.bdf')
        if not path.is_file():raise FileNotFoundError(path)
        return recording_epochs(path,rows)
    for row in sources:
        for rows,x,names in epochs(row['source']):
            if channels is None:channels=names
            assert names==channels
            keep=np.array([labels[r['image']-1] in training for r in rows])
            z=x[keep].astype(np.float64);total+=z.sum((0,2));squares+=(z*z).sum((0,2));n+=len(z)*250
        print('STREAM_FIT',subject,Path(row['source']).name,flush=True)
    assert n==12540*4*250
    mean=total/n;std=np.sqrt(np.maximum(squares/n-mean**2,1e-20))
    dest.mkdir(parents=True);out=np.lib.format.open_memmap(dest/'train_eeg.npy',mode='w+',dtype='float16',shape=shape)
    sessions=np.empty(shape[:2],dtype='U6');recordings=np.empty(shape[:2],dtype='U100');counts=np.zeros(shape[0],dtype=int)
    check_sum=np.zeros(63);check_square=np.zeros(63);check_n=0
    for row in sources:
        for rows,x,names in epochs(row['source']):
            z=(x-mean[None,:,None])/std[None,:,None]
            for rr,epoch in zip(rows,z):
                i=rr['image']-1;rep=counts[i];assert rep<4
                out[i,rep]=epoch;sessions[i,rep]=rr['session'];recordings[i,rep]=rr['run']+':'+str(rr['sample']);counts[i]+=1
                if labels[i] in training:
                    stored=out[i,rep].astype(np.float64);check_sum+=stored.sum(1);check_square+=(stored*stored).sum(1);check_n+=250
        out.flush();print('STREAM_WRITE',subject,Path(row['source']).name,flush=True)
    assert np.all(counts==4) and len(set(recordings.ravel()))==66160
    assert np.max(np.abs(check_sum/check_n))<1e-3
    assert np.max(np.abs(np.sqrt(check_square/check_n-(check_sum/check_n)**2)-1))<1e-3
    del out
    np.savez(dest/'preprocessing.npz',mean=mean,std=std,ch_names=np.array(channels),preprocessing_fit_classes=np.array(sorted(training)),
        times=np.arange(250)/250.,sfreq=250,source_sfreq=1000,repeat_selection_seed=20200220,split_seed=20261102,
        baseline=np.array([-.2,0.]),note='Same baseline/resample/train-only channel z-score as preprocess_eeg; two-pass float16 streaming, no MVNN')
    np.savez(dest/'train_identities.npz',image_id=np.array([r['image_id'] for r in stimuli]),class_id=labels,
        session_id=sessions,recording_id=recordings,subject_id=subject)
    manifest={'source_sha256':sha(__file__),'epoch_source_sha256':sha(Path(__file__).with_name('preprocess_eeg.py')),
        'subject':subject,'shape':list(shape),'dtype':'float16','fit_classes':sorted(training),'all_training_pool_concepts':1654,
        'normalized_train_mean_abs_max':float(np.max(np.abs(check_sum/check_n))),
        'normalized_train_std_error_max':float(np.max(np.abs(np.sqrt(check_square/check_n-(check_sum/check_n)**2)-1))),
        'official_test_preprocessed':False,'reserved_concepts_evaluated':False,'seconds':time.time()-begin}
    (dest/'stream_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('STREAM_COMPLETE',subject,round(manifest['seconds'],1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);a=p.parse_args()
    for subject in a.subjects:
        if subject not in [f'sub-{i:02}' for i in range(4,11)]:raise ValueError('This run only creates previously unprocessed subjects 04--10')
        prepare(subject)
