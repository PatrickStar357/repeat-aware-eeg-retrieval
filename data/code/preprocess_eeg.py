"""Strict train-category-only preprocessing of local THINGS-EEG2 BIDS data.

No downloading, no fitted statistics from validation/calibration/test classes.
Default: 63 EEG channels, per-trial [-200,0) ms baseline, polyphase downsampling
1000 -> 250 Hz, retain [0,1000) ms; no additional bandpass or MVNN.
The new split differs from published whitened/posterior-channel protocols.
Use --smoke to validate an actual recording without preparing a full subject.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sqlite3,shutil
from collections import defaultdict
import numpy as np
from scipy.signal import resample_poly
import mne
ROOT=Path(__file__).resolve().parents[1]

def recording_epochs(bdf,rows,batch_size=32):
    """Yield (trial rows, baseline-corrected 250-Hz epochs), modest RAM per run."""
    table=bdf.with_name(bdf.name.replace('_eeg.bdf','_channels.tsv'))
    with table.open(encoding='utf-8-sig',newline='') as f:
        names=[r['name'] for r in csv.DictReader(f,delimiter='\t') if r['type'].upper()=='EEG']
    raw=mne.io.read_raw_bdf(bdf,preload=False,verbose=False)
    if raw.info['sfreq']!=1000:raise ValueError('Unexpected source sampling rate')
    if len(names)!=63 or not set(names)<=set(raw.ch_names):raise ValueError('EEG channel identity mismatch')
    signal=raw.get_data(picks=names).astype(np.float32);raw.close()
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size];epochs=[]
        for row in chunk:
            s=int(row['sample'])
            if s<200 or s+1000>signal.shape[1]:raise ValueError('Epoch outside source recording')
            e=signal[:,s-200:s+1000].copy()
            e-=e[:,:200].mean(axis=-1,keepdims=True)
            # 1,200 samples -> 300; first 50 are the prestimulus baseline.
            e=resample_poly(e,1,4,axis=-1)[:,50:300].astype(np.float32)
            if e.shape!=(63,250) or not np.isfinite(e).all():raise ValueError('Invalid epoched EEG')
            epochs.append(e)
        yield chunk,np.stack(epochs),names

def prepare(subject,smoke=False,dtype='float16'):
    db=sqlite3.connect(ROOT/'metadata/data_index.sqlite');db.row_factory=sqlite3.Row
    sources=db.execute('SELECT DISTINCT source FROM trials WHERE subject=? AND selected=1 ORDER BY source',(subject,)).fetchall()
    if len(sources)!=24:raise ValueError('Need 24 indexed task recordings for this subject')
    if smoke:
        source=sources[1]['source'] # first training run, after task-test
        rows=[dict(r) for r in db.execute('SELECT * FROM trials WHERE source=? AND selected=1 ORDER BY sample LIMIT 32',(source,))]
        bdf=ROOT/source.replace('_events.tsv','_eeg.bdf')
        _,epochs,names=next(recording_epochs(bdf,rows))
        print(json.dumps({'status':'real_recording_read_and_epoched','subject':subject,
                          'source':str(bdf),'epochs':list(epochs.shape),'channels':names,
                          'rms_microvolt':float(np.sqrt(np.mean(epochs.astype(float)**2))*1e6),
                          'note':'No global fitted statistics, training or research results were produced.'},indent=2))
        return
    dest=ROOT/'processed'/subject
    if dest.exists():raise ValueError('Output exists; use a new subject output or review it explicitly')
    shapes={'train':(16540,4,63,250),'test':(200,80,63,250)}
    # Working float32 arrays plus final arrays must fit, with an 8-GiB reserve.
    required=sum(np.prod(s)*(4+np.dtype(dtype).itemsize) for s in shapes.values())+8*1024**3
    if shutil.disk_usage(ROOT).free<required:raise RuntimeError('Not enough disk for this subject and temporary arrays; process/archive one subject at a time')
    for r in sources:
        if not (ROOT/r['source'].replace('_events.tsv','_eeg.bdf')).is_file():raise FileNotFoundError(r['source']+' requires completed raw download')
    dest.mkdir(parents=True);arrays={};sessions={};recordings={};counts={}
    for t,shape in shapes.items():
        arrays[t]=np.lib.format.open_memmap(dest/(t+'_working.npy'),mode='w+',dtype='float32',shape=shape)
        sessions[t]=np.empty(shape[:2],dtype='U6');recordings[t]=np.empty(shape[:2],dtype='U100')
        counts[t]=np.zeros(shape[0],dtype=int)
    train_classes={r[0] for r in db.execute("SELECT class_id FROM splits WHERE partition='train'")}
    stimuli={t:db.execute('SELECT image_id,class_id FROM stimuli WHERE task=? ORDER BY image',(t,)).fetchall() for t in shapes}
    total=np.zeros(63);square=np.zeros(63);n=0;channel_names=None
    for r in sources:
        source=r['source'];rows=[dict(r) for r in db.execute('SELECT * FROM trials WHERE source=? AND selected=1 ORDER BY sample',(source,))]
        bdf=ROOT/source.replace('_events.tsv','_eeg.bdf')
        for chunk,e,names in recording_epochs(bdf,rows):
            if channel_names is None:channel_names=names
            if names!=channel_names:raise ValueError('Channel order changed')
            for row,epoch in zip(chunk,e):
                t=row['task'];i=row['image']-1;rep=counts[t][i]
                arrays[t][i,rep]=epoch;sessions[t][i,rep]=row['session']
                recordings[t][i,rep]=row['run']+':'+str(row['sample']);counts[t][i]+=1
                if t=='train' and stimuli[t][i]['class_id'] in train_classes:
                    z=epoch.astype(np.float64);total+=z.sum(axis=1);square+=(z*z).sum(axis=1);n+=z.shape[1]
        print('EPOCHED',subject,bdf.name,flush=True)
    for t in shapes:
        if not np.all(counts[t]==shapes[t][1]):raise ValueError('Repetition count mismatch after extraction')
    mean=total/n;std=np.sqrt(np.maximum(square/n-mean**2,1e-20))
    # Save fitted parameters before float16 conversion; raw volts are never cast to float16.
    np.savez(dest/'preprocessing.npz',mean=mean,std=std,ch_names=np.array(channel_names),
             preprocessing_fit_classes=np.array(sorted(train_classes)),times=np.arange(250)/250.,sfreq=250,
             source_sfreq=1000,repeat_selection_seed=20200220,split_seed=20261102,
             baseline=np.array([-.2,0.]),note='baseline excludes time zero; train-only channel z-score; no MVNN')
    for t,shape in shapes.items():
        final=np.lib.format.open_memmap(dest/(t+'_eeg.npy'),mode='w+',dtype=dtype,shape=shape)
        for start in range(0,shape[0],64):
            z=(np.asarray(arrays[t][start:start+64])-mean[None,None,:,None])/std[None,None,:,None]
            final[start:start+64]=z
            if not np.isfinite(final[start:start+64]).all():raise ValueError('Non-finite normalized output')
        final.flush();del final;arrays[t].flush();del arrays[t]
        # This exact working file was created by this invocation inside the new output directory.
        (dest/(t+'_working.npy')).unlink()
        np.savez(dest/(t+'_identities.npz'),image_id=np.array([r['image_id'] for r in stimuli[t]]),
                 class_id=np.array([r['class_id'] for r in stimuli[t]]),session_id=sessions[t],recording_id=recordings[t],subject_id=subject)
    print('PREPROCESSED',dest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',default='sub-01');p.add_argument('--smoke',action='store_true');p.add_argument('--dtype',choices=['float16','float32'],default='float16')
    args=p.parse_args();prepare(args.subject,args.smoke,args.dtype)
