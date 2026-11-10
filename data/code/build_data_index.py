"""Audit canonical image identities and index real BIDS events (no EEG decoding).

Output is a reusable SQLite data index, not a generated experimental result.
Never infer image identity from trigger `value`; use tot_img_number and verify
its category against the authors' image_metadata.npy.
"""
from pathlib import Path
import argparse,csv,json,sqlite3,hashlib,itertools
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def build(root=ROOT):
    meta=root/'THINGS-EEG2/stimuli/image_metadata.npy'
    x=np.load(meta,allow_pickle=True).item() # hash-verified author metadata only
    path=root/'metadata/data_index.sqlite'
    db=sqlite3.connect(path)
    db.execute('PRAGMA temp_store=MEMORY')
    db.executescript('''DROP TABLE IF EXISTS stimuli; DROP TABLE IF EXISTS trials;
    DROP TABLE IF EXISTS splits;
    CREATE TABLE stimuli(task TEXT,image INTEGER,class_id TEXT,image_id TEXT,path TEXT,
      PRIMARY KEY(task,image));
    CREATE TABLE trials(subject TEXT,task TEXT,image INTEGER,session TEXT,run TEXT,
      sample INTEGER,onset REAL,source TEXT,selected INTEGER DEFAULT 0);
    CREATE TABLE splits(class_id TEXT PRIMARY KEY,partition TEXT);
    ''')
    for task in ('train','test'):
        names=x[task+'_img_concepts'];files=x[task+'_img_files']
        assert len(names)==len(files)==(16540 if task=='train' else 200)
        for i,(c,f) in enumerate(zip(names,files),1):
            p=f'THINGS-EEG2/stimuli/Image_set/{"training_images" if task=="train" else "test_images"}/{c}/{f}'
            db.execute('INSERT INTO stimuli VALUES(?,?,?,?,?)',(task,i,c,task+'/'+c+'/'+f,p))
    classes=sorted(set(x['train_img_concepts']))
    assert len(classes)==1654
    assert set(Counter(x['train_img_concepts']).values())=={10}
    shuffled=np.random.default_rng(20261102).permutation(classes)
    start=0
    for name,n in [('train',1254),('validation',100),('calibration',100),('fine_test',200)]:
        db.executemany('INSERT INTO splits VALUES(?,?)',[(str(c),name) for c in shuffled[start:start+n]]);start+=n
    nfiles=0
    for f in sorted((root/'THINGS-EEG2/raw').glob('sub-*/ses-*/eeg/*_events.tsv')):
        if 'task-rest' in f.name:continue
        subject,session=f.parts[-4:-2];task='train' if 'task-train' in f.name else 'test'
        run=f.stem.replace('_events','');rows=[]
        with f.open(encoding='utf-8-sig',newline='') as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                if r['trial_type']!='image':continue
                i=int(float(r['tot_img_number']));expected=x[task+'_img_concepts'][i-1]
                if r['category_name']!=expected:raise ValueError(f'Canonical identity mismatch: {f}: {r}')
                rows.append((subject,task,i,session,run,int(r['sample']),float(r['onset']),f.relative_to(root).as_posix()))
        db.executemany('INSERT INTO trials(subject,task,image,session,run,sample,onset,source) VALUES(?,?,?,?,?,?,?,?)',rows);nfiles+=1
    assert nfiles==240, f'Need all 240 event files; found {nfiles}'
    db.executescript('CREATE INDEX trial_image ON trials(subject,task,image);CREATE INDEX trial_run ON trials(run);')
    grouped=db.execute('SELECT subject,task,image,count(*),count(DISTINCT session) FROM trials GROUP BY subject,task,image').fetchall()
    for s,t,i,n,ns in grouped:
        assert n>=(4 if t=='train' else 80),(s,t,i,n)
        assert ns==(2 if t=='train' else 4),(s,t,i,ns)
    assert len(grouped)==10*(16540+200)
    # Author preprocessing_utils.epoching: fixed-seed random subset, at most
    # 2 training / 20 test repetitions per image per session. Preserve originals.
    rows=db.execute('SELECT rowid,subject,task,image,session FROM trials ORDER BY subject,task,image,session,source,sample').fetchall()
    selected=[];permutations={}
    for key,group in itertools.groupby(rows,key=lambda r:r[1:]):
        group=list(group);limit=2 if key[1]=='train' else 20
        assert len(group)>=limit,(key,len(group))
        n=len(group)
        if n not in permutations:permutations[n]=np.random.RandomState(20200220).permutation(n)
        selected.extend((group[int(i)][0],) for i in permutations[n][:limit])
    db.executemany('UPDATE trials SET selected=1 WHERE rowid=?',selected)
    db.commit()
    report={'subjects':10,'event_files':nfiles,'training_images':16540,'official_test_images':200,
            'train_repeats':4,'official_test_repeats':80,'training_sessions_per_image':2,'test_sessions_per_image':4,
            'total_trials':db.execute('SELECT count(*) FROM trials').fetchone()[0],
            'selected_trials':len(selected),'repeat_selection_seed':20200220,
            'split_seed':20261102,'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest()}
    db.close();print(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();build(a.root)
