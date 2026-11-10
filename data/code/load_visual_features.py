"""Load RN50 features by canonical class/filename, never dictionary order.

No duplicate feature files are written. FoveaBlur contains several dictionaries;
the level must be explicit. Edited images still require CLIP encoding; these
release files cover the original paired THINGS stimuli only.
"""
from pathlib import Path
import argparse,sqlite3
import numpy as np
import torch
from download_datasets import inventory,digest
ROOT=Path(__file__).resolve().parents[1]
def load_features(task='train',kind='baseline',level=None):
    folder=ROOT/'THINGS-EEG2/visual_features'
    files=list(folder.rglob(f'*_{kind}_EEGProjectLayer_RN50_{task}.pt'))
    if len(files)!=1:raise ValueError('Expected one pinned visual feature file')
    row=next(r for r in inventory(['features']) if (ROOT/r['path']).resolve()==files[0].resolve())
    if digest(files[0],row['algorithm'])!=row['hash']:raise ValueError('Visual feature checksum differs from pinned author release')
    # This author release stores NumPy scalar metadata as well as tensors.
    # Full unpickling is restricted here to an exact hash-verified release file.
    features=torch.load(files[0],map_location='cpu',weights_only=False)['img_features']
    if kind=='ubp':
        if level not in features:raise ValueError('Specify --level from '+str(list(features)))
        features=features[level]
    by_identity={}
    for key,value in features.items():
        parts=str(key).replace('\\','/').split('/')
        identity='/'.join(parts[-2:])
        if identity in by_identity:raise ValueError('Duplicate canonical feature identity')
        by_identity[identity]=value
    db=sqlite3.connect(ROOT/'metadata/data_index.sqlite')
    ids=[r[0] for r in db.execute('SELECT image_id FROM stimuli WHERE task=? ORDER BY image',(task,))];db.close()
    expected=[s.split('/',1)[1] for s in ids]
    if set(expected)!=set(by_identity):raise ValueError('Feature identities differ from canonical image metadata')
    matrix=torch.stack([by_identity[k].float() for k in expected]).numpy()
    if matrix.shape!=(len(ids),1024) or not np.isfinite(matrix).all():raise ValueError('Invalid RN50 feature matrix')
    if np.any(np.linalg.norm(matrix,axis=1)==0):raise ValueError('Zero visual feature')
    return matrix,np.array(ids)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=['train','test'],default='train');p.add_argument('--kind',choices=['baseline','ubp'],default='baseline');p.add_argument('--level');a=p.parse_args()
    x,ids=load_features(a.task,a.kind,a.level);print('Canonical feature identities verified:',x.shape,ids[0],ids[-1])
