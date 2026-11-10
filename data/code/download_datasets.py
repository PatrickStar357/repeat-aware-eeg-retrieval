"""Resumable, checksum-verified downloads to the local data directory only.

Inventories are the NEMAR pinned manifest and Hugging Face repository trees.
Existing source files are never removed. Resume state is kept in SQLite.
"""
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,re,sqlite3,threading,time,shutil,os

ROOT=Path(__file__).resolve().parents[1]
LOCK=threading.Lock(); TRANSFER=0; DONE=0; ERRORS=[]
DB=None

def digest(path,algorithm):
    h=hashlib.sha256() if algorithm=='sha256' else hashlib.sha1()
    if algorithm=='git':h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def inventory(groups):
    jobs=[]
    if 'raw' in groups:
        rows=json.loads((ROOT/'metadata'/'nemar_manifest.json').read_text(encoding='utf8'))
        for r in rows:
            p=r['path']
            if p.startswith(('derivatives/','sourcedata/','.')):continue
            if p.startswith('sub-'):
                if 'task-rest' in p:continue
            elif '/' in p:continue
            if p.lower().endswith(('.zip','.npy','.fif','.mat')):continue
            jobs.append({'path':'THINGS-EEG2/raw/'+p,'url':r['bytes_url'],
                         'size':r['size'],'algorithm':r['checksum_algorithm'],'hash':r['checksum']})
    for group,filename,repo,prefix in [
        ('images','things_all.json','Haitao999/things-eeg','THINGS-EEG2/stimuli'),
        ('features','things_all.json','Haitao999/things-eeg','THINGS-EEG2/visual_features'),
        ('editbench','editbench_all.json','xiaozgg/EEG-EditBench','EEG-EditBench')]:
        if group not in groups:continue
        rows=json.loads((ROOT/'metadata'/filename).read_text(encoding='utf8'))
        info=json.loads((ROOT/'metadata'/('things_repo.json' if group!='editbench' else 'editbench_repo.json')).read_text())
        revision=info['sha']
        for r in rows:
            if r['type']!='file':continue
            p=r['path']
            if group=='images' and not (p.startswith('Image_set/') or p.startswith('image_set/')):continue
            if group=='images' and p.endswith('.DS_Store'):continue
            if group=='features' and not ('RN50' in p and p.startswith('Image_feature/') and not 'RN50x' in p):continue
            if group=='features' and not any(x in p for x in ('DirectT/','FoveaBlur/')):continue
            if group=='features' and 'intra-subject' not in p:continue
            if group=='editbench' and p=='.gitattributes':continue
            lfs=r.get('lfs')
            jobs.append({'path':prefix+'/'+p,'url':f'https://huggingface.co/datasets/{repo}/resolve/{revision}/'+quote(p,safe='/'),
                         'size':r['size'],'algorithm':'sha256' if lfs else 'git','hash':lfs['oid'] if lfs else r['oid']})
    if 'images' in groups:
        rows=json.loads((ROOT/'metadata'/'nemar_manifest.json').read_text(encoding='utf8'))
        for r in rows:
            if r['path'] not in ('stimuli/image_metadata.npy','stimuli/LICENSE.txt','stimuli/stimuli.json','stimuli/stimuli.tsv'):continue
            jobs.append({'path':'THINGS-EEG2/'+r['path'],'url':r['bytes_url'],'size':r['size'],
                         'algorithm':r['checksum_algorithm'],'hash':r['checksum']})
    return jobs

def state(row,status,message=''):
    dest=ROOT/row['path']
    mtime=dest.stat().st_mtime_ns if dest.exists() else 0
    with LOCK:
        DB.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?,?,?)',
                   (row['path'],row['size'],row['algorithm'],row['hash'],status,mtime,message,time.time()))
        DB.commit()

def fetch(row):
    global TRANSFER,DONE
    dest=(ROOT/row['path']).resolve()
    if not dest.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe source path')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists() and '/visual_features/' in row['path']:
        # Reuse previously verified bytes on the same volume without a second copy.
        legacy=ROOT.parent/'outputs'/'对标论文复现及创新设计'/'对标论文复现代码'/'data'
        name=Path(row['path']).name
        alias=('ubp_' if '/FoveaBlur/' in row['path'] else 'image_')+('test.pt' if name.endswith('_test.pt') else 'train.pt')
        existing=legacy/alias
        if existing.is_file() and existing.stat().st_size==row['size'] and digest(existing,row['algorithm'])==row['hash']:
            os.link(existing,dest)
    if dest.exists():
        with LOCK:cached=DB.execute('SELECT status,mtime,checksum FROM files WHERE path=?',(row['path'],)).fetchone()
        if dest.stat().st_size==row['size'] and ((cached and cached[0]=='verified' and cached[1]==dest.stat().st_mtime_ns and cached[2]==row['hash']) or digest(dest,row['algorithm'])==row['hash']):
            state(row,'verified')
            with LOCK:DONE+=1
            return
        raise ValueError('Existing file differs; preserved for review: '+str(dest))
    partial=dest.with_name(dest.name+'.partial')
    last=None
    for attempt in range(12):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>row['size']:raise ValueError('Partial larger than manifest')
            if offset<row['size']:
                headers={'User-Agent':'Mozilla/5.0 EEGResearch/1.0','Accept-Encoding':'identity'}
                if offset:headers['Range']=f'bytes={offset}-'
                response=urlopen(Request(row['url'],headers=headers),timeout=90)
                if offset and response.status!=206:offset=0
                if response.status==206:
                    value=response.headers.get('Content-Range','')
                    if not value.startswith(f'bytes {offset}-'):raise ValueError('Invalid Content-Range: '+value)
                with response,partial.open('ab' if offset else 'wb') as f:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        f.write(block)
                        with LOCK:TRANSFER+=len(block)
            if partial.stat().st_size!=row['size']:
                raise ValueError(f'Incomplete bytes: {partial.stat().st_size}/{row["size"]}')
            actual=digest(partial,row['algorithm'])
            if actual!=row['hash']:
                # Keep the mismatched bytes, do not silently delete user/source data.
                partial.rename(partial.with_name(partial.name+f'.checksum_mismatch_{int(time.time())}'))
                raise ValueError('Checksum mismatch')
            partial.replace(dest);state(row,'verified')
            with LOCK:DONE+=1
            return
        except Exception as e:
            last=e;state(row,'retry',str(e))
            time.sleep(min(2**min(attempt,4),20))
    state(row,'failed',repr(last))
    with LOCK:ERRORS.append((row['path'],repr(last)))

def main():
    global DB
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups',nargs='+',choices=['raw','images','features','editbench'],default=['raw','images','features','editbench'])
    p.add_argument('--workers',type=int,default=8);p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();jobs=inventory(args.groups)
    total=sum(r['size'] for r in jobs)
    print('PLAN',len(jobs),'files',f'{total/1e9:.3f} GB',flush=True)
    if args.plan_only:return
    missing=sum(max(0,r['size']-(ROOT/r['path']).stat().st_size) if (ROOT/r['path']).is_file() else r['size'] for r in jobs)
    if shutil.disk_usage(ROOT).free<missing+8*1024**3:raise RuntimeError('Insufficient free space, including 8GiB reserve')
    DB=sqlite3.connect(ROOT/'metadata'/'downloads.sqlite',check_same_thread=False,timeout=60)
    DB.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,size INTEGER,algorithm TEXT,checksum TEXT,status TEXT,mtime INTEGER,message TEXT,updated REAL)');DB.commit()
    start=time.time();stop=threading.Event()
    def progress():
        while not stop.wait(20):
            with LOCK:done,transferred,failed=DONE,TRANSFER,len(ERRORS)
            print(f'PROGRESS verified={done}/{len(jobs)} transferred={transferred/1e9:.3f}GB rate={transferred/max(1,time.time()-start)/1e6:.2f}MB/s failed={failed}',flush=True)
    monitor=threading.Thread(target=progress,daemon=True);monitor.start()
    try:
        # Raw EEG first, within each subject events/small metadata before large recordings.
        jobs.sort(key=lambda r: (r['size']>1024*1024,r['path']))
        with ThreadPoolExecutor(args.workers) as pool:
            futures=[pool.submit(fetch,r) for r in jobs]
            for f in as_completed(futures):f.result()
    finally:stop.set();DB.close()
    print(f'COMPLETE verified={DONE}/{len(jobs)} failed={len(ERRORS)} elapsed={time.time()-start:.1f}s',flush=True)
    if ERRORS:
        for e in ERRORS[:20]:print('FAILED',*e,flush=True)
        raise SystemExit(1)
if __name__=='__main__':main()
