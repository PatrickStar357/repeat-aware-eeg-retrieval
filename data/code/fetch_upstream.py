"""Save pinned upstream source snapshots and licences under data/code/upstream."""
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json,zipfile,hashlib
from source_inventory import get
ROOT=Path(__file__).resolve().parent
def fetch_nemar():
    rows=json.loads((ROOT.parent/'metadata/nemar_manifest.json').read_text(encoding='utf8'))
    dest=ROOT/'upstream/NEMAR';dest.mkdir(exist_ok=True,parents=True)
    for r in rows:
        if r['path'] not in ('code/convert_thingseeg2.py','code/align_stimuli.py','code/build_stim_bids.py'):continue
        b=get(r['bytes_url'])[0]
        h=hashlib.sha256() if r['checksum_algorithm']=='sha256' else hashlib.sha1()
        if r['checksum_algorithm']=='git':h.update(f'blob {len(b)}\0'.encode())
        h.update(b)
        if len(b)!=r['size'] or h.hexdigest()!=r['checksum']:raise ValueError('NEMAR code hash mismatch')
        (dest/Path(r['path']).name).write_bytes(b)
    (dest/'SOURCE_REVISION.txt').write_text('NEMAR nm000232 v1.1.1 conversion and image-identity code.\nSource and checksums: data/metadata/nemar_manifest.json\n',encoding='utf8')
    print('NEMAR conversion code verified',flush=True)
def fetch(repo):
    try:
        info=json.loads(get('https://api.github.com/repos/'+repo+'/commits?per_page=1')[0]);sha=info[0]['sha']
        raw=get(f'https://codeload.github.com/{repo}/zip/{sha}')[0]
    except Exception:
        raw=get(f'https://codeload.github.com/{repo}/zip/refs/heads/main')[0]
        # GitHub ZIP archive comment contains the resolved commit ID.
        with zipfile.ZipFile(BytesIO(raw)) as z:sha=z.comment.decode().strip()
        if len(sha)!=40:raise ValueError('Cannot establish immutable upstream revision')
    dest=ROOT/'upstream'/repo.split('/')[-1];dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(BytesIO(raw)) as z:
        for f in z.infolist():
            p=Path(*Path(f.filename).parts[1:])
            if not p.parts or f.is_dir():continue
            target=(dest/p).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError('Unsafe zip entry')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(f))
    (dest/'SOURCE_REVISION.txt').write_text(f'https://github.com/{repo}\ncommit {sha}\narchive SHA256 {hashlib.sha256(raw).hexdigest()}\n',encoding='utf8')
    print(repo,sha,len(raw),'bytes',flush=True)
if __name__=='__main__':
    with ThreadPoolExecutor(2) as pool:list(pool.map(fetch,['gifale95/eeg_encoding','XiaoZhangYES/EEG-EditBench']))
    fetch_nemar()
