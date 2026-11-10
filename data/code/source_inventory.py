"""Build pinned source inventories from official dataset endpoints."""
from urllib.request import urlopen,Request
from urllib.parse import quote
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,time
ROOT=Path(__file__).resolve().parents[1]
META=ROOT/'metadata';META.mkdir(exist_ok=True)
def get(url):
    for i in range(6):
        try:
            r=urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0 EEGResearch'}),timeout=45)
            return r.read(),dict(r.headers)
        except Exception:
            if i==5:raise
            time.sleep(min(2**i,8))
def hf_tree(repo,sub=''):
    url=f'https://huggingface.co/api/datasets/{repo}/tree/main'+('/'+quote(sub) if sub else '')+'?recursive=true&limit=1000'
    out=[]
    while url:
        b,h=get(url);out.extend(json.loads(b));links=h.get('Link',h.get('link',''))
        import re
        match=re.search(r'<([^>]+)>;\s*rel="next"',links)
        url=match.group(1) if match else None
    return out
def run(item):
    name,kind,value=item
    try:
        if kind=='url':b,_=get(value);data=json.loads(b)
        else:data=hf_tree(value)
        (META/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')
        if isinstance(data,list):
            files=[r for r in data if r.get('type')=='file']
            print(name,'entries',len(data),'files',len(files),'bytes',sum(r.get('size',0) for r in files),flush=True)
            print(name,'sample',files[:2],flush=True)
        else:print(name,'keys',list(data)[:20],str(data)[:1200],flush=True)
    except Exception as e:print(name,'ERROR',repr(e),flush=True)
if __name__=='__main__':
    jobs=[('nemar_manifest','url','https://data.nemar.org/nm000232/v1.1.1/manifest.json'),
          ('things_all','hf','Haitao999/things-eeg'),('editbench_all','hf','xiaozgg/EEG-EditBench')]
    with ThreadPoolExecutor(3) as pool:list(pool.map(run,jobs))
