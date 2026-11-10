"""Validate downloaded data against pinned inventories and recorded checksums.

Default reuses successful transfer hashes only when exact size/mtime match.
--rehash reads every byte again. --images verifies all decoded image files.
Exit 1 indicates incomplete/changed data, never success based on file counts.
"""
from pathlib import Path
from collections import Counter
import argparse,sqlite3,json,sys,time
from download_datasets import ROOT,inventory,digest

def audit(rehash=False,images=False,eeg_headers=False):
    jobs=inventory(['raw','images','features','editbench']);db=sqlite3.connect(ROOT/'metadata/downloads.sqlite')
    state={r[0]:r[1:] for r in db.execute('SELECT path,size,checksum,status,mtime FROM files')};db.close()
    stats=Counter();bytes_by=Counter();missing=[];damaged=[];headers=0
    if eeg_headers and not (ROOT/'metadata/data_index.sqlite').is_file():raise FileNotFoundError('Run build_data_index.py before --eeg-headers')
    index=sqlite3.connect(ROOT/'metadata/data_index.sqlite') if eeg_headers else None
    bounds={r[0]:(r[1],r[2]) for r in index.execute('SELECT source,min(sample),max(sample) FROM trials WHERE selected=1 GROUP BY run')} if index else {}
    for r in jobs:
        p=ROOT/r['path'];group='raw EEG + BIDS' if '/raw/' in r['path'] else ('visual features' if '/visual_features/' in r['path'] else ('EEG-EditBench' if r['path'].startswith('EEG-EditBench') else 'THINGS images + identities'))
        if not p.is_file():missing.append(r['path']);continue
        size=p.stat().st_size;cached=state.get(r['path'])
        valid=size==r['size'] and cached and cached[0]==size and cached[1]==r['hash'] and cached[2]=='verified' and cached[3]==p.stat().st_mtime_ns
        if rehash or not valid:valid=size==r['size'] and digest(p,r['algorithm'])==r['hash']
        if not valid:damaged.append(r['path']);continue
        stats[group]+=1;bytes_by[group]+=size
        if images and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
            from PIL import Image
            try:
                with Image.open(p) as im:im.verify()
            except Exception:damaged.append(r['path']);stats[group]-=1;bytes_by[group]-=size
        if eeg_headers and p.suffix.lower()=='.bdf':
            import mne,csv
            raw=mne.io.read_raw_bdf(p,preload=False,verbose=False)
            table=p.with_name(p.name.replace('_eeg.bdf','_channels.tsv'))
            with table.open(encoding='utf-8-sig') as f:
                names=[v['name'] for v in csv.DictReader(f,delimiter='\t') if v['type'].upper()=='EEG']
            source=r['path'].replace('_eeg.bdf','_events.tsv')
            lo,hi=bounds.get(source,(None,None))
            if raw.info['sfreq']!=1000 or names!=raw.ch_names or len(names)!=63 or lo is None or lo<200 or hi+1000>raw.n_times:
                damaged.append(r['path']);stats[group]-=1;bytes_by[group]-=size
            else:headers+=1
            raw.close()
    if index:index.close()
    done=sum(stats.values());total=sum(r['size'] for r in jobs)
    print(json.dumps({'verified':done,'required':len(jobs),'required_bytes':total,'missing':len(missing),'damaged':len(damaged),'eeg_headers_checked':headers,'groups':dict(stats)},indent=2))
    text=['# 数据文件验收','',f'核验时间：{time.strftime("%Y-%m-%d %H:%M:%S")}（本机时间）。',
          '',f'完成：{done:,} / {len(jobs):,} 个必需文件；缺少 {len(missing):,} 个；校验异常 {len(damaged):,} 个。',
          f'必需数据合计 {total/1e9:.3f} GB（{total/1024**3:.3f} GiB）。不包含可重建缓存及代码运行库。','',
          '| 数据组 | 已核验文件 | 字节数 |','|---|---:|---:|']
    text += [f'| {g} | {stats[g]:,} | {bytes_by[g]:,} |' for g in sorted(stats)]
    text += ['','核验方式：固定版本清单中的大小与 SHA256 / Git blob SHA1；只有大小和修改时间均未改变时才复用下载时已计算的哈希。',
             '本次额外重新读取全部字节。' if rehash else '可用 `--rehash` 对所有字节再验一遍。',
             '本次另用 Pillow 检查全部图片文件结构。' if images else '可用 `--images` 验证图片文件结构。',
             f'本次另外核实 {headers} 份BDF的63个实际通道、1000Hz源采样率，以及选用事件的[-200,1000)ms分段边界。' if eeg_headers else '可用 `--eeg-headers` 核实全部BDF头部与选用事件边界。',
             '', '此验收只证明文件完整性，不代表神经解码实验已经完成。']
    if missing:text+=['','尚缺（前20项）：']+['- '+p for p in missing[:20]]
    if damaged:text+=['','异常（前20项）：']+['- '+p for p in damaged[:20]]
    (ROOT/'数据验收.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    return not missing and not damaged
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rehash',action='store_true');p.add_argument('--images',action='store_true');p.add_argument('--eeg-headers',action='store_true');a=p.parse_args()
    raise SystemExit(0 if audit(a.rehash,a.images,a.eeg_headers) else 1)
