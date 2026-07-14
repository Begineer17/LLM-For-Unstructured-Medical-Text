#!/usr/bin/env python3
"""Download and normalize official ICD-10-CM and RxNorm releases.

Sources are explicit and versioned; no runtime API lookup is performed. RxNorm
full releases may require a UMLS license, so the default uses the public
prescribable release. Pass --rxnorm-url with a licensed full-release URL when
the competition permits it.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, tempfile, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path

ICD_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2026/icd10cm-Code%20Descriptions-2026.zip"
RX_URL = "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_07062026.zip"

def fetch(url: str, dst: Path):
    req=urllib.request.Request(url, headers={"User-Agent":"clinical-nlp-contest/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dst.open('wb') as f: shutil.copyfileobj(r, f)

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def build_icd(archive: Path, out: Path):
    rows=[]
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as td:
        z.extractall(td)
        for p in Path(td).rglob('*'):
            if not p.is_file(): continue
            try: text=p.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError: continue
            for line in text.splitlines():
                # CDC fixed-width code file stores A000 rather than A00.0.
                m=re.match(r'^\s*([A-Z][0-9]{2})([A-Z0-9]{1,4})\s{2,}(.+?)\s*$', line)
                if m: rows.append({'code':m.group(1)+'.'+m.group(2), 'label':m.group(3), 'aliases':[]})
    uniq={r['code']:r for r in rows}; out.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in uniq.values())+'\n',encoding='utf-8'); return len(uniq)

def build_rxnorm(archive: Path, out: Path):
    names=defaultdict(set)
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as td:
        z.extractall(td)
        files=list(Path(td).rglob('RXNCONSO.RRF'))
        if not files: raise RuntimeError('RXNCONSO.RRF not found in RxNorm archive')
        for line in files[0].read_text(encoding='utf-8').splitlines():
            f=line.split('|')
            if len(f)<18 or f[1] != 'ENG' or f[16] not in ('N','') : continue
            rxcui, tty, label = f[0], f[12], f[14]
            if tty in {'SCD','SBD','SCDC','SCDG','SBDG','IN','PIN','MIN','BN','SY','DF'} and label:
                names[rxcui].add(label)
    rows=[{'code':c,'label':sorted(v)[0],'aliases':sorted(v)} for c,v in names.items()]
    out.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8'); return len(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data/ontology'); ap.add_argument('--icd-url',default=ICD_URL); ap.add_argument('--rxnorm-url',default=RX_URL); ap.add_argument('--icd-archive'); ap.add_argument('--rxnorm-archive'); ap.add_argument('--skip-rxnorm',action='store_true'); ap.add_argument('--keep-archives',action='store_true'); args=ap.parse_args()
    root=Path(args.out); icd=root/'icd'; rx=root/'rxnorm'; icd.mkdir(parents=True,exist_ok=True); rx.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); ia=Path(args.icd_archive) if args.icd_archive else td/'icd.zip'; ra=Path(args.rxnorm_archive) if args.rxnorm_archive else td/'rxnorm.zip'
        if not args.icd_archive: print('Downloading ICD:',args.icd_url); fetch(args.icd_url,ia)
        ni=build_icd(ia,icd/'concepts.jsonl'); manifest={'icd':{'source':args.icd_archive or args.icd_url,'sha256':sha256(ia),'concepts':ni}}
        if not args.skip_rxnorm:
            if not args.rxnorm_archive: print('Downloading RxNorm:',args.rxnorm_url); fetch(args.rxnorm_url,ra)
            nr=build_rxnorm(ra,rx/'concepts.jsonl'); manifest['rxnorm']={'source':args.rxnorm_archive or args.rxnorm_url,'sha256':sha256(ra),'concepts':nr}
        if args.keep_archives:
            shutil.copy2(ia, root/'icd-release.zip')
            if not args.skip_rxnorm: shutil.copy2(ra, root/'rxnorm-release.zip')
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
