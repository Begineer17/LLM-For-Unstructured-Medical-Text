"""Normalize a simple ICD/RxNorm CSV/JSONL file to concepts.jsonl."""
import argparse, csv, hashlib, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--release',default='unspecified'); args=ap.parse_args()
p=Path(args.input); rows=[]
if p.suffix.lower()=='.csv':
    with p.open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            code=r.get('code') or r.get('rxcui') or r.get('RXCUI') or r.get('id'); label=r.get('label') or r.get('name') or r.get('STR')
            if code and label: rows.append({'code':str(code),'label':label,'aliases':[x.strip() for x in (r.get('aliases') or '').split('|') if x.strip()]})
else:
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            r=json.loads(line); rows.append({'code':str(r.get('code') or r.get('rxcui') or r.get('RXCUI') or r.get('id')),'label':r.get('label') or r.get('name') or r.get('STR'),'aliases':r.get('aliases',[])})
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
print(json.dumps({'concepts':len(rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'release':args.release},indent=2))
