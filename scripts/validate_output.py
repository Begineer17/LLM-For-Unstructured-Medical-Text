#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical_nlp_pipeline import validate_pair

ap=argparse.ArgumentParser(); ap.add_argument('--input', default='input'); ap.add_argument('--output', default='output'); args=ap.parse_args()
bad=0
for p in sorted(Path(args.input).glob('*.txt'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem):
    q=Path(args.output)/(p.stem+'.json')
    if not q.exists(): print(f'MISSING {q}'); bad+=1; continue
    try: data=json.loads(q.read_text(encoding='utf-8'))
    except Exception as e: print(f'{q}: invalid JSON: {e}'); bad+=1; continue
    err=validate_pair(p.read_text(encoding='utf-8'), data)
    if err: print(f'{q}: ' + '; '.join(err)); bad+=1
print('VALID' if not bad else f'FAILED ({bad} files)')
sys.exit(1 if bad else 0)
