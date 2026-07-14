import argparse, zipfile
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--output',default='output'); ap.add_argument('--zip',default='output.zip'); args=ap.parse_args()
out=Path(args.output); files=sorted(out.glob('*.json'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
if not files: raise SystemExit(f'no JSON files in {out}')
with zipfile.ZipFile(args.zip,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p, 'output/'+p.name)
print(f'packaged {len(files)} files -> {args.zip}')
