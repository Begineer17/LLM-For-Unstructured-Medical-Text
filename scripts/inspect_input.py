import argparse, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--input', default='input'); args=ap.parse_args()
files=list(Path(args.input).glob('*.txt')); chars=sum(len(p.read_text(encoding='utf-8')) for p in files)
print(json.dumps({'files':len(files),'characters':chars,'min_chars':min((len(p.read_text(encoding='utf-8')) for p in files),default=0),'max_chars':max((len(p.read_text(encoding='utf-8')) for p in files),default=0)},ensure_ascii=False,indent=2))
