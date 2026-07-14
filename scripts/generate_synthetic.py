"""Generate provenance-tagged synthetic records for optional training."""
import argparse, json, random
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--output',default='data/synthetic'); ap.add_argument('--count',type=int,default=100); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
random.seed(args.seed); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
templates=[('Bệnh nhân có {symptom}, chẩn đoán {diag}.','đau bụng','trào ngược dạ dày - thực quản'),('Tiền sử dùng {drug}, hiện không có {symptom}.','metoprolol 25mg po bid','ho'),('Xét nghiệm {test}: {value}.','WBC','14,43')]
for i in range(args.count):
    t,a,b=random.choice(templates); text=t.format(symptom=a,diag=b,drug=a,test=a,value=b)
    (out/f'{i+1}.json').write_text(json.dumps({'text':text,'source':'synthetic-template','seed':args.seed},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'wrote {args.count} synthetic records to {out}')
