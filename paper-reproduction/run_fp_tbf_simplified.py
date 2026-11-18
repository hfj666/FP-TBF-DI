"""Run one fixed cohort/batch of the simplified method without prediction history."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_name,'1')

import numpy as np

from src.fp_tbf_simplified import ACOR_FUSION_WEIGHTS, SimplifiedFPTBF
from src.fptree import FPTree
from paper.audit_revision_evidence import metrics

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(17,29,43),default=17)
    parser.add_argument('--batch-size',type=int,choices=(50,100),default=50)
    parser.add_argument('--fusion',choices=('equal','acor'),default='equal')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    weights = ACOR_FUSION_WEIGHTS if args.fusion == 'acor' else None
    data=np.loadtxt(ROOT/'data/dataset_1062.txt')
    split=next(r for r in json.loads((ROOT/'results/online_study/splits.json').read_text()) if r['seed']==args.seed)
    ids,test=split['train_stream'],split['holdout']
    assert len(set(ids))==849 and len(set(test))==213 and not set(ids)&set(test)
    args.out.mkdir(parents=True,exist_ok=False)
    x,y=data[ids,1:],data[ids,0].astype(int)
    tree=FPTree();previous=0;rows=[];fits=[]
    for n in [*range(300,849,args.batch_size),849]:
        tree.update(x[previous:n],y[previous:n]);previous=n
        model=SimplifiedFPTBF(fusion_weights=weights).fit(x[:n],y[:n],tree)
        query_ids=ids[n:min(n+args.batch_size,849)] if n<849 else test
        score=model.decision_scores(data[query_ids,1:])
        assert np.isfinite(score).all()
        fits.append(dict(training_n=n,optimization=model.optimization,c_details=model.last_c_details))
        # Labels are attached only after full-query prediction is complete.
        for rid,values in zip(query_ids,score,strict=True):
            rows.append(dict(evaluation='stream' if n<849 else 'holdout_849',seed=args.seed,
                    batch=args.batch_size,training_n=n,row_index=rid,label=int(data[rid,0]),
                    **dict(zip(('base','logistic','A','B','C','fusion'),map(float,values),strict=True))))
        print(f'Predicted seed={args.seed} batch={args.batch_size} training_n={n}',flush=True)
    with (args.out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={}
    for ev in ('stream','holdout_849'):
        subset=[r for r in rows if r['evaluation']==ev]
        result[ev]=metrics(np.array([r['label'] for r in subset]),np.array([r['fusion'] for r in subset]))
    sources=[ROOT/'src/fp_tbf_simplified.py',ROOT/'data/dataset_1062.txt',ROOT/'results/online_study/splits.json']
    manifest=dict(method='FP-TBF-DI',seed=args.seed,batch=args.batch_size,history_predictions_used=False,
                  fusion_mode=args.fusion,
                  fusion_weights=dict(zip(('A','B','C'), weights or (1/3,)*3, strict=True)),
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for name,value in [('summary',result),('fits',fits),('manifest',manifest)]:
        (args.out/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
