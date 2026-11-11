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

from src.fp_tbf_simplified import SimplifiedFPTBF
from src.fptree import FPTree
from src.metrics import metrics

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(17,29,43),default=17)
    parser.add_argument('--batch-size',type=int,choices=(50,100),default=50)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    data = np.loadtxt(args.data)
    if (data.shape != (1062, 19) or not np.isfinite(data).all()
            or set(data[:, 0]) != {-1, 1} or not np.equal(data[:, 1:], np.rint(data[:, 1:])).all()):
        parser.error('data must have 1062 rows: a -1/+1 label followed by 18 integer scores')
    records = json.loads(args.splits.read_text(encoding='utf-8'))
    split = next((r for r in records if r['seed'] == args.seed), None)
    if split is None:
        parser.error('splits does not contain the requested seed')
    ids, test = split['train_stream'], split['holdout']
    if (len(ids) != 849 or len(test) != 213 or len(set(ids + test)) != 1062
            or any(type(i) is not int or not 0 <= i < len(data) for i in ids + test)):
        parser.error('split must contain disjoint ordered zero-based row indices, 849 for training and 213 for holdout')
    args.out.mkdir(parents=True, exist_ok=False)
    x,y=data[ids,1:],data[ids,0].astype(int)
    tree=FPTree();previous=0;rows=[];fits=[]
    for n in [*range(300,849,args.batch_size),849]:
        tree.update(x[previous:n],y[previous:n]);previous=n
        model=SimplifiedFPTBF().fit(x[:n],y[:n],tree)
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
    sources={'implementation': ROOT/'src/fp_tbf_simplified.py', 'data': args.data, 'splits': args.splits}
    manifest=dict(method='FP-TBF-DI',seed=args.seed,batch=args.batch_size,history_predictions_used=False,
                  sources={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()})
    for name,value in [('summary',result),('fits',fits),('manifest',manifest)]:
        (args.out/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
