"""Fixed-budget selector fits and C-stage ablations for the manuscript.

Run with single-thread numerical libraries. Checkpoints allow an identical run
to resume without repeating completed prefixes. No original artifacts mutate.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn

from src.fptree import FPTree
from src.models import SparseSugeno
from src.projected_selection import select
from src.selection import CoveragePool
from src.paper_c_ablation import VARIANTS, evaluate
from paper.audit_revision_evidence import metrics, paired

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paper_budget_c_ablation'
ARMS=('coverage_500','projection_500','coverage_2000','projection_2000')


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def read_archive(path,method=None):
    groups=defaultdict(list)
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if method is None or r['method']==method:
                groups[int(r['seed']),int(r['batch_size'])].append(r)
    return groups


def write_csv(path,rows):
    assert rows
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def initialize():
    files=[Path(__file__),ROOT/'src/paper_c_ablation.py',
           ROOT/'paper/统一预算与C路消融实验协议.md',ROOT/'data/dataset_1062.txt',
           ROOT/'results/online_study/splits.json',ROOT/'src/models.py',
           ROOT/'src/selection.py',ROOT/'src/projected_selection.py',ROOT/'src/fptree.py',
           ROOT/'run_three_way_retrospective_holdout.py',
           ROOT/'parallel_work/robust_learning/rescue_panel_learning.py',
           ROOT/'parallel_work/robust_learning/rescue_geometry_learning.py',
           ROOT/'parallel_work/robust_learning/run_rescue_prefix.py',
           ROOT/'parallel_work/robust_learning/run_rescue_interactions.py',
           ROOT/'parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv',
           ROOT/'results/three_way_retrospective_holdout/predictions.csv']
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=dict(sources=sources,budgets=[500,2000],seeds=[17,29,43],batches=[50,100],
                  initial_n=300,variants=VARIANTS,archive_tolerance=1e-7,
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  sklearn=sklearn.__version__,threads={k:os.environ.get(k) for k in
                      ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
                  evaluation='Repeated internal splits and historically reused retrospective holdouts; no tuning.')
    OUT.mkdir(exist_ok=True)
    (OUT/'selection').mkdir(exist_ok=True)
    (OUT/'c_stages').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))==json.loads(json.dumps(manifest)), 'Resume requires identical inputs and runtime.'
    else:write_json(OUT/'manifest.json',manifest)
    return manifest


def selection_prefix(seed,n,x,y,query,tree,scale):
    path=OUT/'selection'/f'seed{seed}_n{n}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    print(f'SELECTION start seed={seed} n={n}',flush=True)
    tick=perf_counter()
    patterns=tree.mine(.1,3)
    pool=CoveragePool(patterns,x,y,confidence=.65,min_rows=5)
    assert all(len(v)>=32 for v in pool.pools.values())
    mine_seconds=perf_counter()-tick
    tick=perf_counter();coverage=pool.select(.5,64);coverage_seconds=perf_counter()-tick
    tick=perf_counter();projection,selector=select(pool,x,y,scale,joint=True);projection_seconds=perf_counter()-tick
    result=dict(seed=seed,n=n,candidate_count=sum(map(len,pool.pools.values())),
                mine_pool_seconds=mine_seconds,selection_seconds=dict(coverage=coverage_seconds,projection=projection_seconds),
                selector={k:v for k,v in selector.items() if k not in ('gate',)},models={})
    for budget in (500,2000):
        for strategy,rules in [('coverage',coverage),('projection',projection)]:
            assert len(rules)==64 and sum(r.label==1 for r in rules)==32
            tick=perf_counter()
            model=SparseSugeno(rules,maxiter=budget).fit(x,y,scaling=scale)
            seconds=perf_counter()-tick
            score=model.decision_function(query)
            assert np.isfinite(score).all() and np.isfinite(model.theta).all()
            result['models'][f'{strategy}_{budget}']=dict(score=score.tolist(),
                              optimization=model.optimization,fit_seconds=seconds,
                              theta=model.theta.tolist(),
                              rule_keys=[[list(r.columns),r.values.tolist(),int(r.label)] for r in rules])
    write_json(path,result)
    print(f'SELECTION done seed={seed} n={n} fit_s={sum(v["fit_seconds"] for v in result["models"].values()):.1f}',flush=True)
    return result


def selection_experiment(data,splits,archive,holdout):
    predictions=[];fits=[];checks=[]
    for split in splits:
        seed=split['seed'];ids=split['train_stream'];test=split['holdout']
        x,y=data[ids,1:],data[ids,0].astype(int)
        scale=x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.)
        sizes=sorted({849,*range(300,849,50),*range(300,849,100)})
        tree=FPTree();previous=0;cache={}
        for n in sizes:
            tree.update(x[previous:n],y[previous:n]);previous=n
            query=data[test,1:] if n==849 else x[n:min(n+100,849)]
            r=selection_prefix(seed,n,x[:n],y[:n],query,tree,scale)
            cache[n]=r
            for arm in ARMS:
                fits.append(dict(seed=seed,training_n=n,arm=arm,
                     selection_seconds=r['selection_seconds'][arm.split('_')[0]],
                     fit_seconds=r['models'][arm]['fit_seconds'],**r['models'][arm]['optimization']))
        for batch in (50,100):
            assert [int(r['row_index']) for r in archive[seed,batch]]==ids[300:]
            for evaluation,starts in [('stream',range(300,849,batch)),('holdout',[849])]:
                for n in starts:
                    query_ids=test if n==849 else ids[n:min(n+batch,849)]
                    expected=holdout[seed,batch] if n==849 else archive[seed,batch][n-300:n-300+len(query_ids)]
                    r=cache[n]
                    for arm,col in [('coverage_500','coverage'),('projection_2000','projection')]:
                        actual=np.array(r['models'][arm]['score'][:len(query_ids)])
                        saved=np.array([float(v[col]) for v in expected])
                        error=float(np.max(abs(actual-saved)))
                        np.testing.assert_allclose(actual,saved,atol=1e-7,rtol=0)
                        assert np.array_equal(actual>=0,saved>=0)
                        checks.append(dict(experiment='selection',evaluation=evaluation,seed=seed,batch=batch,n=n,arm=arm,max_abs_error=error))
                    for j,rid in enumerate(query_ids):
                        predictions.append(dict(evaluation=evaluation,seed=seed,batch=batch,training_n=n,
                              row_index=rid,label=int(data[rid,0]),
                              **{arm:r['models'][arm]['score'][j] for arm in ARMS}))
    write_csv(OUT/'selection_predictions.csv',predictions)
    write_csv(OUT/'selection_fits.csv',fits)
    write_json(OUT/'selection_replay_checks.json',checks)
    return predictions,fits,checks


def c_experiment(data,splits,archive,holdout):
    predictions=[];costs=[];checks=[]
    for split in splits:
        seed=split['seed'];ids=split['train_stream'];test=split['holdout']
        x,y=data[ids,1:],data[ids,0].astype(int)
        scale=x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.)
        for batch in (50,100):
            saved=archive[seed,batch]
            assert [int(r['row_index']) for r in saved]==ids[300:]
            history=np.array([float(r['coverage']) for r in saved])
            for evaluation,starts in [('stream',range(300,849,batch)),('holdout',[849])]:
                for n in starts:
                    query_ids=test if n==849 else ids[n:min(n+batch,849)]
                    rows=holdout[seed,batch] if n==849 else saved[n-300:n-300+len(query_ids)]
                    query=data[query_ids,1:]
                    base=np.array([float(r['coverage']) for r in rows])
                    path=OUT/'c_stages'/f'{evaluation}_seed{seed}_b{batch}_n{n}.json'
                    if path.exists():r=json.loads(path.read_text(encoding='utf-8'))
                    else:
                        print(f'C start {evaluation} seed={seed} b={batch} n={n}',flush=True)
                        output,details=evaluate(x[:n],y[:n],query,base,x[300:n],y[300:n],history[:n-300],scale)
                        r=dict(details=details,scores={k:v.tolist() for k,v in output.items()})
                        # Persist only after the original full route is verified.
                        np.testing.assert_allclose(output['full'],[float(v['new_branch']) for v in rows],atol=1e-7,rtol=0)
                        write_json(path,r)
                        print(f'C done {evaluation} seed={seed} b={batch} n={n} seconds={details["seconds"]:.1f}',flush=True)
                    a=np.array([float(v['projection']) for v in rows])
                    b=np.array([float(v['hvdm_offset']) for v in rows])
                    full=np.array(r['scores']['full']);old=np.array([float(v['new_branch']) for v in rows])
                    np.testing.assert_allclose(full,old,atol=1e-7,rtol=0)
                    assert np.array_equal(full>=0,old>=0)
                    fusion=np.mean([a,b,full],axis=0);old_fusion=np.array([float(v['new_ensemble']) for v in rows])
                    np.testing.assert_allclose(fusion,old_fusion,atol=1e-7,rtol=0)
                    assert np.array_equal(fusion>=0,old_fusion>=0)
                    checks.append(dict(evaluation=evaluation,seed=seed,batch=batch,n=n,
                          branch_error=float(np.max(abs(full-old))),fusion_error=float(np.max(abs(fusion-old_fusion)))))
                    costs.append(dict(evaluation=evaluation,seed=seed,batch=batch,n=n,**r['details']))
                    for variant in VARIANTS:
                        c=np.array(r['scores'][variant]);combined=np.mean([a,b,c],axis=0)
                        for j,rid in enumerate(query_ids):
                            predictions.append(dict(evaluation=evaluation,seed=seed,batch=batch,training_n=n,
                                  row_index=rid,label=int(data[rid,0]),variant=variant,c_score=float(c[j]),fusion_score=float(combined[j])))
    write_csv(OUT/'c_predictions.csv',predictions)
    write_json(OUT/'c_shared_costs.json',costs)
    write_json(OUT/'c_replay_checks.json',checks)
    return predictions,costs,checks


def aggregate(records,arms,column=None):
    summary={}
    for evaluation in ('stream','holdout'):
        summary[evaluation]={}
        for arm in arms:
            rows=[r for r in records if r['evaluation']==evaluation and (column is None or r['variant']==arm)]
            groups=defaultdict(list)
            for r in rows:groups[r['seed'],r['batch']].append(r)
            assert len(groups)==6
            configs=[];ys=[];ss=[]
            for (seed,batch),values in sorted(groups.items()):
                assert len(values)==(549 if evaluation=='stream' else 213)
                y=np.array([r['label'] for r in values]);score=np.array([r[arm if column is None else column] for r in values])
                configs.append(dict(seed=seed,batch=batch,**metrics(y,score)))
                ys.extend(y.tolist());ss.extend(score.tolist())
            summary[evaluation][arm]=dict(configurations=configs,
                  means={k:float(np.mean([r[k] for r in configs])) for k in
                         ('accuracy','recall','specificity','f1','balanced_accuracy','auc')},
                  pooled=metrics(np.array(ys),np.array(ss)))
    return summary


def compare(records,reference,candidate,field=None):
    outputs={}
    for evaluation in ('stream','holdout'):
        if field is None:
            rows=[r for r in records if r['evaluation']==evaluation]
            y=np.array([r['label'] for r in rows]);a=np.array([r[reference] for r in rows]);b=np.array([r[candidate] for r in rows])
            ids=[r['row_index'] for r in rows]
        else:
            refs={(r['seed'],r['batch'],r['row_index']):r for r in records if r['evaluation']==evaluation and r['variant']==reference}
            rows=[r for r in records if r['evaluation']==evaluation and r['variant']==candidate]
            y=np.array([r['label'] for r in rows]);b=np.array([r[field] for r in rows])
            a=np.array([refs[r['seed'],r['batch'],r['row_index']][field] for r in rows]);ids=[r['row_index'] for r in rows]
        ac=(a>=0)==(y==1);bc=(b>=0)==(y==1)
        outputs[evaluation]=dict(**paired(y,a,b),
              repaired_unique_records=len({i for i,flag in zip(ids,~ac&bc) if flag}),
              introduced_unique_records=len({i for i,flag in zip(ids,ac&~bc) if flag}))
    return outputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','selection','c'],default='all');args=parser.parse_args()
    initialize()
    data=np.loadtxt(ROOT/'data/dataset_1062.txt');assert data.shape==(1062,19) and np.isfinite(data).all()
    splits=json.loads((ROOT/'results/online_study/splits.json').read_text())
    assert [s['seed'] for s in splits]==[17,29,43]
    for split in splits:assert len(set(split['train_stream']))==849 and len(set(split['holdout']))==213 and not set(split['train_stream'])&set(split['holdout'])
    archive=read_archive(ROOT/'parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv','numeric_stable')
    holdout=read_archive(ROOT/'results/three_way_retrospective_holdout/predictions.csv')
    if args.phase in ('all','selection'):
        rows,fits,checks=selection_experiment(data,splits,archive,holdout)
        result=aggregate(rows,ARMS)
        result['paired']={str(b):compare(rows,f'coverage_{b}',f'projection_{b}') for b in (500,2000)}
        result['fit_status']={arm:dict(fits=sum(r['arm']==arm for r in fits),
                      successful=sum(r['arm']==arm and r['success'] for r in fits),
                      mean_iterations=float(np.mean([r['iterations'] for r in fits if r['arm']==arm]))) for arm in ARMS}
        write_json(OUT/'selection_summary.json',result)
        print('SELECTION COMPLETE '+json.dumps({a:result['stream'][a]['means'] for a in ARMS}),flush=True)
    if args.phase in ('all','c'):
        rows,costs,checks=c_experiment(data,splits,archive,holdout)
        result=dict(branch=aggregate(rows,VARIANTS,'c_score'),fusion=aggregate(rows,VARIANTS,'fusion_score'))
        result['vs_full']={v:{field:compare(rows,'full',v,field) for field in ('c_score','fusion_score')} for v in VARIANTS if v!='full'}
        write_json(OUT/'c_summary.json',result)
        print('C COMPLETE '+json.dumps({v:result['fusion']['stream'][v]['means']['accuracy'] for v in VARIANTS}),flush=True)
    write_json(OUT/f'completed_{args.phase}.json',dict(completed=True,phase=args.phase))


if __name__=='__main__':main()
