"""Prospectively specified P/E/U joint reduction, including all holdout stages."""
import csv
import hashlib
import json
import os
import platform
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn

from src.fp_tbf_simplified import SimplifiedFPTBF,predict_c
from src.fptree import FPTree
from run_three_way_retrospective_holdout import panel,geometry,rescue,deletion,interactions
from paper.audit_revision_evidence import metrics,paired
from run_paper_budget_c_ablation import read_archive,write_csv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paper_c_joint_reduction'
STAGES=(300,500,700,849)


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def native_di(x,y,query,base):
    reference=base.copy()
    negative=set(map(tuple,x[y==-1]))
    eligible=np.array([i for i,row in enumerate(query) if reference[i]<0 and tuple(row) not in negative],dtype=int)
    if len(eligible):
        full,stable=deletion(x,y,query[eligible])['categorical_deletion']
        reference[eligible[stable]]=full[stable]
    raw=np.vstack((x,query));n=len(x)
    return interactions(raw,y,n,len(raw),reference)['numeric_stable']


def native_full(x,y,query,base,previous_x,previous_y,previous_scores,scale):
    past=np.column_stack(((previous_x-scale[0])/scale[1],previous_scores))
    current=np.column_stack(((query-scale[0])/scale[1],base))
    p=panel(past,previous_y,current)[0]['rbf_margin']
    e=geometry(past,previous_y,current)[0]['exact_recurrence']
    reference=np.where((p<0)&(e>=0),e,p)
    raw=np.vstack((x,query));n=len(x)
    reference=rescue(raw,y,n,len(raw),reference)[0]['unanimous']
    negative=set(map(tuple,x[y==-1]))
    eligible=np.array([i for i,row in enumerate(query) if reference[i]<0 and tuple(row) not in negative],dtype=int)
    if len(eligible):
        full,stable=deletion(x,y,query[eligible])['categorical_deletion']
        reference[eligible[stable]]=full[stable]
    return interactions(raw,y,n,len(raw),reference)['numeric_stable']


def manifest():
    names=['run_paper_c_joint_reduction.py','src/fp_tbf_simplified.py','paper/C路联合删减实验协议.md',
           'data/dataset_1062.txt','results/online_study/splits.json',
           'src/fptree.py','src/selection.py','src/projected_selection.py','src/models.py',
           'src/reference_comparators.py','run_three_way_retrospective_holdout.py',
           'parallel_work/fuzzy_activation/activation.py','parallel_work/fuzzy_activation/run_round13.py',
           'parallel_work/robust_learning/rescue_panel_learning.py',
           'parallel_work/robust_learning/rescue_geometry_learning.py',
           'parallel_work/robust_learning/run_rescue_prefix.py',
           'parallel_work/robust_learning/run_rescue_feature_deletion.py',
           'parallel_work/robust_learning/run_rescue_interactions.py',
           'parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv',
           'results/three_way_retrospective_holdout/predictions.csv',
           'results/paper_incremental_checkpoints/checkpoint_metrics.csv']
    info=dict(candidate='base -> D -> I; P/E/U jointly removed; no historical prediction state',
              stages=STAGES,seeds=[17,29,43],batches=[50,100],tolerance=1e-7,
              python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
              threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
              sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names})
    OUT.mkdir(exist_ok=True);(OUT/'core').mkdir(exist_ok=True);(OUT/'frames').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))==json.loads(json.dumps(info))
    else:dump(OUT/'manifest.json',info)
    return info


def summarize(rows):
    result={}
    evaluations=['stream',*[f'holdout_{n}' for n in STAGES]]
    for ev in evaluations:
        result[ev]={}
        source=[r for r in rows if r['evaluation']==ev]
        groups=defaultdict(list)
        for r in source:groups[r['seed'],r['batch']].append(r)
        assert len(groups)==6
        for name,field in [('full_branch','full_c'),('simple_branch','simple_c'),('full_fusion','full_fusion'),('simple_fusion','simple_fusion')]:
            configs=[]
            for (seed,batch),values in sorted(groups.items()):
                y=np.array([r['label'] for r in values]);score=np.array([r[field] for r in values])
                assert len(y)==(549 if ev=='stream' else 213)
                configs.append(dict(seed=seed,batch=batch,**metrics(y,score)))
            result[ev][name]=dict(configurations=configs,
                  means={key:float(np.mean([r[key] for r in configs])) for key in ('accuracy','recall','specificity','f1','balanced_accuracy','auc')},
                  pooled=metrics(np.array([r['label'] for r in source]),np.array([r[field] for r in source])))
        result[ev]['comparisons']={}
        for name,original,candidate in [('branch','full_c','simple_c'),('fusion','full_fusion','simple_fusion')]:
            y=np.array([r['label'] for r in source]);a=np.array([r[original] for r in source]);b=np.array([r[candidate] for r in source])
            changed=(a>=0)!=(b>=0);old=(a>=0)==(y==1);new=(b>=0)==(y==1)
            result[ev]['comparisons'][name]=dict(**paired(y,a,b),
                   label_disagreements=int(changed.sum()),unique_changed_records=len({r['row_index'] for r,flag in zip(source,changed) if flag}),
                   repaired_fn=int(((y==1)&~old&new).sum()),introduced_fn=int(((y==1)&old&~new).sum()),
                   repaired_fp=int(((y==-1)&~old&new).sum()),introduced_fp=int(((y==-1)&old&~new).sum()),
                   score_changes=int(np.count_nonzero(a!=b)),max_abs_score_change=float(np.max(abs(a-b))),
                   mean_abs_score_change=float(np.mean(abs(a-b))))
    identical=all(result[ev]['comparisons']['fusion']['label_disagreements']==0 for ev in evaluations)
    nondominated=all(all(b[k]>=a[k]-1e-12 for k in ('balanced_accuracy','recall'))
                        for ev in evaluations for a,b in zip(result[ev]['full_fusion']['configurations'],result[ev]['simple_fusion']['configurations']))
    result['decision']=dict(all_reported_fusion_labels_identical=identical,
                           every_configuration_ba_and_recall_nondecreasing=nondominated,
                           adopt_simplified=identical or nondominated,
                           scope='Internal development evidence; AUC recomputed; no independent equivalence claim.')
    return result


def main():
    manifest()
    data=np.loadtxt(ROOT/'data/dataset_1062.txt');assert data.shape==(1062,19) and np.isfinite(data).all()
    splits=json.loads((ROOT/'results/online_study/splits.json').read_text())
    archive=read_archive(ROOT/'parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv','numeric_stable')
    holdout=read_archive(ROOT/'results/three_way_retrospective_holdout/predictions.csv')
    with (ROOT/'results/paper_incremental_checkpoints/checkpoint_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        old_checkpoints={(int(r['seed']),int(r['batch_size']),int(r['training_n'])):r for r in csv.DictReader(f)}
    rows=[];checks=[];costs=[];core_fits=[]
    for split in splits:
        seed=split['seed'];ids=split['train_stream'];test=split['holdout']
        assert len(set(ids))==849 and len(set(test))==213 and not set(ids)&set(test)
        x,y=data[ids,1:],data[ids,0].astype(int)
        scale=x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.)
        tree=FPTree();previous=0;cache={}
        for n in sorted({849,*range(300,849,50),*range(300,849,100)}):
            tree.update(x[previous:n],y[previous:n]);previous=n
            query_ids=ids[n:min(n+100,849)]+(test if n in STAGES else [])
            path=OUT/'core'/f'seed{seed}_n{n}.json'
            if path.exists():r=json.loads(path.read_text(encoding='utf-8'))
            else:
                print(f'CORE start seed={seed} n={n}',flush=True)
                tick=perf_counter();model=SimplifiedFPTBF().fit(x[:n],y[:n],tree)
                values=model.core_scores(data[query_ids,1:])
                r=dict(seed=seed,n=n,query_ids=query_ids,score={k:v.tolist() for k,v in values.items()},
                       optimization=model.optimization,seconds=perf_counter()-tick)
                assert all(np.isfinite(v).all() for v in values.values())
                dump(path,r)
                print(f'CORE done seed={seed} n={n} seconds={r["seconds"]:.1f}',flush=True)
            assert r['query_ids']==query_ids
            cache[n]=r;core_fits.append(dict(seed=seed,n=n,seconds=r['seconds'],optimization=r['optimization']))
        for batch in (50,100):
            saved=archive[seed,batch]
            assert [int(r['row_index']) for r in saved]==ids[300:]
            history=np.array([float(r['coverage']) for r in saved])
            schedule=[('stream',n) for n in range(300,849,batch)]+[(f'holdout_{n}',n) for n in STAGES]
            for evaluation,n in schedule:
                query_ids=ids[n:min(n+batch,849)] if evaluation=='stream' else test
                rawcore=cache[n];positions={rid:i for i,rid in enumerate(rawcore['query_ids'])}
                positions=np.array([positions[rid] for rid in query_ids])
                core={k:np.array(v)[positions] for k,v in rawcore['score'].items()}
                expected=saved[n-300:n-300+len(query_ids)] if evaluation=='stream' else holdout[seed,batch] if n==849 else None
                if expected is not None:
                    for key,col in [('base','coverage'),('logistic','logistic'),('A','projection'),('B','hvdm_offset')]:
                        np.testing.assert_allclose(core[key],[float(r[col]) for r in expected],atol=1e-7,rtol=0)
                query=data[query_ids,1:]
                path=OUT/'frames'/f'{evaluation}_seed{seed}_b{batch}_n{n}.json'
                if path.exists():r=json.loads(path.read_text(encoding='utf-8'))
                else:
                    print(f'C JOINT start {evaluation} seed={seed} b={batch} n={n}',flush=True)
                    tick=perf_counter()
                    full=native_full(x[:n],y[:n],query,core['base'],x[300:n],y[300:n],history[:n-300],scale)
                    full_seconds=perf_counter()-tick
                    simple,after_d,details=predict_c(x[:n],y[:n],query,core['base'])
                    # Native-helper checks on every final holdout and the last stream batch.
                    native_error=None
                    if n in (800,849):
                        native=native_di(x[:n],y[:n],query,core['base'])
                        np.testing.assert_allclose(simple,native,atol=1e-7,rtol=0)
                        assert np.array_equal(simple>=0,native>=0)
                        native_error=float(np.max(abs(simple-native)))
                    r=dict(full=full.tolist(),simple=simple.tolist(),after_d=after_d.tolist(),
                           details=details,full_native_seconds=full_seconds,native_di_error=native_error)
                    if expected is not None:np.testing.assert_allclose(full,[float(v['new_branch']) for v in expected],atol=1e-7,rtol=0)
                    dump(path,r)
                full=np.array(r['full']);simple=np.array(r['simple'])
                fused=np.mean([core['A'],core['B'],full],axis=0)
                simple_fused=np.mean([core['A'],core['B'],simple],axis=0)
                archive_error=None
                if expected is not None:
                    np.testing.assert_allclose(full,[float(v['new_branch']) for v in expected],atol=1e-7,rtol=0)
                    np.testing.assert_allclose(fused,[float(v['new_ensemble']) for v in expected],atol=1e-7,rtol=0)
                    archive_error=float(np.max(abs(fused-np.array([float(v['new_ensemble']) for v in expected]))))
                if evaluation!='stream':
                    old=old_checkpoints[seed,batch,n]
                    m=metrics(data[query_ids,0].astype(int),fused)
                    for key in ('accuracy','recall','specificity','f1','balanced_accuracy','auc'):
                        assert abs(m[key]-float(old['roc_auc' if key=='auc' else key]))<1e-10,(seed,batch,n,key)
                checks.append(dict(evaluation=evaluation,seed=seed,batch=batch,n=n,archive_error=archive_error,native_di_error=r['native_di_error']))
                costs.append(dict(evaluation=evaluation,seed=seed,batch=batch,n=n,full_native_seconds=r['full_native_seconds'],**r['details']))
                for j,rid in enumerate(query_ids):
                    rows.append(dict(evaluation=evaluation,seed=seed,batch=batch,training_n=n,row_index=rid,
                            label=int(data[rid,0]),base=float(core['base'][j]),A=float(core['A'][j]),B=float(core['B'][j]),
                            logistic=float(core['logistic'][j]),full_c=float(full[j]),simple_c=float(simple[j]),after_d=float(r['after_d'][j]),
                            full_fusion=float(fused[j]),simple_fusion=float(simple_fused[j])))
            print(f'CONFIG complete seed={seed} batch={batch}',flush=True)
    assert len(rows)==8406 and len(checks)==75 and len(core_fits)==36
    write_csv(OUT/'predictions.csv',rows)
    dump(OUT/'checks.json',checks);dump(OUT/'costs.json',costs);dump(OUT/'core_fits.json',core_fits)
    summary=summarize(rows);dump(OUT/'summary.json',summary)
    dump(OUT/'completed.json',dict(completed=True,rows=len(rows),frames=len(checks)))
    print('COMPLETED '+json.dumps(summary['decision']),flush=True)
    for evaluation in ('stream',*[f'holdout_{n}' for n in STAGES]):
        print(evaluation,json.dumps(summary[evaluation]['comparisons']),flush=True)


if __name__=='__main__':main()
