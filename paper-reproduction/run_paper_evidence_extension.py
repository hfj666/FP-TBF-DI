"""Fixed conventional baselines, final-fusion mechanisms, and vector-group CV."""
import csv
import hashlib
import json
import os
import platform
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,StandardScaler
from sklearn.svm import SVC

from src.fp_tbf_simplified import SimplifiedFPTBF
from src.fptree import FPTree
from src.models import SparseSugeno
from src.selection import CoveragePool
from run_three_way_retrospective_holdout import offset_fit
from paper.audit_revision_evidence import metrics,paired

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paper_evidence_extension'
CURRENT=ROOT/'results/paper_c_joint_reduction'
BASELINES=('raw_lr','onehot_lr','interaction_lr','rbf_svm','random_forest','hist_gradient_boosting')
MECHANISMS=('A_coverage','B_logistic','B_numeric','C_base')
STAGES=(300,500,700,849)


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fit_baselines(x,y,query):
    scale=x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.)
    raw,raw_test=(x-scale[0])/scale[1],(query-scale[0])/scale[1]
    encoder=OneHotEncoder(handle_unknown='ignore').fit(x)
    categorical,categorical_test=encoder.transform(x),encoder.transform(query)
    scaler=StandardScaler().fit(x)
    numeric,numeric_test=scaler.transform(x),scaler.transform(query)
    polynomial=PolynomialFeatures(2,interaction_only=True,include_bias=False).fit(numeric)
    interaction,interaction_test=[np.column_stack((b,polynomial.transform(b)[:,18:]*np.sqrt(2/17))) for b in (numeric,numeric_test)]
    arms=[('raw_lr',LogisticRegression(C=1,solver='lbfgs',max_iter=500),raw,raw_test,False),
          ('onehot_lr',LogisticRegression(C=1,solver='liblinear',max_iter=1000),categorical,categorical_test,False),
          ('interaction_lr',LogisticRegression(C=1,solver='liblinear',max_iter=1000),interaction,interaction_test,False),
          ('rbf_svm',SVC(C=1,gamma='scale',max_iter=10000),numeric,numeric_test,False),
          ('random_forest',RandomForestClassifier(n_estimators=300,max_features='sqrt',min_samples_leaf=1,random_state=1108,n_jobs=1),x,query,True),
          ('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=0,early_stopping=False,random_state=1108),x,query,True)]
    scores={};status={};models={}
    for name,model,train,test,probability in arms:
        tick=perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(train,y)
        score=model.predict_proba(test)[:,list(model.classes_).index(1)]-.5 if probability else model.decision_function(test)
        assert np.isfinite(score).all()
        scores[name]=score;models[name]=model
        iterations=getattr(model,'n_iter_',None)
        status[name]=dict(seconds=perf_counter()-tick,warnings=[str(w.message) for w in caught],
                    iterations=iterations.tolist() if hasattr(iterations,'tolist') else iterations,
                    fit_status=int(getattr(model,'fit_status_',0)))
    return scores,status,models['raw_lr'],raw


def initialize():
    names=['run_paper_evidence_extension.py','paper/创新复核与扩展论证实验协议.md',
           'src/fp_tbf_simplified.py','src/fptree.py','src/models.py','src/selection.py',
           'src/projected_selection.py','src/reference_comparators.py',
           'parallel_work/fuzzy_activation/activation.py','parallel_work/fuzzy_activation/run_round13.py',
           'data/dataset_1062.txt','results/online_study/splits.json',
           'results/paper_c_joint_reduction/predictions.csv']
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CURRENT/'core').glob('*.json')})
    info=dict(sources=sources,baselines=BASELINES,mechanisms=MECHANISMS,group_folds=5,group_seed=20261108,
              python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
              threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')})
    OUT.mkdir(exist_ok=True);(OUT/'prefixes').mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))==json.loads(json.dumps(info))
    else:dump(OUT/'manifest.json',info)


def aggregate(rows,arms,key_fields):
    result={}
    for arm in arms:
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in key_fields)].append(r)
        configs=[]
        for key,values in sorted(groups.items()):
            y=np.array([r['label'] for r in values]);z=np.array([r[arm] for r in values])
            configs.append(dict(zip(key_fields,key),**metrics(y,z)))
        y=np.array([r['label'] for r in rows]);a=np.array([r['fp_tbf_di'] for r in rows]);b=np.array([r[arm] for r in rows])
        result[arm]=dict(configurations=configs,means={k:float(np.mean([v[k] for v in configs])) for k in ('accuracy','recall','specificity','f1','balanced_accuracy','auc')},
                        pooled=metrics(y,b),candidate_vs_main=paired(y,a,b))
    return result


def main():
    initialize()
    data=np.loadtxt(ROOT/'data/dataset_1062.txt');assert data.shape==(1062,19) and np.isfinite(data).all()
    splits=json.loads((ROOT/'results/online_study/splits.json').read_text())
    with (CURRENT/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
    lookup={(r['evaluation'],int(r['seed']),int(r['batch']),int(r['row_index'])):r for r in old}
    output=[];fit_records=[];checks=[]
    for split in splits:
        seed=split['seed'];ids=split['train_stream'];test=split['holdout']
        x,y=data[ids,1:],data[ids,0].astype(int)
        scale=x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.)
        tree=FPTree();previous=0;cache={}
        for n in sorted({849,*range(300,849,50),*range(300,849,100)}):
            tree.update(x[previous:n],y[previous:n]);previous=n
            core=json.loads((CURRENT/'core'/f'seed{seed}_n{n}.json').read_text(encoding='utf-8'))
            query_ids=core['query_ids'];query=data[query_ids,1:]
            path=OUT/'prefixes'/f'seed{seed}_n{n}.json'
            if path.exists():record=json.loads(path.read_text(encoding='utf-8'))
            else:
                print(f'PREFIX start seed={seed} n={n}',flush=True)
                scores,status,log,normalized=fit_baselines(x[:n],y[:n],query)
                rules=CoveragePool(tree.mine(.1,3),x[:n],y[:n]).select(.5,64)
                tick=perf_counter();base=SparseSugeno(rules,maxiter=500).fit(x[:n],y[:n],scaling=scale)
                base_query=base.decision_function(query)
                coefficient=offset_fit(log.decision_function(normalized),base.decision_function(x[:n])/10,y[:n]==1)
                assert coefficient.success
                scores['numeric_B']=scores['raw_lr']+coefficient.x[0]*base_query/10
                status['numeric_B']=dict(seconds=perf_counter()-tick,base_optimization=base.optimization,beta=float(coefficient.x[0]))
                np.testing.assert_allclose(scores['raw_lr'],core['score']['logistic'],atol=1e-7,rtol=0)
                np.testing.assert_allclose(base_query,core['score']['base'],atol=1e-7,rtol=0)
                record=dict(query_ids=query_ids,scores={k:v.tolist() for k,v in scores.items()},status=status,
                       raw_lr_error=float(np.max(abs(scores['raw_lr']-np.array(core['score']['logistic'])))),
                       base_error=float(np.max(abs(base_query-np.array(core['score']['base'])))))
                dump(path,record)
                print(f'PREFIX done seed={seed} n={n}',flush=True)
            cache[n]=record;fit_records.append(dict(seed=seed,n=n,status=record['status']))
            checks.append(dict(seed=seed,n=n,raw_lr_error=record['raw_lr_error'],base_error=record['base_error']))
        for batch in (50,100):
            schedule=[('stream',n) for n in range(300,849,batch)]+[(f'holdout_{n}',n) for n in STAGES]
            for evaluation,n in schedule:
                query_ids=ids[n:min(n+batch,849)] if evaluation=='stream' else test
                r=cache[n];positions={rid:i for i,rid in enumerate(r['query_ids'])}
                for rid in query_ids:
                    i=positions[rid];reference=lookup[evaluation,seed,batch,rid]
                    a,b,c,base,log=[float(reference[k]) for k in ('A','B','simple_c','base','logistic')]
                    output.append(dict(evaluation=evaluation,seed=seed,batch=batch,training_n=n,row_index=rid,label=int(data[rid,0]),
                        fp_tbf_di=float(reference['simple_fusion']),**{k:r['scores'][k][i] for k in BASELINES},
                        A_coverage=(base+b+c)/3,B_logistic=(a+log+c)/3,
                        B_numeric=(a+r['scores']['numeric_B'][i]+c)/3,C_base=(a+b+base)/3))
    assert len(output)==8406 and len(checks)==36
    csv_write(OUT/'matched_predictions.csv',output);dump(OUT/'fit_status.json',fit_records);dump(OUT/'replay_checks.json',checks)
    matched={ev:aggregate([r for r in output if r['evaluation']==ev],('fp_tbf_di',*BASELINES,*MECHANISMS),('seed','batch'))
             for ev in ('stream',*[f'holdout_{n}' for n in STAGES])}
    dump(OUT/'matched_summary.json',matched)
    print('MATCHED COMPLETE',flush=True)
    # Grouping is by raw scores, not patient identity. No group crosses folds.
    x,y=data[:,1:],data[:,0].astype(int)
    _,groups=np.unique(x,axis=0,return_inverse=True)
    assert len(set(groups))==741
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261108)
    group_predictions=[];group_splits=[];group_status=[]
    for fold,(train,test) in enumerate(splitter.split(x,y,groups),1):
        train=np.random.default_rng(1108+fold).permutation(train)
        assert not set(groups[train])&set(groups[test])
        path=OUT/'folds'/f'fold{fold}.json'
        if path.exists():record=json.loads(path.read_text(encoding='utf-8'))
        else:
            print(f'GROUP fold={fold} train={len(train)} test={len(test)}',flush=True)
            scores,status,_,_=fit_baselines(x[train],y[train],x[test])
            tick=perf_counter();model=SimplifiedFPTBF().fit(x[train],y[train])
            scores['fp_tbf_di']=model.decision_function(x[test])
            status['fp_tbf_di']=dict(seconds=perf_counter()-tick,optimization=model.optimization,c_details=model.last_c_details)
            assert all(np.isfinite(v).all() for v in scores.values())
            record=dict(train=train.tolist(),test=test.tolist(),scores={k:v.tolist() for k,v in scores.items()},status=status)
            dump(path,record)
        assert record['train']==train.tolist() and record['test']==test.tolist()
        group_splits.append(dict(fold=fold,train=train.tolist(),test=test.tolist(),train_vectors=len(set(groups[train])),test_vectors=len(set(groups[test])),
                                  test_positive=int((y[test]==1).sum()),test_negative=int((y[test]==-1).sum())))
        group_status.append(dict(fold=fold,status=record['status']))
        for i,rid in enumerate(test):group_predictions.append(dict(fold=fold,row_index=int(rid),label=int(y[rid]),**{k:record['scores'][k][i] for k in ('fp_tbf_di',*BASELINES)}))
    assert len(group_predictions)==1062 and len({r['row_index'] for r in group_predictions})==1062
    grouped=aggregate(group_predictions,('fp_tbf_di',*BASELINES),('fold',))
    csv_write(OUT/'grouped_predictions.csv',group_predictions)
    dump(OUT/'grouped_splits.json',group_splits);dump(OUT/'grouped_fit_status.json',group_status);dump(OUT/'grouped_summary.json',grouped)
    dump(OUT/'completed.json',dict(completed=True,matched_events=8406,grouped_events=1062))
    print('ALL COMPLETE',flush=True)
    for ev in ('stream','holdout_849'):
        print(ev,json.dumps({k:v['means'] for k,v in matched[ev].items()}),flush=True)
    print('grouped',json.dumps({k:v['means'] for k,v in grouped.items()}),flush=True)


if __name__=='__main__':main()
