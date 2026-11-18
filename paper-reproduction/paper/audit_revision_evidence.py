"""Recompute manuscript revision evidence from fixed data and saved scores only.

No model fitting, threshold search, resampling, or source-file mutation.
"""
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/revision_evidence'
BRANCHES = {'A': 'projection', 'B': 'hvdm_offset', 'C': 'new_branch'}


def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    pred = np.where(score >= 0, 1, -1)
    pos, neg = y == 1, y == -1
    return dict(n=len(y), positive=int(pos.sum()), negative=int(neg.sum()), correct=int((pred == y).sum()),
                accuracy=float(accuracy_score(y, pred)),
                recall=float((pred[pos] == 1).mean()) if pos.any() else None,
                specificity=float((pred[neg] == -1).mean()) if neg.any() else None,
                f1=float(f1_score(y, pred, pos_label=1)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                auc=float(roc_auc_score(y == 1, score)) if pos.any() and neg.any() else None)


def paired(y, original, candidate):
    a = np.where(original >= 0, 1, -1) == y
    b = np.where(candidate >= 0, 1, -1) == y
    return dict(repaired=int((~a & b).sum()), introduced=int((a & ~b).sum()),
                net=int(b.sum()-a.sum()))


def read_predictions(path, method=None):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if method is None or r['method'] == method]
    groups = defaultdict(list)
    for r in rows:
        groups[int(r['seed']), int(r['batch_size'])].append(r)
    return groups


def summarize(groups, data):
    out = {}
    pooled = defaultdict(lambda: [[], [], []])
    for key, rows in sorted(groups.items()):
        y = np.array([int(r['label']) for r in rows])
        assert all(data[int(r['row_index']), 0] == int(r['label']) for r in rows)
        branch = {k: np.array([float(r[v]) for r in rows]) for k,v in BRANCHES.items()}
        full = np.mean(list(branch.values()), axis=0)
        np.testing.assert_allclose(full, [float(r['new_ensemble']) for r in rows], atol=1e-12, rtol=0)
        for n in (1,2,3):
            for chosen in itertools.combinations(BRANCHES, n):
                name = ''.join(chosen)
                score = np.mean([branch[k] for k in chosen], axis=0)
                record = dict(seed=key[0], batch=key[1], **metrics(y,score),
                              full_vs_this=paired(y,score,full))
                out.setdefault(name,dict(configurations=[]))['configurations'].append(record)
                pooled[name][0].extend(y.tolist())
                pooled[name][1].extend(score.tolist())
                pooled[name][2].extend(full.tolist())
        for name,col in [('coverage','coverage'),('logistic','logistic')]:
            score=np.array([float(r[col]) for r in rows])
            record=dict(seed=key[0],batch=key[1],**metrics(y,score),full_vs_this=paired(y,score,full))
            out.setdefault(name,dict(configurations=[]))['configurations'].append(record)
            pooled[name][0].extend(y.tolist())
            pooled[name][1].extend(score.tolist())
            pooled[name][2].extend(full.tolist())
    for name, record in out.items():
        configs = record['configurations']
        record['means'] = {k:float(np.mean([r[k] for r in configs]))
                           for k in ('accuracy','recall','specificity','f1','balanced_accuracy','auc')}
        record['accuracy_minmax'] = [min(r['accuracy'] for r in configs),max(r['accuracy'] for r in configs)]
        record['pooled'] = metrics(*pooled[name][:2])
        record['full_vs_this'] = paired(*map(np.asarray,pooled[name]))
    return out


def main():
    paths = dict(data=ROOT/'data/dataset_1062.txt', splits=ROOT/'results/online_study/splits.json',
                 stream=ROOT/'parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv',
                 holdout=ROOT/'results/three_way_retrospective_holdout/predictions.csv')
    data = np.loadtxt(paths['data'])
    assert data.shape == (1062,19) and np.isfinite(data).all()
    assert set(data[:,0]) == {-1,1}
    splits = json.loads(paths['splits'].read_text())
    keys = list(map(tuple,data[:,1:]))
    groups = defaultdict(list)
    for i,k in enumerate(keys):groups[k].append(i)
    duplicate_groups = [v for v in groups.values() if len(v)>1]
    conflicts = [v for v in groups.values() if len(set(data[v,0]))>1]
    stream = read_predictions(paths['stream'],'numeric_stable')
    holdout = read_predictions(paths['holdout'])
    assert len(stream)==len(holdout)==6
    audit = dict(unique_vectors=len(groups), repeated_vector_groups=len(duplicate_groups),
                 rows_in_repeated_groups=sum(map(len,duplicate_groups)),
                 excess_rows=len(data)-len(groups), contradictory_vector_groups=len(conflicts),
                 rows_in_contradictory_groups=sum(map(len,conflicts)), splits=[])
    seen_groups = defaultdict(lambda: defaultdict(lambda: [[],[]]))
    holdout_seen_groups = defaultdict(lambda: defaultdict(lambda: [[],[]]))
    for split in splits:
        seed, ids, test = split['seed'],split['train_stream'],split['holdout']
        assert len(ids)==len(set(ids))==849 and len(test)==len(set(test))==213
        assert not set(ids)&set(test)
        known = set(keys[i] for i in ids)
        audit['splits'].append(dict(seed=seed, holdout_seen_vectors=sum(keys[i] in known for i in test),
                                    holdout_unseen_vectors=sum(keys[i] not in known for i in test)))
        for batch in (50,100):
            rows=stream[seed,batch]
            assert [int(r['row_index']) for r in rows]==ids[300:]
            for start in range(300,849,batch):
                prior=set(keys[i] for i in ids[:start])
                stop=min(849,start+batch)
                for r in rows[start-300:stop-300]:
                    name='seen' if keys[int(r['row_index'])] in prior else 'unseen'
                    for model in ('coverage','logistic','new_ensemble'):
                        seen_groups[name][model][0].append(int(r['label']))
                        seen_groups[name][model][1].append(float(r[model]))
            for r in holdout[seed,batch]:
                name='seen' if keys[int(r['row_index'])] in known else 'unseen'
                for model in ('coverage','logistic','new_ensemble'):
                    holdout_seen_groups[name][model][0].append(int(r['label']))
                    holdout_seen_groups[name][model][1].append(float(r[model]))
    audit['stream_by_prior_vector_presence']={k:{m:metrics(*v) for m,v in methods.items()} for k,methods in seen_groups.items()}
    audit['holdout_by_training_vector_presence']={k:{m:metrics(*v) for m,v in methods.items()} for k,methods in holdout_seen_groups.items()}
    audit['paired_by_vector_presence']={ev:{k:{m:paired(np.array(methods[m][0]),np.array(methods[m][1]),np.array(methods['new_ensemble'][1]))
                                for m in ('coverage','logistic')} for k,methods in groups.items()}
                                for ev,groups in [('stream',seen_groups),('holdout',holdout_seen_groups)]}
    audit['field_value_counts']=[len(set(data[:,j])) for j in range(1,19)]
    stream_summary=summarize(stream,data)
    holdout_summary=summarize(holdout,data)
    repaired_interactions=[r for rows in stream.values() for r in rows
                           if (float(r['three_way'])>=0)!=(int(r['label'])==1)
                           and (float(r['new_ensemble'])>=0)==(int(r['label'])==1)]
    audit['stable_interaction_repairs']=dict(events=len(repaired_interactions),
                        unique_record_rows=len({int(r['row_index']) for r in repaired_interactions}))
    previous=json.loads((ROOT/'paper/FP_TBF_中文论文数据核对.json').read_text(encoding='utf-8'))
    assert abs(stream_summary['ABC']['means']['accuracy']-previous['means']['new_ensemble']['accuracy'])<1e-12
    assert stream_summary['ABC']['pooled']['correct']==3175
    scales={}
    for name,col in BRANCHES.items():
        values=np.array([float(r[col]) for rows in stream.values() for r in rows])
        scales[name]=dict(median_absolute=float(np.median(abs(values))),
                          mean_absolute=float(abs(values).mean()),std=float(values.std()))
    result=dict(protocol='Post hoc descriptive recombination of fixed archived scores; no refitting or tuning.',
                sources={k:dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
                data_audit=audit,stream=stream_summary,retrospective_holdout=holdout_summary,
                raw_score_scale=scales)
    OUT.mkdir(exist_ok=True)
    (OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'fusion_combinations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['evaluation','combination','accuracy','recall','specificity','f1','balanced_accuracy','auc','full_repaired','full_introduced','full_net'])
        writer.writeheader()
        for evaluation,summary in [('stream',stream_summary),('retrospective_holdout',holdout_summary)]:
            for name,r in summary.items():
                writer.writerow(dict(evaluation=evaluation,combination=name,**r['means'],
                                **{'full_'+k:v for k,v in r['full_vs_this'].items()}))
    print(json.dumps(dict(data_audit=audit, stream={k:v['means'] for k,v in stream_summary.items()},
                          holdout={k:v['means']['accuracy'] for k,v in holdout_summary.items()},
                          raw_score_scale=scales),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
