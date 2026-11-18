"""Independent native-helper spot replay for every C ablation at one prefix."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_three_way_retrospective_holdout import panel, geometry, rescue, deletion, interactions
from src.paper_c_ablation import VARIANTS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paper_budget_c_ablation'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    cache=list((OUT/'selection').glob('seed*_n*.json'))
    assert len(cache)==36
    budget_differences={}
    for path in cache:
        record=json.loads(path.read_text(encoding='utf-8'))
        for strategy in ('coverage','projection'):
            a,b=(record['models'][f'{strategy}_{budget}'] for budget in (500,2000))
            assert a['rule_keys']==b['rule_keys']
            delta=float(np.max(abs(np.array(a['score'])-b['score'])))
            budget_differences[f'{record["seed"]}/{record["n"]}/{strategy}']=delta
    seed,batch,n=43,100,800
    split=next(r for r in json.loads((ROOT/'results/online_study/splits.json').read_text()) if r['seed']==seed)
    ids=split['train_stream'];data=np.loadtxt(ROOT/'data/dataset_1062.txt')
    x,y=data[ids,1:],data[ids,0].astype(int)
    with (ROOT/'parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv').open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['method']=='numeric_stable' and int(r['seed'])==seed and int(r['batch_size'])==batch]
    assert [int(r['row_index']) for r in rows]==ids[300:]
    scores=np.array([float(r['coverage']) for r in rows])
    base=scores[n-300:];query=x[n:]
    scale=x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.)
    past=np.column_stack(((x[300:n]-scale[0])/scale[1],scores[:n-300]))
    current=np.column_stack(((query-scale[0])/scale[1],base))
    shared=json.loads((OUT/'c_stages'/f'stream_seed{seed}_b{batch}_n{n}.json').read_text(encoding='utf-8'))
    errors={}
    for name,enabled in VARIANTS.items():
        score=base.copy()
        if 'P' in enabled:score=panel(past,y[300:n],current)[0]['rbf_margin']
        if 'E' in enabled:
            exact=geometry(past,y[300:n],current)[0]['exact_recurrence']
            score=np.where((score<0)&(exact>=0),exact,score)
        raw=np.vstack((x[:n],query))
        if 'U' in enabled:score=rescue(raw,y[:n],n,len(raw),score)[0]['unanimous']
        if 'D' in enabled:
            negative=set(map(tuple,x[:n][y[:n]==-1]))
            eligible=np.array([i for i,row in enumerate(query) if score[i]<0 and tuple(row) not in negative],dtype=int)
            if len(eligible):
                full,stable=deletion(x[:n],y[:n],query[eligible])['categorical_deletion']
                score[eligible[stable]]=full[stable]
        if 'I' in enabled:score=interactions(raw,y[:n],n,len(raw),score)['numeric_stable']
        reference=np.array(shared['scores'][name])
        np.testing.assert_allclose(score,reference,atol=1e-7,rtol=0)
        assert np.array_equal(score>=0,reference>=0)
        errors[name]=float(np.max(abs(score-reference)))
        print('Native helper verified',name,errors[name],flush=True)
    checks=json.loads((OUT/'c_replay_checks.json').read_text())
    assert len(checks)==57
    result=dict(source_hashes_unchanged=True,selection_prefixes=len(cache),
                budget_max_score_difference=max(budget_differences.values()),
                budget_differences=budget_differences,full_route_archive_replays=len(checks),
                full_route_max_error=max(r['branch_error'] for r in checks),
                native_helper_spot_check=dict(seed=seed,batch=batch,n=n,variants=errors))
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VERIFIED',json.dumps({k:v for k,v in result.items() if k not in ('budget_differences','native_helper_spot_check')}),flush=True)


if __name__=='__main__':main()
