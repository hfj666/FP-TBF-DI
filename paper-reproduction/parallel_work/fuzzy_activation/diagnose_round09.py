"""Replay both sides of the selector's paired errors."""
import json

import numpy as np

from diagnose_round05 import OLD, ROOT, load_state, read_predictions
from diagnose_round08 import OUT as PREV
from run_component import HERE, digest, load_stream
from activation import restore
from run_round08 import POLICY

OUT = HERE / 'results_09'


def replay(snapshot, x):
    model = restore(snapshot)
    rules = model.raw_rules
    w = model.firing_weights(x[None])[0]
    c = model.theta[64:]
    keys = [tuple(zip(r.columns, map(int, r.values))) for r in rules]
    pos = np.array([r.label == 1 for r in rules])
    exact = np.array([all(x[j] == v for j, v in k) for k in keys])
    return dict(score=float(w @ c), mass=[float(w[m].sum()) for m in (pos,~pos)],
                contributions=[float(w[m] @ c[m]) for m in (pos,~pos)],
                rules={k: dict(label=int(r.label), exact=bool(e), weight=float(a), head=float(b))
                       for k,r,e,a,b in zip(keys,rules,exact,w,c,strict=True)})


def main():
    verified = json.loads((PREV / 'experiment/verification.json').read_text(encoding='utf-8'))
    pair = verified['pairs']['HVDM']
    targets = {tuple(k): kind for kind in ('new_fn','corrected_fn') for k in pair[kind]}
    assert len(targets) == 17
    old_scores = read_predictions(OLD / 'predictions.csv', 'hvdm_mean')
    new_scores = read_predictions(PREV / 'experiment/predictions.csv', POLICY)
    fits = {(r['seed'],r['batch'],r['stage']):r['snapshot']
            for r in json.loads((PREV / 'experiment/fits.json').read_text())}
    audit = {(r['seed'],r['batch_size'],r['stage']):r['sha256']
             for r in json.loads((OLD / 'state_audit.json').read_text())}
    rows = []
    for sp in json.loads((ROOT / 'results/online_study/splits.json').read_text()):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(sp['holdout'])
        x,y = load_stream(ROOT / 'data/dataset_1062.txt', ids)
        positions = {int(row):i for i,row in enumerate(ids)}
        for key,kind in sorted(targets.items()):
            s,batch,stage,row = key
            if s != seed:
                continue
            i = positions[row]
            offset = stage-1
            end = 300+offset*batch
            record = load_state(seed,batch,offset,audit)
            assert end <= i < min(end+batch,849) and y[i] == 1
            assert digest(x[:end],y[:end],ids[:end]) == record['prefix_input_sha256']
            a = replay(record['models']['hvdm_mean'],x[i])
            b = replay(fits[seed,batch if offset else 0,offset],x[i])
            np.testing.assert_allclose([a['score'],b['score']],[old_scores[key][1],new_scores[key][1]],rtol=0,atol=1e-12)
            aa,bb = a.pop('rules'),b.pop('rules')
            groups = {}
            for name, table, keys in (('removed',aa,aa.keys()-bb.keys()),('added',bb,bb.keys()-aa.keys()),
                                      ('retained_old',aa,aa.keys()&bb.keys()),('retained_new',bb,aa.keys()&bb.keys())):
                groups[name] = []
                for label in (1,-1):
                    rr = [table[k] for k in sorted(keys) if table[k]['label'] == label]
                    groups[name].append(dict(exact=sum(r['exact'] for r in rr),
                        mass=sum(r['weight'] for r in rr),
                        contribution=sum(r['weight']*r['head'] for r in rr),
                        exact_contribution=sum(r['weight']*r['head'] for r in rr if r['exact'])))
            for m,g1,g2 in ((a,'removed','retained_old'),(b,'added','retained_new')):
                m['exact'] = [groups[g1][j]['exact']+groups[g2][j]['exact'] for j in range(2)]
                np.testing.assert_allclose(m['contributions'],[groups[g1][j]['contribution']+groups[g2][j]['contribution'] for j in range(2)],atol=1e-12)
            membership = sum(r['contribution'] for r in groups['added'])-sum(r['contribution'] for r in groups['removed'])
            retained = sum(r['contribution'] for r in groups['retained_new'])-sum(r['contribution'] for r in groups['retained_old'])
            np.testing.assert_allclose(membership+retained,b['score']-a['score'],atol=1e-12)
            rows.append(dict(key=key,kind=kind,old=a,new=b,groups=groups,membership_delta=membership,retained_delta=retained))
    assert len(rows) == 17
    OUT.mkdir(exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for r in rows:
        print(r['kind'],r['key'],r['old']['exact'],r['new']['exact'],
              np.round([r['old']['mass'][0],r['new']['mass'][0],r['membership_delta'],r['retained_delta']],3),
              [[g['exact'],round(g['exact_contribution'],3)] for g in (r['groups']['removed'][0],r['groups']['added'][0])],
              np.round([r['old']['score'],r['new']['score']],3))


if __name__ == '__main__':
    main()
