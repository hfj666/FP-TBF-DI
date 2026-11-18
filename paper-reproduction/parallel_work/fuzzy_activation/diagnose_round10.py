"""Trace rule membership, activation and consequents on deduplicated errors."""
import json

import numpy as np

from diagnose_round05 import OLD, ROOT, load_state, read_predictions
from diagnose_round08 import OUT as HARMONIC
from diagnose_round09 import OUT as RESERVED, replay
from run_component import HERE, digest, load_stream
from run_round08 import POLICY as HARMONIC_POLICY
from run_round09 import POLICY as RESERVED_POLICY

OUT = HERE / 'results_10'


def compare(a, b):
    old, new = a['rules'], b['rules']
    result = []
    for label in (1, -1):
        removed = [old[k] for k in old.keys()-new.keys() if old[k]['label'] == label]
        added = [new[k] for k in new.keys()-old.keys() if new[k]['label'] == label]
        shared = [(old[k],new[k]) for k in old.keys() & new.keys() if old[k]['label'] == label]
        assert all(x['label'] == y['label'] and x['exact'] == y['exact'] for x,y in shared)
        contribution = lambda rr: sum(r['weight']*r['head'] for r in rr)
        useful = lambda r: r['exact'] and r['head'] > 0
        result.append(dict(exact_old=sum(r['exact'] for r in old.values() if r['label'] == label),
                           exact_new=sum(r['exact'] for r in new.values() if r['label'] == label),
                           useful_removed=sum(useful(r) for r in removed),
                           useful_retained=sum(useful(x) for x,_ in shared),
                           useful_retained_head_nonpositive=sum(useful(x) and y['head'] <= 0 for x,y in shared),
                           useful_removed_contribution=contribution([r for r in removed if useful(r)]),
                           useful_retained_old=contribution([x for x,_ in shared if useful(x)]),
                           useful_retained_new=contribution([y for x,y in shared if useful(x)]),
                           removed=-contribution(removed), added=contribution(added),
                           weight=sum((y['weight']-x['weight'])*(y['head']+x['head'])/2 for x,y in shared),
                           head=sum((y['head']-x['head'])*(y['weight']+x['weight'])/2 for x,y in shared)))
    for j,g in enumerate(result):
        np.testing.assert_allclose(sum(g[k] for k in ('removed','added','weight','head')),
                                   b['contributions'][j]-a['contributions'][j],atol=1e-12)
    return result


def main():
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    h = read_predictions(OLD/'predictions.csv','hvdm_mean')
    p = read_predictions(HARMONIC/'experiment/predictions.csv',HARMONIC_POLICY)
    r = read_predictions(RESERVED/'experiment/predictions.csv',RESERVED_POLICY)
    assert h.keys()==p.keys()==r.keys()
    groups = {
        'persistent': {k for k,(y,s) in r.items() if y==1 and s<0 and p[k][1]<0<=h[k][1]},
        'lost_fix': {k for k,(y,s) in r.items() if y==1 and s<0 and h[k][1]<0<=p[k][1]},
        'new_fn': {k for k,(y,s) in r.items() if y==1 and s<0<=h[k][1]}}
    assert [len(g) for g in groups.values()]==[4,5,6]
    targets = set.union(*groups.values())
    audit = {(a['seed'],a['batch_size'],a['stage']):a['sha256'] for a in read(OLD/'state_audit.json')}
    fits = [{(v['seed'],v['batch'],v['stage']):v['snapshot'] for v in read(base/'experiment/fits.json')}
            for base in (HARMONIC,RESERVED)]
    rows = []
    for sp in read(ROOT/'results/online_study/splits.json'):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids)==len(set(ids))==849 and not set(ids)&set(sp['holdout'])
        x,y = load_stream(ROOT/'data/dataset_1062.txt',ids)
        positions = {int(row):i for i,row in enumerate(ids)}
        for key in sorted(targets):
            s,batch,stage,row = key
            if s!=seed:
                continue
            offset = stage-1
            end = 300+offset*batch
            i = positions[row]
            state = load_state(seed,batch,offset,audit)
            assert end<=i<min(end+batch,849) and y[i]==1
            assert digest(x[:end],y[:end],ids[:end])==state['prefix_input_sha256']
            fs = seed,batch if offset else 0,offset
            snapshots = [state['models']['hvdm_mean'],*(f[fs] for f in fits)]
            models = [replay(snap,x[i]) for snap in snapshots]
            np.testing.assert_allclose([m['score'] for m in models],[scores[key][1] for scores in (h,p,r)],rtol=0,atol=1e-12)
            names = [name for name,g in groups.items() if key in g]
            refs = [j for j in (0,1) if models[j]['score']>=0]
            assert refs
            j = 1 if 'lost_fix' in names else 0
            assert j in refs
            comparison = compare(models[j],models[2])
            reserved = [tuple(zip(rule['columns'],map(int,rule['values']))) for rule in snapshots[2]['raw_rules'][:16]+snapshots[2]['raw_rules'][32:48]]
            rr = models[2]['rules']
            fixed = [rr[k] for k in reserved if rr[k]['label']==1 and rr[k]['exact']]
            rows.append(dict(key=key,groups=names,reference=('HVDM','harmonic')[j],
                             models=[{k:v for k,v in m.items() if k!='rules'} for m in models],
                             changes=comparison,anchored_positive_exact=len(fixed),
                             anchored_positive_contribution=sum(a['weight']*a['head'] for a in fixed)))
    assert len(rows)==len(targets)==11
    OUT.mkdir(exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for r in rows:
        print(r['key'],r['groups'],r['reference'],[round(m['score'],3) for m in r['models']],
              [round(m['mass'][0],3) for m in r['models']],r['anchored_positive_exact'])
        for g in r['changes']:
            print(g['exact_old'],g['exact_new'],g['useful_removed'],g['useful_retained'],g['useful_retained_head_nonpositive'],
                  [round(g[k],3) for k in ('useful_removed_contribution','useful_retained_old','useful_retained_new','removed','added','weight','head')])


if __name__=='__main__':
    main()

