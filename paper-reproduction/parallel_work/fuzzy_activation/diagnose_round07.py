"""Replay shared-width errors with paired width/head swaps."""
import json

import numpy as np
from scipy.special import softmax

from diagnose_round05 import OLD, ROOT, load_state, read_predictions
from diagnose_round06 import OUT as PREV
from run_component import HERE, digest, load_stream
from activation import restore
from run_round06 import POLICY

OUT = HERE / 'results_07'


def main():
    verified = json.loads((PREV / 'experiment/verification.json').read_text(encoding='utf-8'))
    new = {tuple(k): 'new_fn' for k in verified['pairs']['HVDM']['new_fn']}
    low = {tuple(r['key']): 'low_mass' for r in json.loads((PREV / 'decomposition.json').read_text())
           if r['group'] == 'low_positive'}
    assert len(new) == 6 and len(low) == 16 and not new.keys() & low.keys()
    targets = new | low
    old_scores = read_predictions(OLD / 'predictions.csv', 'hvdm_mean')
    new_scores = read_predictions(PREV / 'experiment/predictions.csv', POLICY)
    fits = {(r['seed'], r['batch'], r['stage']): np.array(r['theta'])
            for r in json.loads((PREV / 'experiment/fits.json').read_text())}
    audit = {(r['seed'], r['batch_size'], r['stage']): r['sha256']
             for r in json.loads((OLD / 'state_audit.json').read_text())}
    rows = []
    for sp in json.loads((ROOT / 'results/online_study/splits.json').read_text()):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(sp['holdout'])
        x, y = load_stream(ROOT / 'data/dataset_1062.txt', ids)
        positions = {int(row): i for i, row in enumerate(ids)}
        for key, group in sorted(targets.items()):
            s, batch, stage, row = key
            if s != seed:
                continue
            i = positions[row]
            offset = stage - 1
            end = 300 + offset * batch
            state = load_state(seed, batch, offset, audit)
            assert end <= i < min(end + batch, 849) and y[i] == 1
            assert digest(x[:end], y[:end], ids[:end]) == state['prefix_input_sha256']
            model = restore(state['models']['hvdm_mean'])
            old, shared = model.theta, fits[seed, batch if offset else 0, offset]
            rules = model.raw_rules
            pos = np.array([r.label == 1 for r in rules])
            d = model._distances(x[i:i+1])[0]
            q = np.log([r.confidence for r in rules])
            exact = np.array([np.array_equal(x[i, list(r.columns)], r.values) for r in rules])
            w_old, w_new = [softmax(q - .5 * d * np.exp(-2 * lw)) for lw in (old[:64], shared[0])]
            c_old, c_new = old[64:], shared[1:]
            a, b, c, e = [float(w @ h) for w, h in ((w_old,c_old),(w_new,c_old),(w_old,c_new),(w_new,c_new))]
            np.testing.assert_allclose([a,e], [old_scores[key][1],new_scores[key][1]], rtol=0, atol=1e-12)
            assert e < 0
            rows.append(dict(key=key, group=group, exact=[int(np.sum(exact & m)) for m in (pos,~pos)],
                             mass=[float(w[pos].sum()) for w in (w_old,w_new)],
                             contributions=[[float(w[m] @ h[m]) for m in (pos,~pos)]
                                            for w,h in ((w_old,c_old),(w_new,c_new))],
                             old=a, width_only=b, head_only=c, shared=e,
                             width_effect=.5*((b-a)+(e-c)), head_effect=.5*((c-a)+(e-b))))
    assert len(rows) == 22
    OUT.mkdir(exist_ok=True)
    (OUT / 'diagnosis.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for group in ('new_fn', 'low_mass'):
        rr = [r for r in rows if r['group'] == group]
        print(group, 'distinct', len({r['key'][3] for r in rr}), 'exact means', np.mean([r['exact'] for r in rr], axis=0))
        for r in rr:
            print(r['key'], r['exact'], [round(v,3) for v in r['mass']],
                  [round(r[k],3) for k in ('old','width_only','head_only','shared','width_effect','head_effect')])
        print('means mass', np.mean([r['mass'] for r in rr], axis=0))
        print('means effects', *(np.mean([r[k] for r in rr]) for k in ('width_effect','head_effect')))


if __name__ == '__main__':
    main()
