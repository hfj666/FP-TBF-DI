"""Audit eligible exact matches and replay Cauchy disagreements."""
import itertools
import json

import numpy as np
from scipy.special import softmax

from diagnose_round05 import OLD, ROOT, load_state, read_predictions
from diagnose_round07 import OUT as PREV
from run_component import HERE, CoveragePool, FPTree, digest, load_stream, rule_records
from activation import restore
from run_round07 import POLICY

OUT = HERE / 'results_08'


def main():
    verification = json.loads((PREV / 'experiment/verification.json').read_text(encoding='utf-8'))
    added = {tuple(k) for k in verification['pairs']['HVDM']['new_fn']}
    low = {tuple(r['key']) for r in json.loads((PREV / 'diagnosis.json').read_text()) if r['group'] == 'low_mass'}
    assert len(added) == 6 and len(low) == 16 and not added & low
    hvdm = read_predictions(OLD / 'predictions.csv', 'hvdm_mean')
    cauchy = read_predictions(PREV / 'experiment/predictions.csv', POLICY)
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
        pools = {}
        for key in sorted(added | low):
            s, batch, stage, row = key
            if s != seed:
                continue
            i = positions[row]
            offset = stage - 1
            end = 300 + offset * batch
            state = load_state(seed, batch, offset, audit)
            assert end <= i < min(end + batch, 849) and y[i] == 1
            assert digest(x[:end], y[:end], ids[:end]) == state['prefix_input_sha256']
            snap = state['models']['hvdm_mean']
            model = restore(snap)
            if end not in pools:
                patterns = FPTree().update(x[:end], y[:end]).mine(.1, 3)
                pools[end] = CoveragePool(patterns, x[:end], y[:end])
            pool = pools[end]
            assert rule_records(pool.select(.5,64)) == snap['raw_rules']

            def matches(rule):
                return np.array_equal(x[i,list(rule.columns)], rule.values)

            pos = np.array([r.label == 1 for r in model.raw_rules])
            selected = [sum(r.label == label and matches(r) for r in model.raw_rules) for label in (1,-1)]
            eligible = {label: [items for items,r,_,_ in pool.pools[label] if matches(r)] for label in (1,-1)}
            mined = [(items,n,p) for items,(n,p) in pool.patterns.items() if all(x[i,j] == v for j,v in items)]
            row_data = dict(key=key, group='low' if key in low else 'added', prefix=end,
                            mined_exact=len(mined), mined_positive_majority=sum(2*p > n for _,n,p in mined),
                            eligible=[len(eligible[1]),len(eligible[-1])], selected=selected,
                            unselected_positive=[items for items in eligible[1] if items not in {
                                tuple(zip(r.columns,r.values)) for r in model.raw_rules if r.label==1}])
            assert len(row_data['unselected_positive']) == len(eligible[1])-selected[0]
            d = model._distances(x[i:i+1])[0]
            q = np.log([r.confidence for r in model.raw_rules])
            old, new = model.theta, fits[seed,batch if offset else 0,offset]
            swaps, mass = {}, {}
            for curve, width, head in itertools.product((0,1), repeat=3):
                a = .5 * d * np.exp(-2 * (old,new)[width][:64])
                w = softmax(q - (np.log1p(a) if curve else a))
                swaps[curve,width,head] = float(w @ (old,new)[head][64:])
                mass[curve,width,head] = float(w[pos].sum())
            np.testing.assert_allclose([swaps[0,0,0],swaps[1,1,1]], [hvdm[key][1],cauchy[key][1]], rtol=0, atol=1e-12)
            effects = np.zeros(3)
            for order in itertools.permutations(range(3)):
                bits = [0,0,0]
                for j in order:
                    before = swaps[tuple(bits)]
                    bits[j] = 1
                    effects[j] += (swaps[tuple(bits)] - before)/6
            row_data.update(scores={''.join(map(str,k)):v for k,v in swaps.items()},
                            old_mass=mass[0,0,0], cauchy_mass=mass[1,1,1], effects=effects.tolist())
            rows.append(row_data)
    assert len(rows) == 22
    OUT.mkdir(exist_ok=True)
    (OUT / 'diagnosis.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for r in rows:
        print(r['group'],r['key'],r['mined_exact'],r['mined_positive_majority'],r['eligible'],r['selected'],
              [round(r['scores'][k],3) for k in ('000','100','010','001','111')],np.round(r['effects'],3))


if __name__ == '__main__':
    main()
