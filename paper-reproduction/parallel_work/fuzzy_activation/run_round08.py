"""Fit original HVDM with a single change to its rule selector."""
import hashlib
import json

import numpy as np

from diagnose_round05 import OLD, load_state
from diagnose_round08 import OUT
from run_component import HERE, ROOT, CoveragePool, FPTree, digest, load_stream, rule_records, write_csv
from activation import DistanceSugeno, header_table
from harmonic_coverage import select

POLICY = 'hvdm_harmonic_coverage'


def main():
    output = OUT / 'experiment'
    output.mkdir(exist_ok=False)
    paths = [HERE / name for name in ('PROTOCOL_08.md', 'run_round08.py', 'diagnose_round08.py',
                                      'activation.py', 'harmonic_coverage.py', 'results_08/diagnosis.json',
                                      'results_01/state_audit.json', 'results_01/input_audit.json')]
    paths += [ROOT / name for name in ('src/selection.py', 'src/models.py', 'src/fptree.py',
                                       'results/online_study/splits.json')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output / 'protocol.json').write_text(json.dumps(dict(hashes=hashes, threshold=0), indent=2))
    audit = {(r['seed'], r['batch_size'], r['stage']): r['sha256']
             for r in json.loads((OLD / 'state_audit.json').read_text())}
    inputs = {r['seed']: r['allowed_stream_sha256']
              for r in json.loads((OLD / 'input_audit.json').read_text())}
    fits, predictions = [], []
    for sp in json.loads((ROOT / 'results/online_study/splits.json').read_text()):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(sp['holdout'])
        x, y = load_stream(ROOT / 'data/dataset_1062.txt', ids)
        assert digest(x, y, ids) == inputs[seed]
        cache = {}
        for batch in (50, 100):
            for stage, end in enumerate([*range(300, 849, batch), 849]):
                state = load_state(seed, batch, stage, audit)
                assert state['prefix'] == end and state['prefix_input_sha256'] == digest(x[:end], y[:end], ids[:end])
                original = state['models']['hvdm_mean']
                if end not in cache:
                    tree = FPTree().update(x[:end], y[:end])
                    pool = CoveragePool(tree.mine(.1, 3), x[:end], y[:end])
                    assert rule_records(pool.select(.5, 64)) == original['raw_rules']
                    rules = select(pool)
                    assert len(rules) == 64 and sum(r.label == 1 for r in rules) == 32
                    assert len({tuple(zip(r.columns, r.values)) for r in rules}) == 64
                    table = header_table(tree)
                    assert [[j,v,n,p] for (j,v),(n,p) in sorted(table.items())] == original['table']
                    cache[end] = rules, table
                rules, table = cache[end]
                if stage or batch == 50:
                    model = DistanceSugeno(rules, 'hvdm_mean', table).fit(
                        x[:end], y[:end], scaling=(original['minimum'], original['span']))
                    fits.append(dict(seed=seed, batch=batch if stage else 0, stage=stage,
                                     prefix=end, snapshot=model.snapshot()))
                    if not stage:
                        initial = model
                else:
                    model = initial
                if end < 849:
                    stop = min(end + batch, 849)
                    scores = model.decision_function(x[end:stop])
                    assert np.isfinite(scores).all()
                    predictions.extend(dict(seed=seed, batch_size=batch, stage=stage+1, row_index=int(i),
                                            policy=POLICY, label=int(v), score=float(s))
                                       for i,v,s in zip(ids[end:stop],y[end:stop],scores,strict=True))
            print(f'{seed}/{batch} complete', flush=True)
    assert len(fits) == 54 and len(predictions) == 3294
    (output / 'fits.json').write_text(json.dumps(fits, separators=(',', ':')), encoding='utf-8')
    write_csv(output / 'predictions.csv', predictions)


if __name__ == '__main__':
    main()
