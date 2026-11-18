"""Compare frozen and cumulative FP-TBF-DI on the same 3294 stream events.

Fit the frozen model once per split on the initial 300 rows. Do not tune it
or reveal any query labels. Evaluate with the original zero threshold and
average across the six configurations. This compares update policies.
"""

import csv
import json

import numpy as np

from run_paper_evidence_extension import OUT, ROOT, aggregate, csv_write, dump
from src.fp_tbf_simplified import SimplifiedFPTBF


def main():
    target = OUT / 'frozen_update'
    data = np.loadtxt(ROOT / 'data/dataset_1062.txt')
    splits = json.loads((ROOT / 'results/online_study/splits.json').read_text())
    with (OUT / 'matched_predictions.csv').open(newline='', encoding='utf-8-sig') as file:
        rows = [
            dict(seed=int(r['seed']), batch=int(r['batch']), training_n=int(r['training_n']),
                 row_index=int(r['row_index']), label=int(r['label']), fp_tbf_di=float(r['fp_tbf_di']))
            for r in csv.DictReader(file) if r['evaluation'] == 'stream'
        ]
    for split in splits:
        ids = split['train_stream']
        train, query = data[ids[:300]], data[ids[300:], 1:]
        scores = SimplifiedFPTBF().fit(train[:, 1:], train[:, 0].astype(int)).decision_function(query)
        assert np.isfinite(scores).all()
        frozen = dict(zip(ids[300:], scores.tolist()))
        for row in rows:
            if row['seed'] == split['seed']:
                row['frozen'] = frozen[row['row_index']]
                assert row['label'] == data[row['row_index'], 0]
                if row['training_n'] == 300:
                    np.testing.assert_allclose(row['frozen'], row['fp_tbf_di'], rtol=0, atol=1e-7)
        print(f"Verified frozen predictions for seed {split['seed']}", flush=True)
    assert len(rows) == len({(r['seed'], r['batch'], r['row_index']) for r in rows}) == 3294
    if any(target.glob('*')):
        raise FileExistsError(target)
    target.mkdir(exist_ok=True)
    csv_write(target / 'predictions.csv', rows)
    dump(target / 'summary.json', aggregate(rows, ('fp_tbf_di', 'frozen'), ('seed', 'batch')))


if __name__ == '__main__':
    main()
