"""Fixed sensitivity protocol: original and five new grouped five-fold splits.

Keep full FP-TBF-DI, A+LR, raw LR, RBF-SVM and forest settings fixed.
Also replace A in A+LR by coverage Sugeno at the same 2000-step limit.
Every model uses identical per-fold records. Compare per-repeat means,
without treating repeated observations or folds as independent samples.
"""
import csv
import json
import os

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from run_paper_evidence_extension import ROOT, OUT, aggregate, csv_write, dump, fit_baselines
from src.fp_tbf_simplified import SimplifiedFPTBF
from src.fptree import FPTree
from src.models import SparseSugeno
from src.selection import CoveragePool


SEEDS = (20261108, 20261109, 20261110, 20261111, 20261112, 20261113)
ARMS = ('fp_tbf_di', 'a_lr', 'coverage_lr', 'raw_lr', 'rbf_svm', 'random_forest')


def main():
    target = ROOT / 'results/paper_grouped_stability'
    target.mkdir()
    data = np.loadtxt(ROOT / 'data/dataset_1062.txt')
    x, y = data[:, 1:], data[:, 0].astype(int)
    _, groups = np.unique(x, axis=0, return_inverse=True)
    assert len(np.unique(groups)) == 741
    previous = json.loads((OUT / 'grouped_splits.json').read_text(encoding='utf-8'))
    with (OUT / 'grouped_predictions.csv').open(encoding='utf-8-sig', newline='') as file:
        archived = {int(r['row_index']): r for r in csv.DictReader(file)}
    output, splits, summary = [], [], {}
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        predictions = []
        for fold, (train, test) in enumerate(splitter.split(x, y, groups), 1):
            train = np.random.default_rng(1108 + fold).permutation(train)
            assert not set(groups[train]) & set(groups[test])
            xt, yt, query = x[train], y[train], x[test]
            tree = FPTree().update(xt, yt)
            model = SimplifiedFPTBF().fit(xt, yt, tree)
            _, logistic, a, _, _, fusion = model.decision_scores(query).T
            pool = CoveragePool(tree.mine(.1, 3), xt, yt)
            coverage = SparseSugeno(pool.select(.5, 64), maxiter=2000).fit(xt, yt, scaling=model.scale)
            baseline, status, _, _ = fit_baselines(xt, yt, query)
            scores = dict(fp_tbf_di=fusion, a_lr=(a + logistic) / 2,
                          coverage_lr=(coverage.decision_function(query) + logistic) / 2,
                          **{name: baseline[name] for name in ARMS[3:]})
            assert all(np.isfinite(v).all() for v in scores.values())
            np.testing.assert_allclose(logistic, baseline['raw_lr'], atol=1e-7, rtol=0)
            if seed == SEEDS[0]:
                assert previous[fold-1]['train'] == train.tolist()
                assert previous[fold-1]['test'] == test.tolist()
                for name in ('fp_tbf_di', *ARMS[3:]):
                    np.testing.assert_allclose(scores[name], [float(archived[i][name]) for i in test],
                                               atol=1e-7, rtol=0)
            splits.append(dict(seed=seed, fold=fold, train=train.tolist(), test=test.tolist(),
                               baseline_status={name: status[name] for name in ARMS[3:]},
                               coverage_optimization=coverage.optimization))
            for j, index in enumerate(test):
                predictions.append(dict(seed=seed, fold=fold, row_index=int(index), label=int(y[index]),
                                        **{name: float(scores[name][j]) for name in ARMS}))
        assert len(predictions) == len({r['row_index'] for r in predictions}) == 1062
        output.extend(predictions)
        summary[seed] = dict(all=aggregate(predictions, ARMS, ('fold',)),
                             selection=aggregate([dict(r, fp_tbf_di=r['a_lr']) for r in predictions],
                                                 ('fp_tbf_di', 'coverage_lr'), ('fold',)))
        print(f'Completed grouped five-fold seed={seed}', flush=True)
    csv_write(target / 'predictions.csv', output)
    dump(target / 'splits.json', splits)
    dump(target / 'summary.json', summary)


if __name__ == '__main__':
    main()
