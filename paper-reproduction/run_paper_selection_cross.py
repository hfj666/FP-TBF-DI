"""Cross rule selector, fusion structure, and existing evaluation protocols.

Use a 2000-iteration Sugeno limit in all four cells. Freeze B, C and LR
scores within each event. Reuse the six grouped partitions without tuning.
"""
import csv
import json
import os
from collections import defaultdict

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from run_paper_evidence_extension import ROOT, aggregate, csv_write, dump
from src.fp_tbf_simplified import SimplifiedFPTBF
from src.fptree import FPTree
from src.models import SparseSugeno
from src.selection import CoveragePool


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def key(row):
    return (row['evaluation'].replace('holdout', 'holdout_849')
            if row['evaluation'] == 'holdout' else row['evaluation'],
            *(int(row[k]) for k in ('seed', 'batch', 'row_index')))


def scores(projection, coverage, b, c, logistic):
    return dict(joint_three=(projection+b+c)/3, coverage_three=(coverage+b+c)/3,
                joint_two=(projection+logistic)/2, coverage_two=(coverage+logistic)/2)


def summarize(rows, fields):
    return {
        structure: aggregate([dict(row, fp_tbf_di=row[f'joint_{structure}']) for row in rows],
                             ('fp_tbf_di', f'coverage_{structure}'), fields)
        for structure in ('three', 'two')
    }


def main():
    results = ROOT / 'results'
    target = results / 'paper_selection_cross'
    target.mkdir()
    old = load(results / 'paper_c_joint_reduction/predictions.csv')
    limited = load(results / 'paper_budget_c_ablation/selection_predictions.csv')
    indexed = {key(row): row for row in old}
    output = defaultdict(list)
    for row in limited:
        evaluation, seed, batch, index = key(row)
        original = indexed[evaluation, seed, batch, index]
        assert row['label'] == original['label'] and row['training_n'] == original['training_n']
        a, cov = (float(row[k]) for k in ('projection_2000', 'coverage_2000'))
        b, c, lr = (float(original[k]) for k in ('B', 'simple_c', 'logistic'))
        values = scores(a, cov, b, c, lr)
        np.testing.assert_allclose(values['joint_three'], float(original['simple_fusion']), atol=1e-7, rtol=0)
        output[evaluation].append(dict(evaluation=evaluation, seed=seed, batch=batch,
                                       row_index=index, label=int(row['label']), **values))
    assert [len(output[k]) for k in ('stream', 'holdout_849')] == [3294, 1278]
    csv_write(target / 'matched.csv', output['stream'] + output['holdout_849'])
    summary = {k: summarize(v, ('seed','batch')) for k, v in output.items()}

    prior = results / 'paper_grouped_stability'
    splits = json.loads((prior / 'splits.json').read_text(encoding='utf-8'))
    archive = {(int(r['seed']), int(r['row_index'])): r for r in load(prior / 'predictions.csv')}
    data = np.loadtxt(ROOT / 'data/dataset_1062.txt')
    x, y = data[:, 1:], data[:, 0].astype(int)
    grouped = defaultdict(list)
    for split in splits:
        seed, fold, train, test = (split[k] for k in ('seed','fold','train','test'))
        xt, yt, query = x[train], y[train], x[test]
        assert not set(map(tuple, xt)) & set(map(tuple, query))
        tree = FPTree().update(xt, yt)
        model = SimplifiedFPTBF().fit(xt, yt, tree)
        _, lr, a, b, c, _ = model.decision_scores(query).T
        pool = CoveragePool(tree.mine(.1,3), xt, yt)
        cov = SparseSugeno(pool.select(.5,64), maxiter=2000).fit(xt, yt, scaling=model.scale)
        values = scores(a, cov.decision_function(query), b, c, lr)
        for arm, field in [('joint_three','fp_tbf_di'), ('joint_two','a_lr'), ('coverage_two','coverage_lr')]:
            ref = [float(archive[seed, idx][field]) for idx in test]
            np.testing.assert_allclose(values[arm], ref, atol=1e-7, rtol=0)
            np.testing.assert_array_equal(values[arm]>=0, np.asarray(ref)>=0)
        for j, index in enumerate(test):
            assert archive[seed, index]['label'] == str(y[index])
            grouped[seed].append(dict(seed=seed, fold=fold, row_index=index, label=int(y[index]),
                                      **{k: float(v[j]) for k,v in values.items()}))
        if fold == 5:
            print(f'Completed grouped seed {seed}', flush=True)
    assert len(grouped) == 6 and all(len(v)==len({r['row_index'] for r in v})==1062 for v in grouped.values())
    csv_write(target / 'grouped.csv', [r for rows in grouped.values() for r in rows])
    summary['grouped'] = {seed: summarize(rows, ('fold',)) for seed, rows in grouped.items()}
    dump(target / 'summary.json', summary)


if __name__ == '__main__':
    main()
