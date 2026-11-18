"""Isolate A selection at equal Sugeno iteration limits inside the fusion."""
import csv

import numpy as np

from run_paper_evidence_extension import ROOT, aggregate, csv_write, dump


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        yield from csv.DictReader(file)


def key(row):
    evaluation = 'holdout_849' if row['evaluation'] == 'holdout' else row['evaluation']
    return evaluation, *(int(row[name]) for name in ('seed', 'batch', 'row_index'))


def main():
    results = ROOT / 'results'
    budgets = {key(r): r for r in rows(results / 'paper_budget_c_ablation/selection_predictions.csv')}
    output = []
    for row in rows(results / 'paper_c_joint_reduction/predictions.csv'):
        evaluation, seed, batch, index = key(row)
        if evaluation not in ('stream', 'holdout_849'):
            continue
        other = budgets.pop((evaluation, seed, batch, index))
        assert other['label'] == row['label']
        b, c = (float(row[name]) for name in ('B', 'simple_c'))
        scores = {name: (float(other[name]) + b + c) / 3 for name in
                  ('projection_500', 'projection_2000', 'coverage_500', 'coverage_2000')}
        np.testing.assert_allclose(scores['projection_2000'], float(row['simple_fusion']), atol=1e-7, rtol=0)
        output.append(dict(evaluation=evaluation, seed=seed, batch=batch, row_index=index,
                           label=int(row['label']), **scores))
    assert not budgets and len(output) == 4572
    summary = {}
    for evaluation in dict.fromkeys(r['evaluation'] for r in output):
        summary[evaluation] = {}
        for limit in (500, 2000):
            paired = [dict(r, fp_tbf_di=r[f'projection_{limit}']) for r in output if r['evaluation'] == evaluation]
            summary[evaluation][limit] = aggregate(paired, ('fp_tbf_di', f'coverage_{limit}'), ('seed', 'batch'))
    folder = results / 'paper_parsimony_cost'
    csv_write(folder / 'matched_fusion_budget.csv', output)
    dump(folder / 'matched_fusion_budget.json', summary)


if __name__ == '__main__':
    main()
