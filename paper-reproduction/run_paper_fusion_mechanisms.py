"""Matched comparison of rule selection and fusion choices on saved scores."""

import hashlib
from collections import defaultdict

import numpy as np

from run_paper_evidence_extension import ROOT, aggregate, csv_write, dump
from run_paper_fusion_scale import paired_ba_interval
from run_paper_parsimony_cost import load_csv


def event_key(row):
    return tuple(int(row[k]) for k in ('seed', 'batch', 'training_n', 'row_index'))


def stream_rows(path):
    rows = [row for row in load_csv(path) if row['evaluation'] == 'stream']
    indexed = {event_key(row): row for row in rows}
    assert len(indexed) == len(rows) == 3294
    return indexed


def main():
    joint_path = ROOT / 'results/paper_c_joint_reduction/predictions.csv'
    selection_path = ROOT / 'results/paper_budget_c_ablation/selection_predictions.csv'
    joint = stream_rows(joint_path)
    selection = stream_rows(selection_path)
    assert joint.keys() == selection.keys()
    groups = defaultdict(list)
    for key in sorted(joint):
        assert joint[key]['label'] == selection[key]['label']
        groups[key[:3]].append(key)

    previous = defaultdict(list)
    rows = []
    for (seed, batch, prefix), keys in groups.items():
        scores = np.array([[float(joint[k][name]) for name in ('A', 'B', 'simple_c')]
                           for k in keys])
        projection = [float(selection[k]['projection_2000']) for k in keys]
        np.testing.assert_allclose(scores[:, 0], projection, atol=1e-7, rtol=0)
        coverage = scores.copy()
        coverage[:, 0] = [float(selection[k]['coverage_2000']) for k in keys]
        history = previous[seed, batch]
        medians = np.median(np.abs(np.concatenate(history)), axis=0) if history else np.ones(3)
        scales = np.where(medians > 0, medians, 1.)
        raw = scores.mean(axis=1)
        np.testing.assert_allclose(raw, [float(joint[k]['simple_fusion']) for k in keys],
                                   atol=1e-12, rtol=0)
        adjusted = scores/scales
        arms = {
            'raw_projection': raw,
            'raw_coverage': coverage.mean(axis=1),
            'scaled_projection': adjusted.mean(axis=1),
            'scaled_coverage': (coverage/scales).mean(axis=1),
            'scaled_BC': adjusted[:, [1, 2]].mean(axis=1),
            'scaled_AC': adjusted[:, [0, 2]].mean(axis=1),
            'scaled_AB': adjusted[:, [0, 1]].mean(axis=1),
        }
        for column, branch in enumerate('ABC'):
            weights = np.ones(3)
            weights[column] = 2
            arms[f'scaled_double_{branch}'] = adjusted @ weights/4
        for i, k in enumerate(keys):
            rows.append(dict(seed=seed, batch=batch, training_n=prefix, row_index=k[-1],
                             label=int(joint[k]['label']), fp_tbf_di=float(raw[i]),
                             **{arm: float(values[i]) for arm, values in arms.items()}))
        history.append(scores)

    assert len(rows) == 3294 and len({r['row_index'] for r in rows}) == 944
    summary = aggregate(rows, tuple(arms), ('seed', 'batch'))
    contrasts = [
        ('raw_projection', 'raw_coverage'),
        ('scaled_projection', 'scaled_coverage'),
        *[('scaled_projection', f'scaled_{name}')
          for name in ('BC', 'AC', 'AB', 'double_A', 'double_B', 'double_C')],
    ]
    comparisons = []
    for candidate, reference in contrasts:
        paired = [dict(row, raw=row[reference], scaled=row[candidate]) for row in rows]
        repaired = sum((r['scaled'] >= 0) == (r['label'] == 1)
                       and (r['raw'] >= 0) != (r['label'] == 1) for r in paired)
        introduced = sum((r['scaled'] >= 0) != (r['label'] == 1)
                         and (r['raw'] >= 0) == (r['label'] == 1) for r in paired)
        cm, rm = summary[candidate]['means'], summary[reference]['means']
        comparisons.append(dict(candidate=candidate, reference=reference,
                                ba_difference=cm['balanced_accuracy']-rm['balanced_accuracy'],
                                auc_difference=cm['auc']-rm['auc'],
                                ba_interval=paired_ba_interval(paired),
                                repaired=repaired, introduced=introduced))
    after_first = [row for row in rows if row['training_n'] > 300]
    post_initial = aggregate(after_first, ('scaled_projection', 'scaled_coverage'), ('seed', 'batch'))
    post_pairs = [dict(row, raw=row['scaled_coverage'], scaled=row['scaled_projection'])
                  for row in after_first]
    post_result = dict(
        events=len(after_first),
        ba_difference=post_initial['scaled_projection']['means']['balanced_accuracy']
        - post_initial['scaled_coverage']['means']['balanced_accuracy'],
        ba_interval=paired_ba_interval(post_pairs),
    )
    target = ROOT / 'results/paper_fusion_mechanisms'
    target.mkdir(exist_ok=True)
    csv_write(target / 'predictions.csv', rows)
    dump(target / 'summary.json', dict(
        source_sha256={path.name if path == selection_path else 'joint_predictions.csv':
                       hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (joint_path, selection_path)},
        arms=summary, comparisons=comparisons, after_first_batch=post_result,
    ))


if __name__ == '__main__':
    main()
