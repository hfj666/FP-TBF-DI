"""Compare raw fusion with zero-preserving scaling from previous batches."""

import csv
import hashlib
from collections import defaultdict

import numpy as np

from run_paper_evidence_extension import ROOT, aggregate, csv_write, dump, metrics


def paired_ba_interval(rows):
    labels = np.array([row['label'] for row in rows])
    raw = np.array([row['raw'] >= 0 for row in rows])
    scaled = np.array([row['scaled'] >= 0 for row in rows])
    correct_change = (scaled == (labels == 1)).astype(int) - (raw == (labels == 1))
    _, records = np.unique([row['row_index'] for row in rows], return_inverse=True)
    _, configs = np.unique([(row['seed'], row['batch']) for row in rows],
                           axis=0, return_inverse=True)
    strata = [np.unique(records[labels == label]) for label in (-1, 1)]
    assert not np.intersect1d(*strata).size
    rng = np.random.default_rng(20261110)
    differences = []
    for _ in range(10000):
        sampled = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        weights = np.bincount(sampled, minlength=records.max() + 1)[records]
        class_deltas = []
        for label in (-1, 1):
            mask = labels == label
            totals = np.bincount(configs[mask], weights=weights[mask], minlength=6)
            changes = np.bincount(configs[mask], weights=weights[mask]*correct_change[mask], minlength=6)
            class_deltas.append(np.mean(changes/totals))
        differences.append(float(np.mean(class_deltas)))
    return np.quantile(differences, [.025, .975]).tolist()


def main():
    source = ROOT / 'results/paper_c_joint_reduction/predictions.csv'
    target = ROOT / 'results/paper_fusion_scale'
    groups = defaultdict(lambda: defaultdict(list))
    with source.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            if row['evaluation'] == 'stream':
                groups[int(row['seed']), int(row['batch'])][int(row['training_n'])].append(row)

    rows = []
    scales = []
    batch_auc = {'raw': [], 'scaled': []}
    for (seed, batch), checkpoints in sorted(groups.items()):
        history = []
        for training_n, block in sorted(checkpoints.items()):
            scores = np.array([[float(row[key]) for key in ('A', 'B', 'simple_c')] for row in block])
            median = np.median(np.abs(np.concatenate(history)), axis=0) if history else np.ones(3)
            scale = np.where(median > 0, median, 1.)
            raw = scores.mean(axis=1)
            np.testing.assert_allclose(raw, [float(row['simple_fusion']) for row in block],
                                       atol=1e-12, rtol=0)
            scaled = (scores/scale).mean(axis=1)
            labels = np.array([int(row['label']) for row in block])
            if np.unique(labels).size == 2:
                for name, scores_for_arm in (('raw', raw), ('scaled', scaled)):
                    batch_auc[name].append(metrics(labels, scores_for_arm)['auc'])
            scales.append(dict(seed=seed, batch=batch, training_n=training_n,
                               history_n=sum(len(previous) for previous in history),
                               scale=scale.tolist()))
            for row, raw_score, scaled_score in zip(block, raw, scaled, strict=True):
                rows.append(dict(seed=seed, batch=batch, training_n=training_n,
                                 row_index=int(row['row_index']), label=int(row['label']),
                                 fp_tbf_di=float(raw_score), raw=float(raw_score),
                                 scaled=float(scaled_score)))
            history.append(scores)

    assert len(groups) == 6 and len(rows) == 3294
    assert len({row['row_index'] for row in rows}) == 944
    results = aggregate(rows, ('raw', 'scaled'), ('seed', 'batch'))
    results['within_batch_auc'] = {arm: float(np.mean(values)) for arm, values in batch_auc.items()}
    results['within_batch_auc']['batch_count'] = len(batch_auc['raw'])
    after_first = [row for row in rows if row['training_n'] > min(groups[row['seed'], row['batch']])]
    post_initial = aggregate(after_first, ('raw', 'scaled'), ('seed', 'batch'))
    results['after_first_batch'] = dict(
        events=len(after_first),
        mean_auc={arm: post_initial[arm]['means']['auc'] for arm in ('raw', 'scaled')},
    )
    results['paired_ba_95_percent_interval'] = paired_ba_interval(rows)
    results['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    results['scales'] = scales
    target.mkdir(exist_ok=True)
    csv_write(target / 'predictions.csv', rows)
    dump(target / 'summary.json', results)


if __name__ == '__main__':
    main()
