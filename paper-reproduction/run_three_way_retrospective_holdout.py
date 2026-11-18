"""Evaluate the frozen three-way method on historically reused holdout rows."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from run_original_closeout import load_stream
from src.fptree import FPTree
from src.internal_research import fit_pair
from src.models import SparseSugeno
from src.reference_comparators import CumulativeLogistic
from src.selection import CoveragePool

ROOT = Path(__file__).resolve().parent
ROBUST = ROOT / 'parallel_work/robust_learning'
sys.path[:0] = [str(ROOT / 'parallel_work/fuzzy_activation'), str(ROBUST)]
from activation import DistanceSugeno, header_table
from rescue_geometry_learning import predict as geometry
from rescue_panel_learning import predict as panel
from run_rescue_feature_deletion import predict as deletion
from run_rescue_interactions import predict as interactions
from run_rescue_prefix import rescue
from run_round13 import offset_fit


def score(x, y, query, previous_x, previous_y, previous_scores, *, tree=None):
    n = len(x)
    scale = x[:300].min(0), np.maximum(np.ptp(x[:300], axis=0), 1.)
    if tree is None:
        tree = FPTree().update(x, y)
    elif tree.n != n:
        raise ValueError('FP tree size must match the revealed training prefix.')
    patterns = tree.mine(.1, 3)
    models, _ = fit_pair(patterns, x, y, scale)
    a = models['soft_projected'].decision_function(query)
    rules = CoveragePool(patterns, x, y).select(.5, 64)
    coverage = SparseSugeno(rules, maxiter=500).fit(x, y, scaling=scale)
    base = coverage.decision_function(query)
    hvdm = DistanceSugeno(rules, 'hvdm_mean', header_table(tree)).fit(x, y, scaling=scale)
    log = CumulativeLogistic(x, y, scale)
    calibration = offset_fit(log.decision_function(x), hvdm.decision_function(x)/10, y == 1)
    assert calibration.success
    logistic = log.decision_function(query)
    b = logistic + hvdm.decision_function(query)/10 * calibration.x[0]
    past = np.column_stack(((previous_x-scale[0])/scale[1], previous_scores))
    current = np.column_stack(((query-scale[0])/scale[1], base))
    panel_scores, _ = panel(past, previous_y, current)
    geo_scores, _ = geometry(past, previous_y, current)
    rbf = panel_scores['rbf_margin']
    exact = geo_scores['exact_recurrence']
    reference = np.where((rbf < 0) & (exact >= 0), exact, rbf)
    raw = np.vstack((x, query))
    reference = rescue(raw, y, n, len(raw), reference)[0]['unanimous']
    negatives = set(map(tuple, x[y == -1]))
    eligible = np.array([i for i, row in enumerate(query)
                         if reference[i] < 0 and tuple(row) not in negatives], dtype=int)
    if len(eligible):
        full, stable = deletion(x, y, query[eligible])['categorical_deletion']
        reference[eligible[stable]] = full[stable]
    c = interactions(raw, y, n, len(raw), reference)['numeric_stable']
    return np.column_stack((base, logistic, a, b, c, np.mean([a, b, c], axis=0)))


def read_holdout(path, ids):
    wanted = set(ids)
    with path.open() as f:
        found = {i: np.fromstring(line, sep=' ') for i, line in enumerate(f) if i in wanted}
    rows = np.array([found[i] for i in ids])
    assert rows.shape == (213, 19)
    return rows[:, 1:], rows[:, 0].astype(int)


def main():
    source = ROBUST / 'experiment_rescue_interactions/predictions.csv'
    with source.open(newline='') as f:
        archive = [r for r in csv.DictReader(f) if r['method'] == 'numeric_stable']
    splits = json.loads((ROOT / 'results/online_study/splits.json').read_text())
    path = ROOT / 'data/dataset_1062.txt'
    cols = ('coverage', 'logistic', 'projection', 'hvdm_offset', 'new_branch', 'new_ensemble')
    configs = []
    for split in splits:
        seed, ids, excluded = (split[k] for k in ('seed', 'train_stream', 'holdout'))
        assert len(set(ids)) == 849 and len(set(excluded)) == 213 and not set(ids) & set(excluded)
        x, y = load_stream(path, ids)
        for batch in (50, 100):
            records = [r for r in archive if (int(r['seed']), int(r['batch_size'])) == (seed, batch)]
            assert [int(r['row_index']) for r in records] == ids[300:]
            saved = np.array([[float(r[k]) for k in cols] for r in records])
            k = 800
            reproduced = score(x[:k], y[:k], x[k:], x[300:k], y[300:k], saved[:k-300, 0])
            errors = np.max(np.abs(reproduced-saved[k-300:]), axis=0)
            print('replay', seed, batch, errors, flush=True)
            np.testing.assert_allclose(reproduced, saved[k-300:], rtol=0, atol=1e-7)
            configs.append((seed, batch, excluded, x, y, saved[:, 0]))

    out = ROOT / 'results/three_way_retrospective_holdout'
    out.mkdir(exist_ok=False)
    with (out / 'predictions.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('seed', 'batch_size', 'row_index', 'label', *cols))
        for seed, batch, excluded, x, y, past in configs:
            query, labels = read_holdout(path, excluded)
            scores = score(x, y, query, x[300:], y[300:], past)
            counts = (np.where(scores >= 0, 1, -1) == labels[:, None]).sum(axis=0)
            print('holdout', seed, batch, 'correct of', len(labels), counts, flush=True)
            writer.writerows((seed, batch, i, label, *s)
                             for i, label, s in zip(excluded, labels, scores, strict=True))


if __name__ == '__main__':
    main()
