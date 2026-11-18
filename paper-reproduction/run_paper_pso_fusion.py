"""Fit FP-TBF-DI fusion weights with PSO on previously labeled predictions."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/paper_c_joint_reduction/predictions.csv'
EQUAL = np.full(3, 1 / 3)


def simplex(values):
    ordered = np.sort(values, axis=1)[:, ::-1]
    offsets = (ordered.cumsum(axis=1) - 1) / np.arange(1, 4)
    active = (ordered > offsets).sum(axis=1) - 1
    return np.maximum(values - offsets[np.arange(len(values)), active, None], 0)


def make_fitness(scores, labels):
    def fitness(weights):
        correct = ((scores @ weights.T >= 0) == labels[:, None]).sum(axis=0)
        return correct - 0.001 * ((weights - EQUAL)**2).sum(axis=1)
    return fitness


def initial_weights(rng):
    positions = rng.dirichlet(np.ones(3), size=96)
    positions[:4] = np.vstack((EQUAL, np.eye(3)))
    return positions


def pso(scores, labels, seed=20261117):
    rng = np.random.default_rng(seed)
    positions = initial_weights(rng)
    velocities = np.zeros_like(positions)
    fitness = make_fitness(scores, labels)

    best = positions.copy()
    values = fitness(best)
    for _ in range(100):
        global_best = best[values.argmax()]
        velocities = (0.65 * velocities
                      + 1.4 * rng.random(positions.shape) * (best - positions)
                      + 1.4 * rng.random(positions.shape) * (global_best - positions))
        new_positions = simplex(positions + velocities)
        velocities = new_positions - positions
        positions = new_positions
        new_values = fitness(positions)
        improved = new_values > values
        values[improved] = new_values[improved]
        best[improved] = positions[improved]
    weight = best[values.argmax()]
    assert np.all(weight >= 0) and np.isclose(weight.sum(), 1)
    return weight


def arrays(rows):
    scores = np.array([[float(r[k]) for k in ('A', 'B', 'simple_c')] for r in rows])
    labels = np.array([int(r['label']) == 1 for r in rows])
    equal = scores.mean(axis=1)
    np.testing.assert_allclose(equal, [float(r['simple_fusion']) for r in rows], atol=1e-12, rtol=0)
    return scores, labels, equal


def evaluate(labels, scores):
    pred = scores >= 0
    return dict(n=len(labels), correct=int((labels == pred).sum()),
                fn=int((labels & ~pred).sum()), fp=int((~labels & pred).sum()))


def study(optimizer, check_grid=False):
    groups = defaultdict(list)
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            groups[row['evaluation'], int(row['seed']), int(row['batch'])].append(row)
    splits = json.loads((ROOT / 'results/online_study/splits.json').read_text())
    audited = []
    holdouts = []
    pooled_rows = []
    for split in splits:
        seed = split['seed']
        for batch in (50, 100):
            rows = groups['stream', seed, batch]
            assert [int(r['row_index']) for r in rows] == split['train_stream'][300:]
            assert not set(split['train_stream']) & set(split['holdout'])
            scores, labels, equal = arrays(rows)
            pooled_rows.extend(rows)
            for start in range(0, len(rows), batch):
                stop = min(start + batch, len(rows))
                assert all(int(r['training_n']) == 300 + start for r in rows[start:stop])
                weight = optimizer(scores[:start], labels[:start]) if start else EQUAL
                assert np.all(weight >= 0) and np.isclose(weight.sum(), 1)
                audited.append(dict(seed=seed, batch=batch, history_n=start, weights=weight.tolist(),
                                    equal=evaluate(labels[start:stop], equal[start:stop]),
                                    pso=evaluate(labels[start:stop], scores[start:stop] @ weight)))
            candidate = optimizer(scores, labels)
            held = groups['holdout_849', seed, batch]
            assert len(held) == 213 and all(int(r['training_n']) == 849 for r in held)
            rows_by_id = {int(r['row_index']): r for r in held}
            assert set(rows_by_id) == set(split['holdout'])
            hx, hy, he = arrays([rows_by_id[i] for i in split['holdout']])
            holdouts.append(dict(seed=seed, batch=batch, weights=candidate.tolist(),
                                 equal=evaluate(hy, he), pso=evaluate(hy, hx @ candidate)))

    scores, labels, equal = arrays(pooled_rows)
    fixed = optimizer(scores, labels)
    apparent = dict(weights=fixed.tolist(), equal=evaluate(labels, equal),
                    pso=evaluate(labels, scores @ fixed))
    assert apparent['equal']['n'] == 3294 and apparent['equal']['correct'] == 3175
    if check_grid:
        resolution = 200
        grid = np.array([(i, j, resolution - i - j)
                         for i in range(resolution + 1)
                         for j in range(resolution - i + 1)]) / resolution
        grid_best = 0
        for start in range(0, len(grid), 256):
            count = ((scores @ grid[start:start+256].T >= 0) == labels[:, None]).sum(axis=0)
            grid_best = max(grid_best, int(count.max()))
        apparent['grid_step'] = 1 / resolution
        apparent['grid_best_correct'] = grid_best
    return dict(prequential=audited, retrospective_holdout=holdouts, pooled_in_sample=apparent)


def main():
    result = study(pso, check_grid=True)
    output = ROOT / 'results/paper_pso_fusion'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name in ('prequential', 'retrospective_holdout'):
        records = result[name]
        print(name, {method: {key: sum(r[method][key] for r in records)
                              for key in ('n', 'correct', 'fn', 'fp')}
                     for method in ('equal', 'pso')})
    print('pooled_in_sample', result['pooled_in_sample'])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
