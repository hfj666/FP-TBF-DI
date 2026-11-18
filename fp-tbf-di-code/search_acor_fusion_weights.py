"""Search FP-TBF-DI fusion weights from labeled development scores."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


EQUAL = np.full(3, 1 / 3)
DEFAULT_SEED = 20261117


def _project_simplex(values):
    """Project each candidate onto nonnegative weights summing to one."""
    ordered = np.sort(values, axis=1)[:, ::-1]
    offsets = (ordered.cumsum(axis=1) - 1) / np.arange(1, 4)
    active = (ordered > offsets).sum(axis=1) - 1
    return np.maximum(values - offsets[np.arange(len(values)), active, None], 0)


def ant_colony(scores, labels, seed=DEFAULT_SEED):
    """Return the best A, B, C weights found for labels encoded as -1/+1."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels) or scores.shape != (len(labels), 3)
            or not np.isfinite(scores).all() or set(labels) != {-1, 1}):
        raise ValueError("Expected finite A, B, C scores and both -1/+1 labels.")
    positive = labels == 1

    def fitness(weights):
        correct = ((scores @ weights.T >= 0) == positive[:, None]).sum(axis=0)
        return correct - 0.001 * ((weights - EQUAL)**2).sum(axis=1)

    rng = np.random.default_rng(seed)
    size = 96
    archive = rng.dirichlet(np.ones(3), size=size)
    archive[:4] = np.vstack((EQUAL, np.eye(3)))
    values = fitness(archive)
    order = np.argsort(-values, kind='stable')
    archive, values = archive[order], values[order]
    rank_weights = np.exp(-0.5 * (np.arange(size) / (0.1 * size))**2)
    probabilities = rank_weights / rank_weights.sum()
    for _ in range(100):
        selected = rng.choice(size, size=size, p=probabilities)
        bandwidth = 0.85 * np.abs(archive[:, None] - archive[None, :]).sum(axis=1) / (size - 1)
        proposals = _project_simplex(rng.normal(archive[selected], bandwidth[selected]))
        candidates = np.vstack((archive, proposals))
        candidate_values = np.concatenate((values, fitness(proposals)))
        keep = np.argsort(-candidate_values, kind='stable')[:size]
        archive, values = candidates[keep], candidate_values[keep]
    return archive[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path, help='Development CSV with columns A,B,C,label')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        with args.scores.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'A', 'B', 'C', 'label'}.issubset(reader.fieldnames or ()):
                raise ValueError('CSV must contain columns A,B,C,label.')
            rows = list(reader)
        scores = np.array([[float(row[key]) for key in ('A', 'B', 'C')] for row in rows])
        labels = np.array([int(row['label']) for row in rows])
        with threadpool_limits(limits=1):
            weights = ant_colony(scores, labels, args.seed)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    correct = int(((scores @ weights >= 0) == (labels == 1)).sum())
    print(json.dumps({
        'seed': args.seed,
        'weights': dict(zip(('A', 'B', 'C'), weights.tolist(), strict=True)),
        'development_rows': len(rows),
        'development_correct': correct,
        'development_accuracy': correct / len(rows),
    }, indent=2))


if __name__ == '__main__':
    main()
