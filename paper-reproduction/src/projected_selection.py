"""Capped simplex selection using fixed Sugeno activations."""
from time import perf_counter
import numpy as np
from scipy.special import expit


def project(vector, mass):
    if mass == len(vector):
        return np.ones_like(vector)
    low, high = vector.min()-1., vector.max()
    for _ in range(52):
        mid = (low+high)/2
        if np.clip(vector-mid, 0, 1).sum() > mass:
            low = mid
        else:
            high = mid
    result = np.clip(vector-(low+high)/2, 0, 1)
    assert abs(result.sum()-mass) < 1e-7
    return result


def geometry(x, rules, scaling):
    minimum, span = scaling
    logfire = np.column_stack([
        np.log(r.confidence) - 8*np.mean(((x[:, r.columns]-r.values)/span[list(r.columns)])**2, axis=1)
        for r in rules
    ])
    # A common row scale cancels in numerator/denominator.
    return np.exp(logfire-logfire.max(axis=1, keepdims=True))


def objective(weights, firing, consequents, target):
    denominator = firing @ weights
    score = (firing @ (weights*consequents))/denominator
    residual = (expit(score)-target)/len(target)/denominator
    gradient = consequents*(firing.T @ residual) - firing.T @ (residual*score)
    loss = np.mean(np.logaddexp(0, score)-target*score)
    return float(loss), gradient


def solve(firing, labels, y, joint, quota=32, iterations=100, time_limit=180.):
    begin = perf_counter()
    target = (np.asarray(y) == 1).astype(float)
    consequents = 2*labels
    parts = [np.flatnonzero(labels == c) for c in (-1, 1)]
    masses = [min(quota, len(p)) for p in parts]
    gate = np.zeros(len(labels))
    for p, mass in zip(parts, masses):
        gate[p] = mass/len(p)
    initial, gradient = objective(gate, firing, consequents, target)
    loss = initial
    accepted = 0
    evaluations = 1
    stop = 'linearized'
    if joint:
        step = min(1e6, 1/max(1e-12, np.max(abs(gradient))))
        stop = 'iteration_limit'
        for _ in range(iterations):
            for _ in range(24):
                trial = np.empty_like(gate)
                for p, mass in zip(parts, masses):
                    trial[p] = project(gate[p]-step*gradient[p], mass)
                delta = trial-gate
                if np.max(abs(delta)) <= 1e-9:
                    stop = 'small_projected_move'
                    break
                value, next_gradient = objective(trial, firing, consequents, target)
                evaluations += 1
                if value <= loss+1e-4*(gradient @ delta):
                    gate, loss, gradient = trial, value, next_gradient
                    accepted += 1
                    step = min(1e6, 2*step)
                    break
                step /= 2
            else:
                stop = 'line_search_limit'
            if perf_counter()-begin > time_limit:
                raise TimeoutError('mask selector exceeded 180s')
            if stop != 'iteration_limit':
                break
    rank = -gate if joint else gradient
    indices = np.concatenate([p[np.argsort(rank[p], kind='stable')[:m]] for p, m in zip(parts, masses)])
    hard = np.zeros(len(labels))
    hard[indices] = 1.
    rounded, _ = objective(hard, firing, consequents, target)
    return indices, dict(objective=loss, initial_mask_bce=initial, hard_mask_bce=rounded,
                         gate=gate.tolist(), indices=indices.tolist(), accepted=accepted,
                         evaluations=evaluations, selector_stop=stop)


def select(pool, x, y, scaling, joint=True):
    entries = sorted(pool.pools[-1]+pool.pools[1], key=lambda r: (r[0], r[1].label))
    rules = [r[1] for r in entries]
    labels = np.array([r.label for r in rules])
    indices, stats = solve(geometry(x, rules, scaling), labels, y, joint)
    stats.update(candidate_count=len(rules), selected=len(indices))
    return [rules[i] for i in indices], stats
