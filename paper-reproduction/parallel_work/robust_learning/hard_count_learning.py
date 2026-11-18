"""Coordinate breakpoint search with a hard training false-positive budget."""
import numpy as np
from scipy.special import logsumexp
from src.models import SparseSugeno


def counts(scores, positive):
    predicted = scores >= 0
    return int(np.sum(positive & ~predicted)), int(np.sum(~positive & predicted))


def best_interval(roots, slopes, evaluate, positive, lower, upper, current, cap):
    """Count sign changes at sorted roots and choose a feasible interval midpoint."""
    selected = np.isfinite(roots) & (roots > lower) & (roots < upper)
    if not np.any(selected):
        return None
    root_values, inverse = np.unique(roots[selected], return_inverse=True)
    boundaries = np.r_[lower, root_values, upper]
    middle = (boundaries[:-1] + boundaries[1:]) / 2
    fn, fp = counts(evaluate(middle[0]), positive)
    signs = np.sign(slopes[selected])
    delta_fn = np.bincount(inverse, -signs * positive[selected], minlength=len(root_values))
    delta_fp = np.bincount(inverse, signs * ~positive[selected], minlength=len(root_values))
    missed = np.r_[fn, fn + np.cumsum(delta_fn)]
    false_alarms = np.r_[fp, fp + np.cumsum(delta_fp)]
    feasible = np.flatnonzero(false_alarms <= cap)
    if not len(feasible):
        return None
    order = np.lexsort((middle[feasible], np.abs(middle[feasible] - current),
                        false_alarms[feasible], missed[feasible]))
    return float(middle[feasible[order[0]]])


def score(theta, distances, prior):
    count = len(prior)
    logits = np.log(prior) - .5 * distances * np.exp(-2 * theta[:count])
    weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return weights @ theta[count:]


def coordinate(theta, distances, prior, j, width):
    count = len(prior)
    coefficients = theta[count:]
    log_fire = np.log(prior) - .5 * distances * np.exp(-2 * theta[:count])
    # Scale the unchanged terms and the varied term by the same positive factor.
    scale = np.maximum(np.max(log_fire, axis=1), np.log(prior[j]))
    fires = np.exp(log_fire - scale[:, None])
    other_coefficients = coefficients.copy()
    other_coefficients[j] = 0
    other = fires @ other_coefficients
    if not width:
        slope = fires[:, j]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            roots = -other / slope
        return roots, slope, lambda c: other + slope * c
    amplitude = coefficients[j] * np.exp(np.log(prior[j]) - scale)
    column = distances[:, j]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        inverse_variance = -2 * np.log(-other / amplitude) / column
        roots = -.5 * np.log(inverse_variance)
    return roots, amplitude, lambda u: other + amplitude * np.exp(-.5 * column * np.exp(-2*u))


def fit_component(base, x, y, mode, bank_seed):
    model = SparseSugeno(base.rules, maxiter=500)
    model.minimum, model.span = base.minimum.copy(), base.span.copy()
    distances = model._distances(x)
    prior = np.array([rule.confidence for rule in model.rules])
    positive = y == 1
    theta = base.theta.copy()
    baseline_fn, cap = counts(score(theta, distances, prior), positive)
    best = (baseline_fn, cap)
    width = mode == 'hard_width'
    lower, upper = (np.log(.03), np.log(2.)) if width else (-10., 10.)
    moves = checked = 0
    for sweep in range(1, 4):
        start_moves = moves
        for j in range(len(prior)):
            checked += 1
            index = j if width else len(prior) + j
            roots, slopes, evaluate = coordinate(theta, distances, prior, j, width)
            value = best_interval(roots, slopes, evaluate, positive, lower, upper, theta[index], cap)
            if value is None:
                continue
            trial = theta.copy()
            trial[index] = value
            trial_counts = counts(score(trial, distances, prior), positive)
            if trial_counts[1] <= cap and trial_counts < best:
                theta, best = trial, trial_counts
                moves += 1
        if moves == start_moves:
            break
    model.theta = theta
    assert best == counts(model.decision_function(x), positive)
    model.optimization = dict(success=True, iterations=sweep, function_evaluations=checked,
                              moves=moves, train_fn=best[0], train_fp=best[1],
                              train_baseline_fn=baseline_fn, train_fp_cap=cap,
                              parameter_count=len(theta))
    return model


def test_synthetic():
    rng = np.random.default_rng(154)
    distances = rng.uniform(.01, .5, (19, 4))
    prior = rng.uniform(.7, 1, 4)
    theta = np.r_[rng.uniform(-2, -1, 4), [3., -2., 1., -4.]]
    positive = np.arange(len(distances)) % 2 == 0
    for width in (False, True):
        lower, upper = (np.log(.03), np.log(2.)) if width else (-10., 10.)
        for j in range(4):
            index = j if width else 4+j
            roots, slopes, evaluate = coordinate(theta, distances, prior, j, width)
            edges = np.unique(np.r_[lower, roots[(roots > lower) & (roots < upper)], upper])
            mid = (edges[:-1]+edges[1:])/2
            costs = []
            for value in mid:
                trial = theta.copy()
                trial[index] = value
                actual = score(trial, distances, prior)
                np.testing.assert_array_equal(evaluate(value)>=0, actual>=0)
                costs.append(counts(actual, positive))
            for cap in range(10):
                selected = best_interval(roots, slopes, evaluate, positive, lower, upper, theta[index], cap)
                if len(mid)>1:
                    feasible = [c for c in costs if c[1] <= cap]
                    assert (counts(evaluate(selected), positive) if selected is not None else None) == (min(feasible) if feasible else None)
    print('Both breakpoint derivations and capped scans match exhaustive interval evaluation')
