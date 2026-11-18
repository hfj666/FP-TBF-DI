"""Fit one HVDM activation normalization variant on the saved matched prefixes."""
import hashlib
import json

import numpy as np
from scipy.optimize import check_grad, minimize
from scipy.special import expit, logsumexp

from diagnose_round05 import OLD, OUT, load_state
from run_component import HERE, ROOT, digest, load_stream, write_csv
from activation import restore

POLICY = 'hvdm_group_l2'


def pool(logfire, labels, consequents, gradient=False):
    group_log_norm = np.column_stack([
        .5 * logsumexp(2 * logfire[:, labels == label], axis=1) for label in (-1, 1)
    ])
    mass = np.exp(group_log_norm - logsumexp(group_log_norm, axis=1, keepdims=True))
    weights = np.empty_like(logfire)
    derivative = np.empty_like(logfire) if gradient else None
    means = []
    for group, label in enumerate((-1, 1)):
        mask = labels == label
        lf = logfire[:, mask]
        conditional = np.exp(lf - logsumexp(lf, axis=1, keepdims=True))
        weights[:, mask] = mass[:, group, None] * conditional
        means.append(conditional @ consequents[mask])
    score = weights @ consequents
    if gradient:
        for group, label in enumerate((-1, 1)):
            mask = labels == label
            squared = np.exp(2 * (logfire[:, mask] - group_log_norm[:, group, None]))
            mean = means[group][:, None]
            derivative[:, mask] = (
                weights[:, mask] * (consequents[mask] - mean)
                + mass[:, group, None] * squared * (mean - score[:, None])
            )
    return score, weights, derivative


def objective(theta, distance, prior, labels, target):
    k = len(labels)
    consequents = theta[k:]
    scaled = distance * np.exp(-2 * theta[:k])
    score, weights, derivative = pool(prior - .5 * scaled, labels, consequents, gradient=True)
    loss = np.mean(np.logaddexp(0, score) - target * score) + .001 * np.mean(consequents**2)
    error = (expit(score) - target) / len(target)
    return loss, np.r_[np.sum(error[:, None] * derivative * scaled, axis=0),
                      weights.T @ error + .002 * consequents / k]


def main():
    output = OUT / 'experiment'
    output.mkdir(exist_ok=False)
    files = [HERE / name for name in ('PROTOCOL_05.md', 'run_round05.py', 'diagnose_round05.py',
                                       'activation.py', 'results_05/diagnosis.json',
                                       'results_01/state_audit.json', 'results_01/input_audit.json')]
    files.append(ROOT / 'results/online_study/splits.json')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    rng = np.random.default_rng(20261105)
    labels = np.array([-1, 1, -1, 1, -1, 1])
    args = rng.random((9, 6)), rng.normal(size=6), labels, rng.integers(0, 2, size=9)
    gradient_error = float(check_grad(lambda t: objective(t, *args)[0],
                                      lambda t: objective(t, *args)[1], rng.normal(size=12)))
    assert gradient_error < 1e-7
    (output / 'protocol.json').write_text(json.dumps(dict(
        hashes=hashes, threshold=0, gradient_error=gradient_error), indent=2), encoding='utf-8')
    audit = {(r['seed'], r['batch_size'], r['stage']): r['sha256']
             for r in json.loads((OLD / 'state_audit.json').read_text())}
    inputs = {r['seed']: r['allowed_stream_sha256']
              for r in json.loads((OLD / 'input_audit.json').read_text())}
    fits, predictions = [], []
    for sp in json.loads((ROOT / 'results/online_study/splits.json').read_text()):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(sp['holdout'])
        x, y = load_stream(ROOT / 'data/dataset_1062.txt', ids)
        assert digest(x, y, ids) == inputs[seed]
        initial = None
        for batch in (50, 100):
            for stage, end in enumerate([*range(300, 849, batch), 849]):
                state = load_state(seed, batch, stage, audit)
                assert state['prefix'] == end
                assert state['prefix_input_sha256'] == digest(x[:end], y[:end], ids[:end])
                snapshot = state['models']['hvdm_mean']
                model = restore(snapshot)
                labels = np.array([r.label for r in model.raw_rules])
                assert len(labels) == 64 and np.sum(labels == 1) == 32
                prior = np.log([r.confidence for r in model.raw_rules])
                if stage or initial is None:
                    result = minimize(objective, np.r_[np.full(64, np.log(.25)), 2 * labels],
                                      args=(model._distances(x[:end]), prior, labels, y[:end] == 1),
                                      jac=True, method='L-BFGS-B',
                                      bounds=[(np.log(.03), np.log(2.))] * 64 + [(-10., 10.)] * 64,
                                      options={'maxiter': 500})
                    theta = result.x
                    assert np.isfinite(theta).all() and np.isfinite(result.fun)
                    fits.append(dict(seed=seed, batch=batch if stage else 0, stage=stage,
                                     prefix=end, success=bool(result.success), message=str(result.message),
                                     iterations=int(result.nit), loss=float(result.fun), theta=theta.tolist()))
                    if not stage:
                        initial = snapshot, theta
                else:
                    original, theta = initial
                    assert snapshot == original
                if end < 849:
                    stop = min(end + batch, 849)
                    lf = prior - .5 * model._distances(x[end:stop]) * np.exp(-2 * theta[:64])
                    scores = pool(lf, labels, theta[64:])[0]
                    assert np.isfinite(scores).all()
                    predictions.extend(dict(seed=seed, batch_size=batch, stage=stage + 1, row_index=int(i),
                                            policy=POLICY, label=int(v), score=float(s))
                                       for i, v, s in zip(ids[end:stop], y[end:stop], scores, strict=True))
            print(f'{seed}/{batch} complete', flush=True)
    assert len(fits) == 54 and len(predictions) == 3294
    (output / 'fits.json').write_text(json.dumps(fits, indent=2), encoding='utf-8')
    write_csv(output / 'predictions.csv', predictions)


if __name__ == '__main__':
    main()
