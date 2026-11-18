"""Fit Cauchy firing on HVDM distance on matched saved prefixes."""
import hashlib
import json

import numpy as np
from scipy.optimize import check_grad, minimize
from scipy.special import logsumexp


from diagnose_round05 import OLD, load_state
from diagnose_round07 import OUT
from run_component import HERE, ROOT, digest, load_stream, write_csv
from activation import restore
from activation_cauchy import objective

POLICY = 'hvdm_cauchy'


def main():
    output = OUT / 'experiment'
    output.mkdir(exist_ok=False)
    files = [HERE / name for name in ('PROTOCOL_07.md', 'run_round07.py', 'diagnose_round07.py', 'activation_cauchy.py',
                                       'activation.py', 'results_07/diagnosis.json',
                                       'results_01/state_audit.json', 'results_01/input_audit.json')]
    files.append(ROOT / 'results/online_study/splits.json')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    rng = np.random.default_rng(20261105)
    labels = np.array([-1, 1, -1, 1, -1, 1])
    args = rng.random((9, 6)), rng.integers(0, 2, size=9), rng.uniform(.65, 1, size=len(labels))
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
                confidence = np.array([r.confidence for r in model.raw_rules])
                if stage or initial is None:
                    result = minimize(objective, np.r_[np.full(64, np.log(.25)), 2 * labels],
                                      args=(model._distances(x[:end]), y[:end] == 1, confidence),
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
                    a = .5 * model._distances(x[end:stop]) * np.exp(-2 * theta[:64])
                    lf = np.log(confidence) - np.log1p(a)
                    scores = np.exp(lf - logsumexp(lf, axis=1, keepdims=True)) @ theta[64:]
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


