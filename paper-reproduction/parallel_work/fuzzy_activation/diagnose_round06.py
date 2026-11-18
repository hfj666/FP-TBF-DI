"""Decompose matched activation gaps and pool/refit score changes."""
import json

import numpy as np
from scipy.special import logsumexp

from diagnose_round05 import OLD, OUT as PREV, ROOT, load_state, read_predictions
from run_component import HERE, digest, load_stream
from activation import restore
from run_round05 import POLICY, pool

OUT = HERE / 'results_06'


def gap_terms(d, logq, widths, positive):
    def gap(logfire):
        return float(logsumexp(logfire[positive]) - logsumexp(logfire[~positive]))

    fixed = -.5 * d / .25**2
    fitted = -.5 * d * np.exp(-2 * widths)
    g00, g10, g01, g11 = [gap(f) for f in (fixed, fixed + logq, fitted, fitted + logq)]
    confidence = .5 * ((g10 - g00) + (g11 - g01))
    width = .5 * ((g01 - g00) + (g11 - g10))
    lf = fitted + logq
    peak_gap = float(lf[positive].max() - lf[~positive].max())
    effective_counts = [float(np.exp(logsumexp(lf[m]) - lf[m].max())) for m in (positive, ~positive)]
    count_gap = float(np.log(effective_counts[0] / effective_counts[1]))
    np.testing.assert_allclose([g00 + confidence + width, peak_gap + count_gap], g11, atol=1e-12)
    return dict(fixed_distance=g00, confidence=confidence, width=width, gap=g11,
                peak_gap=peak_gap, count_gap=count_gap, effective_counts=effective_counts)


def main():
    prior_audit = json.loads((PREV / 'experiment/verification.json').read_text(encoding='utf-8'))
    targets = {tuple(k): 'pool_new_fn' for k in prior_audit['pairs']['HVDM']['new_fn']}
    diagnosis = json.loads((PREV / 'diagnosis.json').read_text(encoding='utf-8'))
    low = {(r['seed'], r['batch'], r['stage'], r['row']): 'low_positive'
           for r in diagnosis['events'] if r['exact_pos'] and r['positive_mass'] < .5}
    assert len(targets) == 14 and len(low) == 16 and not targets.keys() & low.keys()
    targets.update(low)
    hvdm = read_predictions(OLD / 'predictions.csv', 'hvdm_mean')
    candidate = read_predictions(PREV / 'experiment/predictions.csv', POLICY)
    saved = json.loads((PREV / 'experiment/fits.json').read_text(encoding='utf-8'))
    fits = {(r['seed'], r['batch'], r['stage']): np.array(r['theta']) for r in saved}
    audit = {(r['seed'], r['batch_size'], r['stage']): r['sha256']
             for r in json.loads((OLD / 'state_audit.json').read_text(encoding='utf-8'))}
    rows = []
    for sp in json.loads((ROOT / 'results/online_study/splits.json').read_text(encoding='utf-8')):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(sp['holdout'])
        x, y = load_stream(ROOT / 'data/dataset_1062.txt', ids)
        positions = {int(row): i for i, row in enumerate(ids)}
        for key, group in sorted(targets.items()):
            s, batch, stage, row = key
            if s != seed:
                continue
            i = positions[row]
            offset = stage - 1
            end = 300 + offset * batch
            state = load_state(seed, batch, offset, audit)
            assert end <= i < min(end + batch, 849) and y[i] == 1
            assert digest(x[:end], y[:end], ids[:end]) == state['prefix_input_sha256']
            model = restore(state['models']['hvdm_mean'])
            old = model.theta
            new = fits[seed, batch if offset else 0, offset]
            rules = model.raw_rules
            labels = np.array([r.label for r in rules])
            positive = labels == 1
            assert positive.sum() == 32 and len(labels) == 64
            d = model._distances(x[i:i+1])[0]
            q = np.log([r.confidence for r in rules])
            exact = np.array([np.array_equal(x[i, list(r.columns)], r.values) for r in rules])
            old_terms = gap_terms(d, q, old[:64], positive)
            new_terms = gap_terms(d, q, new[:64], positive)

            def logfire(theta):
                return q - .5 * d * np.exp(-2 * theta[:64])

            lf_old, lf_new = logfire(old), logfire(new)
            # Evaluate all 8 swaps for a symmetric allocation of score change.
            scores = {}
            for norm in (0, 1):
                for widths, lf in enumerate((lf_old, lf_new)):
                    for conseq, c in enumerate((old[64:], new[64:])):
                        scores[norm, widths, conseq] = float(pool(lf[None], labels, c)[0][0] if norm
                            else np.exp(lf - logsumexp(lf)) @ c)
            np.testing.assert_allclose([scores[0,0,0], scores[1,1,1]],
                                       [hvdm[key][1], candidate[key][1]], atol=1e-12)
            import itertools
            deltas = np.zeros(3)
            for order in itertools.permutations(range(3)):
                state_key = [0, 0, 0]
                for factor in order:
                    before = scores[tuple(state_key)]
                    state_key[factor] = 1
                    deltas[factor] += (scores[tuple(state_key)] - before) / 6
            def pooled_gap(lf):
                return float(.5 * (logsumexp(2 * lf[positive]) - logsumexp(2 * lf[~positive])))

            rows.append(dict(key=key, group=group, exact=[int(np.sum(exact & m)) for m in (positive, ~positive)],
                             exact_confidence=[[round(float(v), 8) for v in np.exp(q[exact & m])] for m in (positive, ~positive)],
                             old=old_terms, refit=new_terms,
                             pooled_gap_old_width=pooled_gap(lf_old), pooled_gap_new_width=pooled_gap(lf_new),
                             old_score=scores[0,0,0], pooled_score_old_params=scores[1,0,0],
                             new_score=scores[1,1,1], score_deltas=dict(zip(('pool','width','consequent'), deltas.tolist()))))
    assert len(rows) == 30
    OUT.mkdir(exist_ok=True)
    (OUT / 'decomposition.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for group in ('pool_new_fn', 'low_positive'):
        rr = [r for r in rows if r['group'] == group]
        print(group, 'distinct', len({r['key'][3] for r in rr}), 'exact mean', np.mean([r['exact'] for r in rr], axis=0))
        for model in ('old', 'refit'):
            for factor in ('fixed_distance','confidence','width','peak_gap','count_gap','gap'):
                values = np.array([r[model][factor] for r in rr])
                print(model, factor, round(values.mean(), 3), 'negative', sum(values < -1e-10))
        for factor in ('pool','width','consequent'):
            values = np.array([r['score_deltas'][factor] for r in rr])
            print('score', factor, round(values.mean(), 3), 'negative', sum(values < -1e-10))
        print('fixed param pooling flips to negative', sum(r['pooled_score_old_params'] < 0 for r in rr))
        print('negative pooled gaps', [sum(r[k] < 0 for r in rr) for k in ('pooled_gap_old_width','pooled_gap_new_width')])
        print('events', [(r['key'][3],r['exact'],round(r['old']['confidence'],2),round(r['old']['width'],2)) for r in rr])


if __name__ == '__main__':
    main()


