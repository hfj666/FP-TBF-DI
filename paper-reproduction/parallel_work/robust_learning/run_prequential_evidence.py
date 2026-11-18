"""Replay a predeclared deployment policy on already verified shadow scores."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, betaincc
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHADOW = HERE / 'experiment_block_guard_bank1701'


def evidence(wins, losses):
    log_scale = (wins + losses + 1) * np.log(2) + betaln(wins + 1, losses + 1)
    return float(np.exp(log_scale) * betaincc(wins + 1, losses + 1, .5))


def paired_counts(y, baseline, candidate, label):
    mask = y == label
    reference = (baseline[mask] >= 0) == (label == 1)
    proposed = (candidate[mask] >= 0) == (label == 1)
    return int(np.sum(proposed & ~reference)), int(np.sum(reference & ~proposed))


def metrics(y, scores):
    predicted = scores >= 0
    sensitivity = float(predicted[y == 1].mean())
    specificity = float((~predicted[y == -1]).mean())
    return dict(accuracy=float(np.mean(predicted == (y == 1))),
                balanced_accuracy=(sensitivity + specificity) / 2,
                sensitivity=sensitivity, specificity=specificity,
                auc=float(roc_auc_score(y, scores)))


def main():
    audited = json.loads((SHADOW/'independent_audit.json').read_text())
    assert audited['fit_successes'] == 57 and audited['replayed_scores'] == 6588
    path = SHADOW/'predictions.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    splits = {int(s['seed']): s for s in json.loads((ROOT/'results/online_study/splits.json').read_text())}
    decisions, saved, paired = [], [], []
    for seed in (17, 29, 43):
        for batch in (50, 100):
            subset = [r for r in rows if int(r['seed']) == seed and int(r['batch_size']) == batch]
            base = [r for r in subset if r['policy'] == 'baseline']
            shadow = [r for r in subset if r['policy'] == 'block_guard']
            identity = lambda r: (r['stage'], int(r['row_index']), int(r['label']))
            assert [identity(r) for r in base] == [identity(r) for r in shadow]
            ids = [int(r['row_index']) for r in base]
            assert ids == splits[seed]['train_stream'][300:]
            assert not set(ids) & set(splits[seed]['holdout'])
            b = np.array([float(r['score']) for r in base])
            c = np.array([float(r['score']) for r in shadow])
            y = np.array([int(r['label']) for r in base])
            selected = b.copy()
            record = np.zeros((2, 2), dtype=int)
            for start in range(0, len(base), batch):
                stop = min(start + batch, len(base))
                e = [evidence(*pair) for pair in record]
                # Recompute the mixture by numerical integration independently.
                for (wins, losses), value in zip(record, e):
                    numeric = quad(lambda p: 2*(2*p)**int(wins)*(2-2*p)**int(losses), .5, 1)[0]
                    np.testing.assert_allclose(value, numeric, rtol=1e-8, atol=1e-12)
                accept = min(e) >= 20
                if accept:
                    selected[start:stop] = c[start:stop]
                decisions.append(dict(seed=seed, batch_size=batch, stage=start//batch+1,
                                      e_negative=e[0], e_positive=e[1], accepted=accept,
                                      history=record.tolist(), rows=stop-start))
                # Labels for the current batch enter the evidence after its choice.
                for i, label in enumerate((-1, 1)):
                    record[i] += paired_counts(y[start:stop], b[start:stop], c[start:stop], label)
            new_scores = [dict(row, score=str(value), policy='prequential_evidence')
                          for row, value in zip(base, selected, strict=True)]
            saved.extend(new_scores)
            fixed_fn, new_fn = paired_counts(y, b, selected, 1)
            fixed_fp, new_fp = paired_counts(y, b, selected, -1)
            paired.append(dict(seed=seed, batch_size=batch, baseline=metrics(y, b),
                               candidate=metrics(y, selected), new_fn=new_fn, resolved_fn=fixed_fn,
                               new_fp=new_fp, resolved_fp=fixed_fp))
    # Replay every batch choice from a fresh prefix of the shadow stream.
    for d in decisions:
        seed, batch, stage = d['seed'], d['batch_size'], d['stage']
        past = [r for r in rows if int(r['seed'])==seed and int(r['batch_size'])==batch and int(r['stage'])<stage]
        reference = [r for r in past if r['policy']=='baseline']
        proposal = [r for r in past if r['policy']=='block_guard']
        y = np.array([int(r['label']) for r in reference])
        b = np.array([float(r['score']) for r in reference])
        c = np.array([float(r['score']) for r in proposal])
        history = [list(paired_counts(y, b, c, label)) for label in (-1, 1)]
        assert history == d['history']
        assert all(evidence(*pair)>=20 for pair in history) == d['accepted']
        expected_policy = 'block_guard' if d['accepted'] else 'baseline'
        expected = [float(r['score']) for r in rows if int(r['seed'])==seed and int(r['batch_size'])==batch
                    and int(r['stage'])==stage and r['policy']==expected_policy]
        actual = [float(r['score']) for r in saved if int(r['seed'])==seed and int(r['batch_size'])==batch
                  and int(r['stage'])==stage]
        np.testing.assert_array_equal(expected, actual)
    result = dict(mode='prequential_evidence', accepted_batches=sum(d['accepted'] for d in decisions),
                  accepted_rows=sum(d['rows'] for d in decisions if d['accepted']),
                  decisions=decisions, pairs=paired, replayed_scores=len(saved),
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (path, SHADOW/'independent_audit.json', Path(__file__),
                                           HERE/'protocol_prequential_evidence.md')})
    out = HERE/'experiment_prequential_evidence'
    out.mkdir(exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(saved[0]))
        writer.writeheader()
        writer.writerows(saved)
    print(f"Accepted {result['accepted_batches']}/{len(decisions)} batches, replayed {len(saved)} scores")
    print('Max prior evidence by class', max(d['e_negative'] for d in decisions),
          max(d['e_positive'] for d in decisions))


if __name__ == '__main__':
    main()
