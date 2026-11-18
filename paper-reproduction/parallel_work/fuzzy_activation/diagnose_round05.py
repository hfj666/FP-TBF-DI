"""Replay paired false negatives using saved, pre-prediction HVDM states."""
import csv
import gzip
import hashlib
import json
from collections import Counter

import numpy as np

from run_component import HERE, ROOT, digest, load_stream
from activation import restore

OLD = HERE / 'results_01'
OUT = HERE / 'results_05'
LOGIC = ROOT / 'results/direct_comparison_v2/predictions.csv'


def read_predictions(path, policy):
    with path.open(encoding='utf-8-sig', newline='') as source:
        rows = {}
        for row in csv.DictReader(source):
            if row['policy'] == policy:
                key = tuple(int(row[k]) for k in ('seed', 'batch_size', 'stage', 'row_index'))
                assert key not in rows
                rows[key] = int(row['label']), float(row['score'])
    return rows


def load_state(seed, batch, stage, audit):
    raw = gzip.open(OLD / f'state_s{seed}_b{batch}_t{stage}.json.gz', 'rt', encoding='utf-8').read()
    assert hashlib.sha256(raw.encode()).hexdigest() == audit[seed, batch, stage]
    return json.loads(raw)


def main():
    hvdm = read_predictions(OLD / 'predictions.csv', 'hvdm_mean')
    logic = read_predictions(LOGIC, 'logistic_cumulative')
    assert hvdm.keys() == logic.keys() and len(hvdm) == 3294
    assert all(hvdm[k][0] == logic[k][0] for k in hvdm)
    targets = {k for k, (label, score) in hvdm.items() if label == 1 and score < 0 <= logic[k][1]}
    assert len(targets) == 23
    audit = {(r['seed'], r['batch_size'], r['stage']): r['sha256']
             for r in json.loads((OLD / 'state_audit.json').read_text())}
    inputs = {r['seed']: r['allowed_stream_sha256']
              for r in json.loads((OLD / 'input_audit.json').read_text())}
    events = []
    error = 0.
    for sp in json.loads((ROOT / 'results/online_study/splits.json').read_text()):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(sp['holdout'])
        x, y = load_stream(ROOT / 'data/dataset_1062.txt', ids)
        assert digest(x, y, ids) == inputs[seed]
        for batch in (50, 100):
            for offset, end in enumerate(range(300, 849, batch)):
                indices = [i for i in range(end, min(end + batch, 849))
                           if (seed, batch, offset + 1, int(ids[i])) in targets]
                if not indices:
                    continue
                state = load_state(seed, batch, offset, audit)
                assert state['prefix'] == end
                assert state['prefix_input_sha256'] == digest(x[:end], y[:end], ids[:end])
                model = restore(state['models']['hvdm_mean'])
                rules = model.raw_rules
                positive = np.array([r.label == 1 for r in rules])
                assert positive.sum() == 32 and len(rules) == 64
                consequents = model.theta[64:]
                distances = model._distances(x[indices])
                weights = model.firing_weights(x[indices])
                for i, d, w in zip(indices, distances, weights, strict=True):
                    key = seed, batch, offset + 1, int(ids[i])
                    exact = np.array([np.array_equal(x[i, list(r.columns)], r.values) for r in rules])
                    error = max(error, abs(w @ consequents - hvdm[key][1]))
                    assert y[i] == 1
                    best_pos = int(np.argmax(np.where(positive, w, -1)))
                    best_neg = int(np.argmax(np.where(~positive, w, -1)))
                    mass = float(w[positive].sum())
                    top = 'positive' if w[best_pos] > w[best_neg] else 'negative' if w[best_pos] < w[best_neg] else 'tie'
                    events.append(dict(seed=seed, batch=batch, stage=offset + 1, row=int(ids[i]),
                                       hvdm_score=hvdm[key][1], logistic_score=logic[key][1],
                                       exact_pos=int(sum(exact & positive)), exact_neg=int(sum(exact & ~positive)),
                                       nearest_pos=float(min(d[positive])), nearest_neg=float(min(d[~positive])),
                                       positive_mass=mass, top_label=top,
                                       best_pos_rule=best_pos, best_neg_rule=best_neg,
                                       best_pos_weight=float(w[best_pos]), best_neg_weight=float(w[best_neg]),
                                       best_pos_consequent=float(consequents[best_pos]),
                                       best_neg_consequent=float(consequents[best_neg]),
                                       signed_vote=2 * mass - 1,
                                       pos_group_contribution=float(w[positive] @ consequents[positive]),
                                       neg_group_contribution=float(w[~positive] @ consequents[~positive])))
    assert len(events) == len(targets) and error < 1e-12
    OUT.mkdir(exist_ok=True)
    result = dict(events=events, unique_rows=len({r['row'] for r in events}), max_score_error=error,
                  groups=[dict(exact_pos=ep, top_label=top, positive_mass_majority=majority, count=n)
                          for (ep, top, majority), n in sorted(Counter(
                              (r['exact_pos'] > 0, r['top_label'], r['positive_mass'] >= .5) for r in events).items())])
    (OUT / 'diagnosis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}, indent=2))
    for r in events:
        print(*(r[k] for k in ('seed','batch','stage','row','exact_pos','exact_neg','top_label')),
              *(round(r[k], 4) for k in ('nearest_pos','nearest_neg','positive_mass','pos_group_contribution','neg_group_contribution','hvdm_score')))


if __name__ == '__main__':
    main()
