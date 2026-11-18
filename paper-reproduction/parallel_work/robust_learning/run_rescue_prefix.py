"""Evaluate fixed whole-prefix auxiliary classifiers on champion negatives."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_prequential_evidence import metrics, paired_counts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from run_original_closeout import load_stream

METHODS = ('numeric', 'categorical', 'unanimous')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_policy(path, policy):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [row for row in csv.DictReader(stream) if row['policy'] == policy]


def fit_score(x, y, query):
    model = LogisticRegression(C=1, solver='liblinear', max_iter=1000)
    return model.fit(x, y).decision_function(query)


def rescue(raw, labels, start, stop, reference):
    history, y = raw[:start], labels[:start]
    query = raw[start:stop]
    targets = np.flatnonzero(reference < 0)
    negative_keys = {tuple(row) for row in history[y == -1]}
    positive_keys = {tuple(row) for row in history[y == 1]}
    keys = [tuple(query[i]) for i in targets]
    eligible = targets[[key not in negative_keys for key in keys]]
    result = {method: reference.copy() for method in METHODS}
    if len(eligible) == 0:
        return result, keys, negative_keys, positive_keys
    query = query[eligible]
    scaler = StandardScaler().fit(history)
    numeric = fit_score(scaler.transform(history), y, scaler.transform(query))
    encoder = OneHotEncoder(handle_unknown='ignore').fit(history)
    design, query = encoder.transform(history), encoder.transform(query)
    categorical = fit_score(design, y, query)
    supported = categorical >= 0
    rng = np.random.default_rng(1105)
    classes = [np.flatnonzero(y == label) for label in (-1, 1)]
    for _ in range(32):
        sample = np.concatenate([rng.choice(group, size=len(group), replace=True)
                                 for group in classes])
        supported &= fit_score(design[sample], y[sample], query) >= 0
    for method, scores, select in (
        ('numeric', numeric, numeric >= 0),
        ('categorical', categorical, categorical >= 0),
        ('unanimous', categorical, supported),
    ):
        result[method][eligible[select]] = scores[select]
    return result, keys, negative_keys, positive_keys


def comparison(y, reference, candidate):
    repaired_fn, new_fn = paired_counts(y, reference, candidate, 1)
    repaired_fp, new_fp = paired_counts(y, reference, candidate, -1)
    return dict(new_fn=new_fn, repaired_fn=repaired_fn,
                new_fp=new_fp, repaired_fp=repaired_fp)


def main():
    split_path = ROOT/'results/online_study/splits.json'
    baseline_path = HERE/'experiment_hard_width_bank1701/predictions.csv'
    champion_path = HERE/'experiment_rescue_geometry/add_only_predictions.csv'
    manifest = baseline_path.with_name('manifest.json')
    prior = {r['seed']: r for r in json.loads(manifest.read_text())['inputs']}
    baseline_rows = read_policy(baseline_path, 'baseline')
    champion_rows = read_policy(champion_path, 'exact_recurrence')
    configs, predictions = [], []
    for split in json.loads(split_path.read_text()):
        seed, ids = split['seed'], split['train_stream']
        if seed not in (17, 29, 43):
            continue
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(split['holdout'])
        raw, labels = load_stream(ROOT/'data/dataset_1062.txt', ids)
        assert digest(raw.tobytes()) == prior[seed]['allowed_x_sha256']
        assert digest(labels.tobytes()) == prior[seed]['allowed_y_sha256']
        y = labels[300:]
        for batch in (50, 100):
            scores = {}
            for name, rows in (('baseline', baseline_rows), ('champion', champion_rows)):
                chosen = [r for r in rows if (int(r['seed']), int(r['batch_size'])) == (seed, batch)]
                assert [int(r['row_index']) for r in chosen] == ids[300:]
                np.testing.assert_array_equal([int(r['label']) for r in chosen], y)
                np.testing.assert_array_equal([int(r['stage']) for r in chosen],
                                              np.arange(549)//batch + 1)
                scores[name] = np.array([float(r['score']) for r in chosen])
            reference = scores['champion']
            candidate = {name: reference.copy() for name in METHODS}
            witness = dict(absent=0, negative_only=0, conflict=0, positive_only=0)
            for start in range(300, len(raw), batch):
                stop = min(start + batch, len(raw))
                block = slice(start-300, stop-300)
                outcome, keys, negative, positive = rescue(raw, labels, start, stop, reference[block])
                changed_labels = labels.copy()
                changed_labels[start:] *= -1
                replay, *_ = rescue(raw, changed_labels, start, stop, reference[block])
                for name in METHODS:
                    np.testing.assert_array_equal(replay[name], outcome[name])
                    candidate[name][block] = outcome[name]
                for key, label in zip(keys, y[block][reference[block] < 0], strict=True):
                    if label == 1:
                        category = {(False, False): 'absent', (True, False): 'negative_only',
                                    (True, True): 'conflict', (False, True): 'positive_only'}
                        witness[category[key in negative, key in positive]] += 1
            record = dict(seed=seed, batch_size=batch, witness=witness,
                          baseline=metrics(y, scores['baseline']), champion=metrics(y, reference),
                          methods={})
            for name, proposed in candidate.items():
                np.testing.assert_array_equal(proposed[reference >= 0], reference[reference >= 0])
                record['methods'][name] = dict(metrics=metrics(y, proposed),
                    vs_champion=comparison(y, reference, proposed),
                    vs_baseline=comparison(y, scores['baseline'], proposed))
                predictions.extend(dict(seed=seed, batch_size=batch, stage=i//batch+1,
                    row_index=rid, label=int(label), policy=name, score=score)
                    for i, (rid, label, score) in enumerate(zip(ids[300:], y, proposed, strict=True)))
            configs.append(record)
            print(seed, batch, witness, [v['vs_champion'] for v in record['methods'].values()], flush=True)
    assert len(configs) == 6 and len(predictions) == 9882
    out = HERE/'experiment_rescue_prefix'
    out.mkdir(exist_ok=False)
    paths = (split_path, baseline_path, champion_path, manifest, Path(__file__),
             HERE/'protocol_rescue_prefix.md', HERE/'run_prequential_evidence.py', ROOT/'run_original_closeout.py')
    result = dict(configurations=configs, replayed_scores=len(predictions),
                  holdout_parsed_or_evaluated=0,
                  source_hashes={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths})
    (out/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == '__main__':
    main()
