"""Add stable pairwise score interactions and replay a fixed branch replacement."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from run_rescue_prefix import HERE, ROOT, comparison, digest, fit_score, load_stream, metrics

METHODS = ('numeric', 'numeric_stable', 'categorical', 'categorical_stable')


def interact(train, query):
    expansion = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit(train)
    scale = np.sqrt(2 / 17)
    result = []
    for block in (train, query):
        pairwise = expansion.transform(block)[:, block.shape[1]:] * scale
        result.append(hstack((block, pairwise), format='csr') if hasattr(block, 'tocsr')
                      else np.column_stack((block, pairwise)))
    return result


def predict(raw, labels, start, stop, reference):
    history, y = raw[:start], labels[:start]
    negative = {tuple(row) for row in history[y == -1]}
    idx = np.array([i for i in range(start, stop) if reference[i-start] < 0
                    and tuple(raw[i]) not in negative], dtype=int)
    result = {name: reference.copy() for name in METHODS}
    if not len(idx):
        return result
    for name, encoder in (('numeric', StandardScaler()),
                          ('categorical', OneHotEncoder(handle_unknown='ignore'))):
        train, query = interact(encoder.fit_transform(history), encoder.transform(raw[idx]))
        full = fit_score(train, y, query)
        selected = full >= 0
        stable = selected.copy()
        rng = np.random.default_rng(1105)
        groups = [np.flatnonzero(y == label) for label in (-1, 1)]
        for _ in range(32):
            if not stable.any():
                break
            sample = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
            stable[stable] = fit_score(train[sample], y[sample], query[stable]) >= 0
        result[name][idx[selected]-start] = full[selected]
        result[name+'_stable'][idx[stable]-start] = full[stable]
    return result


def main():
    matched = ROOT/'parallel_work/three_branch_fusion_v1/results/matched_predictions.csv'
    split_path = ROOT/'results/online_study/splits.json'
    manifest = HERE/'experiment_hard_width_bank1701/manifest.json'
    inputs = {r['seed']: r for r in json.loads(manifest.read_text())['inputs']}
    archived = defaultdict(list)
    with matched.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            archived[int(row['seed']), int(row['batch_size'])].append(row)
    records, predictions = [], []
    for split in json.loads(split_path.read_text()):
        seed, ids = split['seed'], split['train_stream']
        if seed not in (17, 29, 43):
            continue
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(split['holdout'])
        raw, labels = load_stream(ROOT/'data/dataset_1062.txt', ids)
        assert digest(raw.tobytes()) == inputs[seed]['allowed_x_sha256']
        assert digest(labels.tobytes()) == inputs[seed]['allowed_y_sha256']
        y = labels[300:]
        for batch in (50, 100):
            rows = archived[seed, batch]
            assert [int(row['row_index']) for row in rows] == ids[300:]
            np.testing.assert_array_equal([int(row['label']) for row in rows], y)
            np.testing.assert_array_equal([int(row['stage']) for row in rows], np.arange(549)//batch+1)
            projection, hvdm, reference, ensemble = (
                np.array([float(row[key]) for row in rows])
                for key in ('projection', 'hvdm_offset', 'rescue', 'three_way'))
            np.testing.assert_array_equal(np.mean([projection, hvdm, reference], axis=0), ensemble)
            shared_fn = (y == 1) & (projection < 0) & (hvdm < 0) & (reference < 0)
            candidates = {name: reference.copy() for name in METHODS}
            for start in range(300, len(raw), batch):
                stop = min(start+batch, len(raw))
                block = slice(start-300, stop-300)
                changed = labels.copy()
                changed[start:] *= -1
                output = predict(raw, labels, start, stop, reference[block])
                replay = predict(raw, changed, start, stop, reference[block])
                for name in METHODS:
                    np.testing.assert_array_equal(output[name], replay[name])
                    candidates[name][block] = output[name]
            record = dict(seed=seed, batch_size=batch, branch=metrics(y, reference),
                          ensemble=metrics(y, ensemble), shared_fn=int(shared_fn.sum()), methods={})
            for name, score in candidates.items():
                np.testing.assert_array_equal(score[reference >= 0], reference[reference >= 0])
                fusion = np.mean([projection, hvdm, score], axis=0)
                record['methods'][name] = dict(branch=metrics(y, score), ensemble=metrics(y, fusion),
                    paired_branch=comparison(y, reference, score), paired_ensemble=comparison(y, ensemble, fusion),
                    shared_fn_repaired=int((shared_fn & (fusion >= 0)).sum()))
                predictions.extend(dict(row, method=name, new_branch=float(s), new_ensemble=float(f))
                                   for row, s, f in zip(rows, score, fusion, strict=True))
            records.append(record)
            print(seed, batch, [(m['paired_branch'], m['paired_ensemble'], m['shared_fn_repaired'])
                               for m in record['methods'].values()], flush=True)
    assert len(records) == 6 and len(predictions) == 13176
    out = HERE/'experiment_rescue_interactions'
    out.mkdir(exist_ok=True)
    paths = (matched, split_path, manifest, Path(__file__), HERE/'run_rescue_prefix.py',
             HERE/'protocol_rescue_interactions.md', ROOT/'run_original_closeout.py')
    audit = dict(configurations=records, replayed_scores=len(predictions), holdout_parsed_or_evaluated=0,
                 source_hashes={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths})
    with (out/'audit.json').open('x', encoding='utf-8') as file:
        json.dump(audit, file, indent=2)
    with (out/'predictions.csv').open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == '__main__':
    main()

