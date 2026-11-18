"""Check whether stable evidence survives deletion of each score feature."""
import csv
import json

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_rescue_prefix import HERE, ROOT, comparison, digest, fit_score, load_stream, metrics, read_policy


def predict(history, y, query):
    result = {}
    for name, encoder in (('numeric_deletion', StandardScaler()),
                          ('categorical_deletion', OneHotEncoder(handle_unknown='ignore'))):
        train = encoder.fit_transform(history)
        test = encoder.transform(query)
        widths = [1] * history.shape[1] if name == 'numeric_deletion' else [
            len(categories) for categories in encoder.categories_]
        borders = np.cumsum([0, *widths])
        full = fit_score(train, y, test)
        stable = full >= 0
        for first, last in zip(borders[:-1], borders[1:], strict=True):
            keep = (np.arange(train.shape[1]) < first) | (np.arange(train.shape[1]) >= last)
            stable &= fit_score(train[:, keep], y, test[:, keep]) >= 0
        result[name] = (full, stable)
    return result


def main():
    previous = HERE/'experiment_rescue_prefix/predictions.csv'
    base_path = HERE/'experiment_hard_width_bank1701/predictions.csv'
    split_path = ROOT/'results/online_study/splits.json'
    manifest = base_path.with_name('manifest.json')
    approved = {r['seed']: r for r in json.loads(manifest.read_text())['inputs']}
    reference_rows = read_policy(previous, 'unanimous')
    base_rows = read_policy(base_path, 'baseline')
    configs, rows = [], []
    for split in json.loads(split_path.read_text()):
        seed, ids = split['seed'], split['train_stream']
        if seed not in (17, 29, 43):
            continue
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(split['holdout'])
        raw, labels = load_stream(ROOT/'data/dataset_1062.txt', ids)
        assert digest(raw.tobytes()) == approved[seed]['allowed_x_sha256']
        assert digest(labels.tobytes()) == approved[seed]['allowed_y_sha256']
        y = labels[300:]
        for batch in (50, 100):
            matched = []
            for source in (reference_rows, base_rows):
                subset = [r for r in source if (int(r['seed']), int(r['batch_size'])) == (seed, batch)]
                assert [int(r['row_index']) for r in subset] == ids[300:]
                np.testing.assert_array_equal([int(r['label']) for r in subset], y)
                np.testing.assert_array_equal([int(r['stage']) for r in subset], np.arange(549)//batch+1)
                matched.append(np.array([float(r['score']) for r in subset]))
            reference, baseline = matched
            candidates = {name: reference.copy() for name in ('numeric_deletion', 'categorical_deletion')}
            for start in range(300, len(raw), batch):
                stop = min(start+batch, len(raw))
                history, past_labels = raw[:start], labels[:start]
                negative = {tuple(row) for row in history[past_labels == -1]}
                indices = [i for i in range(start, stop) if reference[i-300] < 0
                           and tuple(raw[i]) not in negative]
                if not indices:
                    continue
                altered = labels.copy()
                altered[start:] *= -1
                estimated = predict(history, past_labels, raw[indices])
                replay = predict(history, altered[:start], raw[indices])
                for name, (score, stable) in estimated.items():
                    np.testing.assert_array_equal(score, replay[name][0])
                    np.testing.assert_array_equal(stable, replay[name][1])
                    candidates[name][np.array(indices)[stable]-300] = score[stable]
            record = dict(seed=seed, batch_size=batch, reference=metrics(y, reference), methods={})
            for name, proposed in candidates.items():
                np.testing.assert_array_equal(proposed[reference >= 0], reference[reference >= 0])
                record['methods'][name] = dict(metrics=metrics(y, proposed),
                    vs_champion=comparison(y, reference, proposed),
                    vs_baseline=comparison(y, baseline, proposed))
                rows.extend(dict(seed=seed, batch_size=batch, stage=i//batch+1, row_index=rid,
                    label=int(label), policy=name, score=score)
                    for i, (rid, label, score) in enumerate(zip(ids[300:], y, proposed, strict=True)))
            configs.append(record)
            print(seed, batch, [v['vs_champion'] for v in record['methods'].values()], flush=True)
    assert len(configs) == 6 and len(rows) == 6588
    out = HERE/'experiment_rescue_feature_deletion'
    out.mkdir(exist_ok=False)
    paths = (previous, base_path, manifest, split_path, HERE/'run_rescue_prefix.py',
             HERE/'run_rescue_feature_deletion.py', HERE/'protocol_rescue_feature_deletion.md',
             HERE/'run_prequential_evidence.py', ROOT/'run_original_closeout.py')
    audit = dict(configurations=configs, replayed_scores=len(rows), holdout_parsed_or_evaluated=0,
                 source_hashes={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths})
    (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    with (out/'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
