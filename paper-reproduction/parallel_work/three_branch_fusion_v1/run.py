"""Fixed equal-score fusion of three archived research branches."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_experiments import metrics, write_csv

HERE = Path(__file__).resolve().parent
SOURCES = {
    'coverage': ('results/internal_research_v2/predictions.csv', 'coverage_saved'),
    'current_two': ('results/internal_research_v2/predictions.csv', 'equal_score_mean'),
    'projection': ('results/internal_research_v2/predictions.csv', 'soft_projected'),
    'hvdm_offset': ('parallel_work/fuzzy_activation/results_13/predictions.csv', 'logistic_offset'),
    'rescue': ('parallel_work/robust_learning/experiment_rescue_feature_deletion/predictions.csv', 'categorical_deletion'),
    'logistic': ('results/direct_comparison_v2/predictions.csv', 'logistic_cumulative'),
}
COMPONENTS = ('projection', 'hvdm_offset', 'rescue')


def read_scores(path, policy):
    rows = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('policy', row.get('arm')) != policy:
                continue
            key = tuple(int(row[column]) for column in ('seed', 'batch_size', 'stage', 'row_index'))
            assert key not in rows
            rows[key] = int(row['label']), float(row['score'])
    return rows


def main():
    out = HERE/'results'
    out.mkdir(exist_ok=False)
    split_path = ROOT/'results/online_study/splits.json'
    paths = {p for p, _ in SOURCES.values()} | {str(split_path.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT))}
    protocol = {
        'components': COMPONENTS,
        'sources': SOURCES,
        'rule': 'Arithmetic mean of three archived raw scores; threshold zero. Fixed weights, no calibration or label-dependent choice. Output ensemble, no joint refit.',
        'scope': 'Reuse archived predictions only. Match authorized suffix row, seed, batch, stage and label. Overlapping internal development configurations.',
        'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)},
    }
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    saved = {name: read_scores(ROOT/path, policy) for name, (path, policy) in SOURCES.items()}
    assert all(len(rows) == 3294 for rows in saved.values())
    summary, predictions, paired = [], [], []
    for split in json.loads(split_path.read_text()):
        seed = split['seed']
        ids = split['train_stream'][300:]
        assert len(ids) == len(set(ids)) == 549 and not set(ids) & set(split['holdout'])
        for batch in (50, 100):
            keys = [(seed, batch, i//batch+1, rid) for i, rid in enumerate(ids)]
            y = np.array([saved['coverage'][key][0] for key in keys])
            scores = {}
            for name, rows in saved.items():
                assert {key for key in rows if key[:2] == (seed, batch)} == set(keys)
                np.testing.assert_array_equal([rows[key][0] for key in keys], y)
                scores[name] = np.array([rows[key][1] for key in keys])
                assert np.isfinite(scores[name]).all()
            np.testing.assert_allclose(scores['current_two'], (scores['coverage']+scores['projection'])/2, atol=1e-12)
            scores['three_way'] = np.mean([scores[name] for name in COMPONENTS], axis=0)
            for name, score in scores.items():
                result = metrics(y, score)
                result['ba'] = (result['sensitivity']+result['specificity'])/2
                summary.append(dict(seed=seed, batch_size=batch, policy=name, **result))
            for i, (_, _, stage, row_index) in enumerate(keys):
                predictions.append(dict(seed=seed, batch_size=batch, stage=stage, row_index=row_index,
                    label=int(y[i]), **{name: float(score[i]) for name, score in scores.items()}))
            new_positive = scores['three_way'] >= 0
            for name, score in scores.items():
                if name == 'three_way':
                    continue
                old_positive = score >= 0
                turned_positive = new_positive & ~old_positive
                turned_negative = old_positive & ~new_positive
                paired.append(dict(seed=seed, batch_size=batch, control=name,
                    repaired_fn=int((turned_positive & (y == 1)).sum()), new_fn=int((turned_negative & (y == 1)).sum()),
                    repaired_fp=int((turned_negative & (y == -1)).sum()), new_fp=int((turned_positive & (y == -1)).sum())))
    assert len(predictions) == 3294
    write_csv(out/'summary.csv', summary)
    write_csv(out/'matched_predictions.csv', predictions)
    write_csv(out/'paired.csv', paired)
    fields = ('accuracy', 'ba', 'sensitivity', 'specificity', 'auc')
    aggregate = {name: {field: float(np.mean([row[field] for row in summary if row['policy'] == name]))
                       for field in fields} for name in (*SOURCES, 'three_way')}
    (out/'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
