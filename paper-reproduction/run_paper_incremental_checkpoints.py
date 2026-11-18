"""Report frozen-model metrics as more labeled samples enter the FP tree."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from run_three_way_retrospective_holdout import FPTree, load_stream, read_holdout, score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/paper_incremental_checkpoints'
SIZES = (300, 500, 700, 849)


def grouped_scores(path, method=None):
    groups = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if method is None or row['method'] == method:
                groups[int(row['seed']), int(row['batch_size'])].append(row)
    return groups


def metrics(labels, scores):
    pred = np.where(scores >= 0, 1, -1)
    tn, fp, fn, tp = map(int, confusion_matrix(labels, pred, labels=[-1, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return dict(n=len(labels), tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=(tp+tn)/len(labels), precision=precision, recall=recall,
                specificity=specificity, f1=2*tp/(2*tp+fp+fn),
                balanced_accuracy=(recall+specificity)/2,
                roc_auc=roc_auc_score(labels, scores),
                average_precision=average_precision_score(labels, scores))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    split_path = ROOT / 'results/online_study/splits.json'
    data = ROOT / 'data/dataset_1062.txt'
    archive = grouped_scores(
        ROOT / 'parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv',
        'numeric_stable')
    reference = grouped_scores(ROOT / 'results/three_way_retrospective_holdout/predictions.csv')
    stream_rows, checkpoint_rows = [], []
    all_y, all_scores = [], []
    for split in json.loads(split_path.read_text(encoding='utf-8')):
        seed, ids, excluded = (split[key] for key in ('seed', 'train_stream', 'holdout'))
        assert len(set(ids)) == 849 and len(set(excluded)) == 213 and not set(ids) & set(excluded)
        x, y = load_stream(data, ids)
        query, labels = read_holdout(data, excluded)
        tree, start = FPTree(), 0
        histories = {}
        for batch in (50, 100):
            saved = archive[seed, batch]
            np.testing.assert_array_equal([int(r['row_index']) for r in saved], ids[300:])
            np.testing.assert_array_equal([int(r['label']) for r in saved], y[300:])
            histories[batch] = np.array([float(r['coverage']) for r in saved])
            fused = np.array([float(r['new_ensemble']) for r in saved])
            stream_rows.append(dict(seed=seed, batch_size=batch, **metrics(y[300:], fused)))
            all_y.extend(y[300:])
            all_scores.extend(fused)
        for end in SIZES:
            tree.update(x[start:end], y[start:end])
            assert tree.mine(.1, 3) == FPTree().update(x[:end], y[:end]).mine(.1, 3)
            for batch, history in histories.items():
                output = score(x[:end], y[:end], query, x[300:end], y[300:end],
                               history[:end-300], tree=tree)
                if end == 849:
                    archived = reference[seed, batch]
                    np.testing.assert_array_equal([int(r['row_index']) for r in archived], excluded)
                    columns = ('coverage', 'logistic', 'projection', 'hvdm_offset',
                               'new_branch', 'new_ensemble')
                    expected = [[float(r[k]) for k in columns] for r in archived]
                    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-7)
                checkpoint_rows.append(dict(seed=seed, batch_size=batch, training_n=end,
                                            added_n=end-start, **metrics(labels, output[:, -1])))
            start = end
            print(f'seed={seed} training_n={end}: both histories evaluated', flush=True)
    assert len(stream_rows) == 6 and len(checkpoint_rows) == 24
    overall = metrics(np.array(all_y), np.array(all_scores))
    assert overall['n'] == 3294 and overall['tp'] + overall['tn'] == 3175
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv('stream_metrics.csv', stream_rows)
    write_csv('checkpoint_metrics.csv', checkpoint_rows)
    (OUT / 'stream_pooled.json').write_text(json.dumps(overall, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
