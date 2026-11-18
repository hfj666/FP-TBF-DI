"""Regenerate the manuscript's historical stream scores from labeled rows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_three_way_retrospective_holdout import ROBUST, ROOT, FPTree, load_stream, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/dataset_1062.txt')
    parser.add_argument('--splits', type=Path, default=ROOT/'results/online_study/splits.json')
    parser.add_argument('--out', type=Path, default=ROBUST/'experiment_rescue_interactions/predictions.csv')
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text(encoding='utf-8'))
    rows = []
    with threadpool_limits(limits=1):
        for split in splits:
            seed, ids, holdout = (split[key] for key in ('seed', 'train_stream', 'holdout'))
            if len(ids) != 849 or len(holdout) != 213 or len(set(ids + holdout)) != 1062:
                raise ValueError(f'Expected disjoint study splits for seed {seed}.')
            x, y = load_stream(args.data, ids)
            for batch in (50, 100):
                tree = FPTree()
                previous = 0
                base_history = np.empty(len(ids) - 300)
                for start in range(300, len(ids), batch):
                    stop = min(start + batch, len(ids))
                    tree.update(x[previous:start], y[previous:start])
                    past = base_history[:start-300]
                    predictions = score(x[:start], y[:start], x[start:stop],
                                        x[300:start], y[300:start], past, tree=tree)
                    base_history[start-300:stop-300] = predictions[:, 0]
                    for index, label, values in zip(ids[start:stop], y[start:stop], predictions, strict=True):
                        rows.append((seed, batch, (start-300)//batch+1, index, label,
                                     'numeric_stable', *values))
                    previous = start
                print(f'Regenerated seed={seed} batch={batch}', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('seed', 'batch_size', 'stage', 'row_index', 'label', 'method',
                         'coverage', 'logistic', 'projection', 'hvdm_offset', 'new_branch', 'new_ensemble'))
        writer.writerows(rows)


if __name__ == '__main__':
    main()
