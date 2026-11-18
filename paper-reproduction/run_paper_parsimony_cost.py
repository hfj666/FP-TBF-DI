"""Prespecified follow-up using the existing splits and fixed model settings.

Compare (A + raw LR) / 2 with full fusion on every saved matched event and
the original five vector-grouped folds. No tuning or model selection.
Time the full classifier with append versus rebuild on all six stream
schedules plus their final holdout. Alternate run order across schedules.
Report fit and query time separately, and verify scores against each other
and the archived predictions. Timing measures this machine only.
"""

import csv
import argparse
import json
import os
import platform
from time import perf_counter

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from run_paper_evidence_extension import ROOT, OUT, aggregate, csv_write, dump
from src.fp_tbf_simplified import SimplifiedFPTBF
from src.fptree import FPTree


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    timing_mode = parser.add_mutually_exclusive_group()
    timing_mode.add_argument('--repeat-timing', action='store_true',
                             help='Run the separate three-repeat timing audit for split 17.')
    timing_mode.add_argument('--current-timing', action='store_true',
                             help='Audit the current implementation on all six configurations.')
    args = parser.parse_args()
    if args.repeat_timing or args.current_timing:
        data = np.loadtxt(ROOT / 'data/dataset_1062.txt')
        splits = json.loads((ROOT / 'results/online_study/splits.json').read_text())
        old = load_csv(ROOT / 'results/paper_c_joint_reduction/predictions.csv')
        expected = {(r['evaluation'], int(r['seed']), int(r['batch']), int(r['row_index'])):
                    float(r['simple_fusion']) for r in old}
        target = ROOT / 'results' / ('paper_current_timing' if args.current_timing else 'paper_repeated_timing')
        target.mkdir(exist_ok=args.current_timing)
        selected = splits if args.current_timing else [split for split in splits if split['seed'] == 17]
        timing_audit(data, selected, expected, target, repeats=3, resume=args.current_timing)
        return

    target = ROOT / 'results/paper_parsimony_cost'
    target.mkdir()
    data = np.loadtxt(ROOT / 'data/dataset_1062.txt')
    splits = json.loads((ROOT / 'results/online_study/splits.json').read_text())
    old = load_csv(ROOT / 'results/paper_c_joint_reduction/predictions.csv')
    expected = {}
    paired = []
    for row in old:
        evaluation = row['evaluation']
        seed, batch, index, label = (int(row[k]) for k in ('seed', 'batch', 'row_index', 'label'))
        fusion = float(row['simple_fusion'])
        expected[evaluation, seed, batch, index] = fusion
        paired.append(dict(evaluation=evaluation, seed=seed, batch=batch, row_index=index,
                           label=label, fp_tbf_di=fusion, a_lr=(float(row['A']) + float(row['logistic'])) / 2))
    arms = ('fp_tbf_di', 'a_lr')
    assert len(expected) == len(paired) == 8406
    matched = {ev: aggregate([r for r in paired if r['evaluation'] == ev], arms, ('seed', 'batch'))
               for ev in dict.fromkeys(r['evaluation'] for r in paired)}
    groups = []
    for fold in range(1, 6):
        saved = json.loads((OUT / f'folds/fold{fold}.json').read_text(encoding='utf-8'))
        train, test = data[saved['train']], data[saved['test']]
        model = SimplifiedFPTBF().fit(train[:, 1:], train[:, 0].astype(int))
        scores = model.decision_scores(test[:, 1:])
        np.testing.assert_allclose(scores[:, -1], saved['scores']['fp_tbf_di'], atol=1e-7, rtol=0)
        for index, row in zip(saved['test'], scores, strict=True):
            groups.append(dict(fold=fold, row_index=index, label=int(data[index, 0]),
                               fp_tbf_di=float(row[-1]), a_lr=float((row[1] + row[2]) / 2)))
    assert len({r['row_index'] for r in groups}) == len(groups) == 1062
    csv_write(target / 'matched.csv', paired)
    csv_write(target / 'grouped.csv', groups)
    dump(target / 'comparison.json', dict(matched=matched, grouped=aggregate(groups, arms, ('fold',))))
    print('Saved paired comparisons', flush=True)
    timing_audit(data, splits, expected, target)


def timing_audit(data, splits, expected, target, *, repeats=1, resume=False):
    warm = data[splits[0]['train_stream'][:300]]
    SimplifiedFPTBF().fit(warm[:, 1:], warm[:, 0].astype(int)).decision_function(data[:50, 1:])
    timings = json.loads((target / 'timing.json').read_text(encoding='utf-8'))['schedules'] if resume and (target / 'timing.json').exists() else []
    completed = {(row['repeat'], row['seed'], row['batch']) for row in timings}
    for repeat, split in ((repeat, split) for repeat in range(repeats) for split in splits):
        ids, test = split['train_stream'], split['holdout']
        x, y = data[ids, 1:], data[ids, 0].astype(int)
        for batch in (50, 100):
            if (repeat + 1, split['seed'], batch) in completed:
                continue
            order = ('append', 'rebuild') if (len(timings) + repeat) % 2 == 0 else ('rebuild', 'append')
            profiles = {}
            predictions = {}
            for mode in order:
                tree = FPTree()
                previous = 0
                phases = []
                predictions[mode] = []
                for n in [*range(300, 849, batch), 849]:
                    evaluation = 'stream' if n < 849 else 'holdout_849'
                    query_ids = ids[n:min(n + batch, 849)] if n < 849 else test
                    query = data[query_ids, 1:]
                    start = perf_counter()
                    if mode == 'rebuild':
                        tree = FPTree()
                        tree.update(x[:n], y[:n])
                    else:
                        tree.update(x[previous:n], y[previous:n])
                    after_tree = perf_counter()
                    fit_phases = {}
                    model = SimplifiedFPTBF().fit(x[:n], y[:n], tree, timing=fit_phases)
                    after_fit = perf_counter()
                    score = model.decision_function(query)
                    after_predict = perf_counter()
                    reference = [expected[evaluation, split['seed'], batch, index] for index in query_ids]
                    np.testing.assert_allclose(score, reference, atol=1e-7, rtol=0)
                    np.testing.assert_array_equal(score >= 0, np.asarray(reference) >= 0)
                    predictions[mode].extend(score.tolist())
                    phases.append(dict(n=n, tree=after_tree-start, fit=after_fit-after_tree,
                                       predict=after_predict-after_fit, **fit_phases))
                    previous = n
                profiles[mode] = phases
            np.testing.assert_allclose(predictions['append'], predictions['rebuild'], atol=1e-7, rtol=0)
            timings.append(dict(repeat=repeat + 1, seed=split['seed'], batch=batch,
                                order=order, profiles=profiles))
            dump(target / 'timing.json', dict(platform=platform.platform(), processor=platform.processor(),
                                              python=platform.python_version(), schedules=timings))
            print(f"Verified timings for seed {split['seed']}, batch {batch}", flush=True)


if __name__ == '__main__':
    main()
