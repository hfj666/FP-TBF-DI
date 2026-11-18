"""Time the prespecified A+LR and FP-TBF-DI in paired single-thread runs."""
import csv
import json
import os
import platform
from time import perf_counter

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from run_paper_evidence_extension import ROOT, dump
from src.fp_tbf_simplified import SimplifiedFPTBF
from src.fptree import FPTree
from src.models import SparseSugeno
from src.projected_selection import select
from src.reference_comparators import CumulativeLogistic
from src.selection import CoveragePool


class ALR:
    """The exact A and uncorrected logistic branches, with no other fits."""

    def fit(self, x, y, tree):
        scale = x[:300].min(0), np.maximum(np.ptp(x[:300], axis=0), 1.)
        pool = CoveragePool(tree.mine(.1, 3), x, y)
        rules, _ = select(pool, x, y, scale, joint=True)
        self.a = SparseSugeno(rules, maxiter=2000).fit(x, y, scaling=scale)
        self.lr = CumulativeLogistic(x, y, scale)
        return self

    def decision_function(self, query):
        return (self.a.decision_function(query) + self.lr.decision_function(query)) / 2


def main():
    prior = ROOT / 'results/paper_parsimony_cost'
    target = prior / 'a_lr_timing.json'
    if target.exists():
        raise FileExistsError(target)
    with (prior / 'matched.csv').open(encoding='utf-8-sig', newline='') as file:
        expected = {
            (r['evaluation'], int(r['seed']), int(r['batch']), int(r['row_index'])):
            (float(r['fp_tbf_di']), float(r['a_lr'])) for r in csv.DictReader(file)
        }
    data = np.loadtxt(ROOT / 'data/dataset_1062.txt')
    splits = json.loads((ROOT / 'results/online_study/splits.json').read_text())
    models = (SimplifiedFPTBF, ALR)
    warm = data[splits[0]['train_stream'][:300]]
    for model in models:
        labels = warm[:, 0].astype(int)
        model().fit(warm[:, 1:], labels, FPTree().update(warm[:, 1:], labels)).decision_function(warm[:50, 1:])
    schedules = []
    for split in splits:
        ids, test = split['train_stream'], split['holdout']
        x, y = data[ids, 1:], data[ids, 0].astype(int)
        for batch in (50, 100):
            order = (0, 1) if len(schedules) % 2 == 0 else (1, 0)
            profiles = {}
            for arm in order:
                phases = []
                tree, previous = FPTree(), 0
                for n in [*range(300, 849, batch), 849]:
                    evaluation = 'stream' if n < 849 else 'holdout_849'
                    query_ids = ids[n:min(n + batch, 849)] if n < 849 else test
                    query = data[query_ids, 1:]
                    start = perf_counter()
                    tree.update(x[previous:n], y[previous:n])
                    after_tree = perf_counter()
                    model = models[arm]().fit(x[:n], y[:n], tree)
                    after_fit = perf_counter()
                    score = model.decision_function(query)
                    after_predict = perf_counter()
                    reference = np.asarray([expected[evaluation, split['seed'], batch, rid][arm]
                                            for rid in query_ids])
                    np.testing.assert_allclose(score, reference, atol=1e-7, rtol=0)
                    np.testing.assert_array_equal(score >= 0, reference >= 0)
                    phases.append(dict(n=n, events=len(query_ids), max_error=float(np.max(abs(score-reference))),
                                       tree=after_tree-start, fit=after_fit-after_tree,
                                       predict=after_predict-after_fit))
                    previous = n
                profiles[models[arm].__name__] = phases
            schedules.append(dict(seed=split['seed'], batch=batch, order=order, profiles=profiles))
            print(f"Verified both models for seed {split['seed']}, batch {batch}", flush=True)
    dump(target, dict(platform=platform.platform(), processor=platform.processor(),
                      python=platform.python_version(), schedules=schedules))


if __name__ == '__main__':
    main()
