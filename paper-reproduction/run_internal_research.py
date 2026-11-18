"""Run the current internal FP study on the approved cohort streams."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from run_experiments import ROOT, metrics, write_csv
from run_original_closeout import load_stream
from src import internal_research as research
from src.fp_online import rule_records
from src.fptree import FPTree


SPLITS = ROOT/'results/online_study/splits.json'
INITIAL = 300
STREAM_SIZE = 849
SEEDS = (17, 29, 43)
BATCHES = (50, 100)


def hash_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    splits = json.loads(SPLITS.read_text(encoding='utf8'))
    if sorted(s['seed'] for s in splits) != list(SEEDS):
        raise ValueError('Unexpected internal research split seeds.')
    source_files = (
        'run_internal_research.py', 'run_original_closeout.py', 'run_experiments.py',
        'src/internal_research.py', 'src/projected_selection.py', 'src/models.py',
        'src/selection.py', 'src/fptree.py', 'src/fp_online.py',
    )
    protocol = dict(default_arm=research.DEFAULT_ARM, control_arm=research.CONTROL_ARM,
        component_arms=research.MODEL_ARMS, fusion_weights=[.5, .5],
        seeds=SEEDS, batches=BATCHES, initial=INITIAL, stream_rows=STREAM_SIZE,
        support=research.SUPPORT, max_length=research.MAX_LENGTH, strict_purity=research.PURITY,
        min_rows=research.MIN_ROWS, rules_per_class=research.RULES_PER_CLASS,
        coverage_alpha=research.COVERAGE_ALPHA, sugeno_maxiter=research.FIT_MAXITER,
        initialization='cold', scale_fit_rows=INITIAL, threshold=0, positive_label=1,
        evaluation='Predict next full batch from revealed prefix only. Skip unused terminal fit.',
        split_sha256=hashlib.sha256(SPLITS.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')

    predictions, summaries, fits, inputs = [], [], [], []
    for split in splits:
        seed = split['seed']
        ids = np.array(split['train_stream'])
        excluded = set(split['holdout'])
        if (len(ids) != STREAM_SIZE or len(set(ids)) != STREAM_SIZE
                or len(excluded) != 213 or set(ids) & excluded):
            raise ValueError(f'Invalid approved stream for seed {seed}.')
        x, y = load_stream(ROOT/'data/dataset_1062.txt', ids)
        inputs.append(dict(seed=seed, row_ids_sha256=hash_array(ids),
                           x_sha256=hash_array(x), y_sha256=hash_array(y), excluded_rows_parsed=0))
        scaling = x[:INITIAL].min(0), np.maximum(np.ptp(x[:INITIAL], axis=0), 1.)
        initial_tree = FPTree().update(x[:INITIAL], y[:INITIAL])
        cache = {}
        for batch in BATCHES:
            tree = copy.deepcopy(initial_tree)
            scores = {arm:[] for arm in research.SCORE_ARMS}
            for stage, start in enumerate(range(INITIAL, STREAM_SIZE, batch), 1):
                end = min(start+batch, STREAM_SIZE)
                patterns = tree.mine(research.SUPPORT, research.MAX_LENGTH)
                if start not in cache:
                    models, stats = research.fit_pair(patterns, x[:start], y[:start], scaling)
                    cache[start] = models, patterns
                    state = dict(seed=seed, prefix=start, ids_sha256=hash_array(ids[:start]),
                                 x_sha256=hash_array(x[:start]), y_sha256=hash_array(y[:start]), models={})
                    for arm, model in models.items():
                        fit = stats[arm].copy()
                        selection = fit.pop('selection')
                        fits.append(dict(seed=seed, prefix=start, arm=arm, **fit))
                        state['models'][arm] = dict(theta=model.theta.tolist(), rules=rule_records(model.rules),
                            minimum=model.minimum.tolist(), span=model.span.tolist(), selection=selection)
                    with gzip.open(out/f'state_s{seed}_n{start}.json.gz', 'wt', encoding='utf8') as f:
                        json.dump(state, f)
                models, saved_patterns = cache[start]
                if patterns != saved_patterns:
                    raise ValueError('FP candidates differ for the same revealed prefix.')
                next_scores = research.score_arms(models, x[start:end])
                for arm, values in next_scores.items():
                    scores[arm].extend(values)
                    predictions.extend(dict(seed=seed, batch_size=batch, stage=stage, arm=arm,
                        row_index=int(row), label=int(label), score=float(value))
                        for row, label, value in zip(ids[start:end], y[start:end], values, strict=True))
                if end < STREAM_SIZE:
                    tree.update(x[start:end], y[start:end])
            for arm, values in scores.items():
                values = metrics(y[INITIAL:], np.array(values))
                summaries.append(dict(seed=seed, batch_size=batch, arm=arm, **values,
                                      ba=(values['sensitivity']+values['specificity'])/2))
            print(f'completed seed={seed} batch={batch}', flush=True)

    for name, rows in [('predictions', predictions), ('summary', summaries), ('fits', fits)]:
        write_csv(out/f'{name}.csv', rows)
    (out/'inputs.json').write_text(json.dumps(inputs, indent=2), encoding='utf8')
    completed = all(r['success'] for r in fits)
    (out/'completion.json').write_text(json.dumps(dict(completed=completed, predictions=len(predictions),
        fits=len(fits), excluded_rows_parsed=0), indent=2), encoding='utf8')
    if not completed:
        raise RuntimeError('One or more cold fits did not terminate normally. See fits.csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/internal_research_v2',
                        help='new output directory')
    run(parser.parse_args().out)
