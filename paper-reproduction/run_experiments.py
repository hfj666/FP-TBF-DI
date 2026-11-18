"""Reproducible static cross-validation and simulated cumulative arrivals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import platform
import sys
import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.fptree import FPTree, pattern_rules
from src.models import StaticBiclusterBoost, SparseSugeno

ROOT = Path(__file__).resolve().parent
FEATURES = ['Shape', 'Orientation', 'Margin integrality', 'Margin ambiguity', 'Angular',
            'Microlobulated', 'Spiculated', 'Echo pattern', 'Posterior feature',
            'Calcification in mass', 'Architectural distortion', 'Ducts changes',
            'Skin thickening', 'Edema', 'Vascularity', 'Cyst type',
            'Lymph nodes axillary', 'Vascular abnormalities']


def metrics(y, scores):
    prediction = np.where(scores >= 0, 1, -1)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[-1, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, prediction)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                f1=float(f1_score(y, prediction)), auc=float(roc_auc_score(y, scores)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def write_csv(path, records):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader()
        writer.writerows(records)


def fuzzy_from_tree(tree, x, y, args):
    patterns = tree.mine(args.support, args.max_length)
    rules = pattern_rules(patterns, args.max_rules)
    model = SparseSugeno(rules, maxiter=500).fit(x, y)
    return model, patterns


def cross_validate(x, y, args, output):
    records, predictions, all_scores = [], [], {}
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(x, y), 1):
        estimators = {
            'paper_bicluster_adaboost': StaticBiclusterBoost(),
            'logistic_regression': make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=2000)),
            'rbf_svm': make_pipeline(MinMaxScaler(), SVC(C=1., gamma='scale')),
        }
        for name in [*estimators, 'fp_sparse_sugeno']:
            start = perf_counter()
            if name == 'fp_sparse_sugeno':
                tree = FPTree().update(x[train], y[train])
                model, patterns = fuzzy_from_tree(tree, x[train], y[train], args)
                extra = dict(rules=len(model.rules), patterns=len(patterns),
                             optimizer_success=model.optimization['success'],
                             optimizer_iterations=model.optimization['iterations'])
            else:
                model = estimators[name].fit(x[train], y[train])
                extra = dict(rules=len(model.rules), pairs=len(model.pairs),
                             boosting_rounds=len(model.selected)) if hasattr(model, 'rules') else {}
            fit_seconds = perf_counter()-start
            scores = model.decision_function(x[test])
            all_scores.setdefault(name, np.zeros(len(y)))[test] = scores
            records.append(dict(model=name, fold=fold, train_n=len(train), test_n=len(test),
                                fit_seconds=fit_seconds, **metrics(y[test], scores), **extra))
            predictions.extend(dict(model=name, fold=fold, row_index=int(i), label=int(y[i]),
                                    score=float(score)) for i, score in zip(test, scores))
        print(f'CV fold {fold}/10 completed', flush=True)
    write_csv(output/'cv_folds.csv', records)
    write_csv(output/'cv_predictions.csv', predictions)
    summary = {}
    for name, scores in all_scores.items():
        rows = [r for r in records if r['model'] == name]
        summary[name] = {
            'pooled_oof': metrics(y, scores),
            'fold_mean': {m: float(np.mean([r[m] for r in rows])) for m in
                          ('accuracy', 'sensitivity', 'specificity', 'f1', 'auc', 'fit_seconds')},
            'fold_std': {m: float(np.std([r[m] for r in rows], ddof=1)) for m in
                         ('accuracy', 'sensitivity', 'specificity', 'f1', 'auc')},
        }
        # AUC pooled across independently fitted models need not share a score scale;
        # mean per-fold AUC is the primary discrimination estimate.
    (output/'cv_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def incremental(x, y, args, output):
    indices = np.arange(len(y))
    stream, test = train_test_split(indices, test_size=.2, stratify=y, random_state=args.seed)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(stream)
    tree = FPTree()
    stages, previous, initial_model, initial_static = [], 0, None, None
    checkpoints = list(range(300, len(stream), 100))+[len(stream)]
    for stage, end in enumerate(checkpoints):
        accumulated = stream[:end]
        new = stream[previous:end]
        start = perf_counter()
        tree.update(x[new], y[new])
        insertion_seconds = perf_counter()-start
        start = perf_counter()
        patterns = tree.mine(args.support, args.max_length)
        mining_seconds = perf_counter()-start
        start = perf_counter()
        rebuilt = FPTree().update(x[accumulated], y[accumulated])
        reference = rebuilt.mine(args.support, args.max_length)
        rebuild_seconds = perf_counter()-start
        if patterns != reference:
            raise AssertionError('Incremental and rebuilt patterns differ.')
        start = perf_counter()
        rules = pattern_rules(patterns, args.max_rules)
        model = SparseSugeno(rules, maxiter=500).fit(x[accumulated], y[accumulated])
        fuzzy_seconds = perf_counter()-start
        static = StaticBiclusterBoost().fit(x[accumulated], y[accumulated])
        if initial_model is None:
            initial_model, initial_static = model, static
        stages.append(dict(stage=stage, train_n=end, added_n=end-previous, test_n=len(test),
                           tree_nodes=tree.nodes, patterns=len(patterns), rules=len(rules),
                           patterns_exactly_equal=True, insertion_seconds=insertion_seconds,
                           mining_seconds=mining_seconds, incremental_seconds=insertion_seconds+mining_seconds,
                           rebuild_seconds=rebuild_seconds, fuzzy_refit_seconds=fuzzy_seconds,
                           speedup=rebuild_seconds/(insertion_seconds+mining_seconds),
                           optimizer=model.optimization,
                           updated_fuzzy=metrics(y[test], model.decision_function(x[test])),
                           frozen_fuzzy=metrics(y[test], initial_model.decision_function(x[test])),
                           updated_static=metrics(y[test], static.decision_function(x[test])),
                           frozen_static=metrics(y[test], initial_static.decision_function(x[test]))))
        previous = end
        print(f'Incremental stage {stage}: n={end}; {len(patterns)} patterns; exact match', flush=True)
    (output/'incremental.json').write_text(json.dumps(stages, indent=2), encoding='utf-8')
    write_csv(output/'incremental.csv', [dict(stage=s['stage'], train_n=s['train_n'],
              patterns=s['patterns'], nodes=s['tree_nodes'], insertion_seconds=s['insertion_seconds'],
              mining_seconds=s['mining_seconds'], rebuild_seconds=s['rebuild_seconds'], speedup=s['speedup'],
              fuzzy_accuracy=s['updated_fuzzy']['accuracy'], frozen_accuracy=s['frozen_fuzzy']['accuracy'],
              static_accuracy=s['updated_static']['accuracy']) for s in stages])
    (output/'stream_split.json').write_text(json.dumps({'train_stream_indices': stream.tolist(),
              'held_out_indices': test.tolist(), 'index_origin': 'zero-based row index in source TXT'}), encoding='utf-8')
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'cv', 'incremental'], default='all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--support', type=float, default=.1)
    parser.add_argument('--max-length', type=int, default=3)
    parser.add_argument('--max-rules', type=int, default=64)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'seed42')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    datafile = ROOT/'data'/'dataset_1062.txt'
    a = np.loadtxt(datafile)
    if a.shape != (1062, 19) or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError('Unexpected input structure or noninteger scores.')
    x, y = a[:, 1:], a[:, 0].astype(int)
    if set(y) != {-1, 1}:
        raise ValueError('Labels must be -1/+1.')
    metadata = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    sha256=hashlib.sha256(datafile.read_bytes()).hexdigest(),
                    shape=list(a.shape), labels={str(v): int((y == v).sum()) for v in [-1, 1]},
                    features=FEATURES, unique_rows=len(np.unique(a, axis=0)),
                    versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                  sklearn=sklearn.__version__, platform=platform.platform()),
                    limitations=['No patient IDs: lesion-level split only; same-patient leakage cannot be excluded.',
                                 'No timestamps: arrivals simulated, not prospective.',
                                 'Paper thresholds/linkage unreported; defaults explicitly chosen.',
                                 'Fuzzy model is sparse zero-order Sugeno, not a full ANFIS reproduction.',
                                 'FP mining exact only within configured max_length; no external validation dataset.',
                                 'Fuzzy model refitted on accumulated training cases; not end-to-end online training.'])
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if args.mode in ('all', 'cv'):
        cross_validate(x, y, args, output)
    if args.mode in ('all', 'incremental'):
        incremental(x, y, args, output)


if __name__ == '__main__':
    main()
