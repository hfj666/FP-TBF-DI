"""Evaluate the Huang-inspired implementation on the matched stream events."""
import argparse
import csv
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from run_original_closeout import load_stream
from run_paper_evidence_extension import OUT, ROOT, aggregate, csv_write, dump
from src.models import StaticBiclusterBoost


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        yield from csv.DictReader(file)


def archived_scores(archived):
    scores = {}
    for row in rows(archived):
        if row['policy'] != 'static_draft_cumulative':
            continue
        key = tuple(int(row[k]) for k in ('seed', 'batch_size', 'row_index'))
        assert key not in scores
        scores[key] = (int(row['label']), int(row['stage']), float(row['score']))
    return scores


def refit_scores():
    splits = json.loads((ROOT/'results/online_study/splits.json').read_text(encoding='utf-8'))
    scores = {}
    with threadpool_limits(limits=1):
        for split in splits:
            seed, ids = split['seed'], split['train_stream']
            x, y = load_stream(ROOT/'data/dataset_1062.txt', ids)
            by_prefix = {}
            for batch in (50, 100):
                for stage, start in enumerate(range(300, len(ids), batch), 1):
                    stop = min(start + batch, len(ids))
                    if start not in by_prefix:
                        model = StaticBiclusterBoost().fit(x[:start], y[:start])
                        by_prefix[start] = model.decision_function(x[start:min(start+100, len(ids))])
                    for index, label, value in zip(ids[start:stop], y[start:stop],
                                                   by_prefix[start][:stop-start], strict=True):
                        scores[seed, batch, index] = (int(label), stage, float(value))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use previously scored direct comparisons.')
    args = parser.parse_args()
    target = ROOT / 'results/paper_parsimony_cost'
    scores = archived_scores(args.archive) if args.archive else refit_scores()
    paired = []
    for row in rows(OUT / 'matched_predictions.csv'):
        if row['evaluation'] != 'stream':
            continue
        seed, batch, index = (int(row[k]) for k in ('seed', 'batch', 'row_index'))
        label, stage, score = scores.pop((seed, batch, index))
        assert label == int(row['label']) and int(row['training_n']) == 300 + (stage - 1) * batch
        paired.append(dict(seed=seed, batch=batch, row_index=index, label=label,
                           fp_tbf_di=float(row['fp_tbf_di']), huang_inspired=score))
    assert len(paired) == 3294 and not scores
    target.mkdir(parents=True, exist_ok=True)
    csv_write(target / 'huang_alignment.csv', paired)
    dump(target / 'huang_alignment.json', aggregate(paired, ('fp_tbf_di', 'huang_inspired'), ('seed', 'batch')))


if __name__ == '__main__':
    main()
