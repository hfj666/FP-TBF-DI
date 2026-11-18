"""Align the archived Huang-inspired implementation with current stream events."""
import csv

from run_paper_evidence_extension import OUT, ROOT, aggregate, csv_write, dump


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        yield from csv.DictReader(file)


def main():
    target = ROOT / 'results/paper_parsimony_cost'
    archived = ROOT / 'results/direct_comparison_v2/predictions.csv'
    scores = {}
    for row in rows(archived):
        if row['policy'] != 'static_draft_cumulative':
            continue
        key = tuple(int(row[k]) for k in ('seed', 'batch_size', 'row_index'))
        assert key not in scores
        scores[key] = (int(row['label']), int(row['stage']), float(row['score']))
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
    csv_write(target / 'huang_alignment.csv', paired)
    dump(target / 'huang_alignment.json', aggregate(paired, ('fp_tbf_di', 'huang_inspired'), ('seed', 'batch')))


if __name__ == '__main__':
    main()
