"""Fixed-split factorial ablation, rule diagnostics, and repeated stage timings.

No test-based parameter selection. Original seed42 outputs remain untouched.
"""
import argparse
import cProfile
import hashlib
import json
import pstats
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.models import StaticBiclusterBoost, SparseSugeno
from src.fptree import FPTree, pattern_rules
from run_experiments import ROOT, FEATURES, metrics, write_csv

# Zero means absence for these particular scoring variables. For other columns,
# zero can mean oval, parallel, normal, etc.; it is NOT missing data.
ABSENCE_COLUMNS = {4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17}
PRIMARY = ('static_boost', 'fp_boost', 'static_sugeno', 'fp_sugeno')


def masks_and_distances(rules, x, minimum, span):
    exact = np.column_stack([np.all(np.isclose(x[:, list(r.columns)], r.values, atol=1e-9, rtol=0), axis=1)
                             for r in rules])
    distances = np.column_stack([np.sqrt(np.mean(((x[:, list(r.columns)]-r.values)/span[list(r.columns)])**2, axis=1))
                                 for r in rules])
    return exact, distances


def rule_diagnostics(rules, x, y, fold, source, minimum, span):
    exact, distances = masks_and_distances(rules, x, minimum, span)
    rule_rows = []
    for j, r in enumerate(rules):
        zeros = sum(abs(v) < 1e-9 for v in r.values)
        absent = sum(c in ABSENCE_COLUMNS and abs(v) < 1e-9 for c, v in zip(r.columns, r.values))
        rule_rows.append(dict(fold=fold, source=source, rule=j, label=r.label,
                    length=len(r.columns), mining_count=r.count, confidence=r.confidence,
                    exact_match_count=int(exact[:, j].sum()), zero_fraction=zeros/len(r.columns),
                    absent_fraction=absent/len(r.columns), all_zero=zeros==len(r.columns),
                    all_absent=absent==len(r.columns),
                    antecedent=' & '.join(f'F{c}:{FEATURES[c]}={v:g}' for c, v in zip(r.columns, r.values))))
    pairs, duplicate_coverage, nested, jaccards = 0, 0, 0, []
    for i, a in enumerate(rules):
        ai = set(zip(a.columns, np.round(a.values, 9)))
        for j in range(i+1, len(rules)):
            b = rules[j]
            if a.label != b.label:
                continue
            pairs += 1
            union = np.sum(exact[:, i] | exact[:, j])
            jac = float(np.sum(exact[:, i] & exact[:, j])/union) if union else 0.
            jaccards.append(jac)
            duplicate_coverage += int(jac >= .9)
            bj = set(zip(b.columns, np.round(b.values, 9)))
            nested += int(ai <= bj or bj <= ai)
    summary = dict(fold=fold, source=source, rules=len(rules),
                   benign_rules=sum(r.label == -1 for r in rules), malignant_rules=sum(r.label == 1 for r in rules),
                   mean_length=float(np.mean([r['length'] for r in rule_rows])),
                   mean_confidence=float(np.mean([r.confidence for r in rules])),
                   mean_zero_fraction=float(np.mean([r['zero_fraction'] for r in rule_rows])),
                   mean_absent_fraction=float(np.mean([r['absent_fraction'] for r in rule_rows])),
                   all_zero_rules=sum(r['all_zero'] for r in rule_rows),
                   all_absent_rules=sum(r['all_absent'] for r in rule_rows),
                   same_class_pairs=pairs, high_overlap_pairs=duplicate_coverage, nested_pairs=nested,
                   high_overlap_rate=duplicate_coverage/pairs if pairs else 0,
                   mean_jaccard=float(np.mean(jaccards)) if jaccards else 0.)
    for label, name in [(-1, 'benign'), (1, 'malignant')]:
        own = [j for j, r in enumerate(rules) if r.label == label]
        other = [j for j, r in enumerate(rules) if r.label != label]
        subset = y == label
        summary[name+'_own_exact_coverage'] = float(np.mean(exact[subset][:, own].any(1)))
        summary[name+'_opposite_exact_coverage'] = float(np.mean(exact[subset][:, other].any(1)))
        summary[name+'_own_near_coverage'] = float(np.mean((distances[subset][:, own] <= .1).any(1)))
    summary['both_classes_exact_coverage'] = float(np.mean(exact[:, [j for j,r in enumerate(rules) if r.label==-1]].any(1)
                 & exact[:, [j for j,r in enumerate(rules) if r.label==1]].any(1)))
    return summary, rule_rows


def run_ablation(x, y, out):
    folds, predictions, rule_summaries, rule_rows, errors, features = [], [], [], [], [], []
    scores_by_model = {}
    for fold, (train, test) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(x, y), 1):
        xt, yt = x[train], y[train]
        start = perf_counter()
        static = StaticBiclusterBoost().mine_rules(xt, yt)
        static_seconds = perf_counter()-start
        start = perf_counter()
        tree = FPTree().update(xt, yt)
        patterns = tree.mine(.1, 3)
        fp_rules = pattern_rules(patterns, 64)
        fp_seconds = perf_counter()-start
        static_rules = static.raw_rules()
        # Auxiliary control: match the number of rules separately in each class,
        # without referring to test labels or test performance.
        matched = []
        for label in (-1, 1):
            limit = sum(r.label == label for r in static_rules)
            matched.extend([r for r in fp_rules if r.label == label][:limit])
        sources = {'static': static_rules, 'fp': fp_rules, 'fp_budget': matched}
        for source, rules in sources.items():
            summary, records = rule_diagnostics(rules, xt, yt, fold, source, static.minimum, static.span)
            summary['split'] = 'train'
            rule_summaries.append(summary)
            test_summary, _ = rule_diagnostics(rules, x[test], y[test], fold, source, static.minimum, static.span)
            test_summary['split'] = 'test'
            rule_summaries.append(test_summary)
            rule_rows.extend(records)
            for label in (-1, 1):
                for j, name in enumerate(FEATURES):
                    use = [r for r in rules if r.label==label and j in r.columns]
                    features.append(dict(fold=fold, source=source, label=label, feature=j, name=name,
                                         rule_occurrences=len(use), zero_occurrences=sum(abs(r.values[r.columns.index(j)])<1e-9 for r in use)))
            for classifier in ('boost', 'sugeno'):
                name = source+'_'+classifier
                start = perf_counter()
                if classifier == 'boost':
                    model = StaticBiclusterBoost().fit_rules(xt, yt, rules)
                    extra = dict(weak_rounds=len(model.selected), candidates=len(model.pairs))
                else:
                    model = SparseSugeno(rules, maxiter=500).fit(xt, yt)
                    c = model.theta[len(rules):]
                    extra = dict(optimizer_success=model.optimization['success'],
                                 optimizer_iterations=model.optimization['iterations'],
                                 consequent_label_flips=int(sum(np.sign(c[i]) != r.label for i,r in enumerate(rules))))
                seconds = perf_counter()-start
                score = model.decision_function(x[test])
                scores_by_model.setdefault(name, np.zeros(len(y)))[test] = score
                folds.append(dict(model=name, fold=fold, rules=len(rules), classifier_seconds=seconds,
                                  mining_seconds=static_seconds if source=='static' else fp_seconds,
                                  **metrics(y[test], score), **extra))
                predictions.extend(dict(model=name, fold=fold, row_index=int(i), label=int(y[i]), score=float(s))
                                   for i,s in zip(test, score))
                exact, distance = masks_and_distances(rules, x[test], static.minimum, static.span)
                firing = model.firing_weights(x[test]) if classifier == 'sugeno' else None
                for local, index in enumerate(test):
                    predicted = 1 if score[local]>=0 else -1
                    if predicted == y[index]:
                        continue
                    own = [j for j,r in enumerate(rules) if r.label == y[index]]
                    opposite = [j for j,r in enumerate(rules) if r.label != y[index]]
                    row = dict(model=name, fold=fold, row_index=int(index), file_line=int(index+1),
                               label=int(y[index]), prediction=predicted,
                               error_type='FP' if y[index]==-1 else 'FN', score=float(score[local]),
                               own_exact_matches=int(exact[local, own].sum()),
                               opposite_exact_matches=int(exact[local, opposite].sum()),
                               nearest_own=float(distance[local, own].min()),
                               nearest_opposite=float(distance[local, opposite].min()),
                               raw_features=' '.join(str(int(v)) for v in x[index]))
                    if firing is not None:
                        contributions = firing[local]*model.theta[len(rules):]
                        strongest = int(np.argmax(np.abs(contributions)))
                        row.update(strongest_rule=strongest, strongest_rule_label=rules[strongest].label,
                                   strongest_contribution=float(contributions[strongest]),
                                   strongest_antecedent=' & '.join(f'F{c}={v:g}' for c,v in zip(rules[strongest].columns,rules[strongest].values)))
                    errors.append(row)
        print(f'Ablation fold {fold}/10 completed', flush=True)
    summary = {}
    for name, scores in scores_by_model.items():
        rows = [r for r in folds if r['model']==name]
        summary[name] = dict(fold_mean={k:float(np.mean([r[k] for r in rows])) for k in
                    ('accuracy','sensitivity','specificity','f1','auc','rules','classifier_seconds','mining_seconds')},
                    fold_std={k:float(np.std([r[k] for r in rows],ddof=1)) for k in ('accuracy','sensitivity','specificity','auc')},
                    pooled=metrics(y,scores))
    # Descriptive paired differences, not an independence-based significance test.
    effects = {}
    for a,b in [('static_boost','fp_boost'),('static_sugeno','fp_sugeno'),
                ('static_boost','static_sugeno'),('fp_boost','fp_sugeno')]:
        effects[b+' minus '+a] = {key:summary[b]['fold_mean'][key]-summary[a]['fold_mean'][key]
                                  for key in ('accuracy','sensitivity','specificity','auc')}
    summary['paired_effects'] = effects
    (out/'ablation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name, records in [('ablation_folds',folds),('oof_predictions',predictions),
                           ('rule_summary',rule_summaries),('rules',rule_rows),
                           ('misclassified',errors),('feature_usage',features)]:
        write_csv(out/(name+'.csv'),records)
    return summary


def timings(x,y,out,repeats):
    split = json.loads((ROOT/'results/seed42/stream_split.json').read_text())
    stream = np.array(split['train_stream_indices'])
    records = []
    # Warm up code paths before timing.
    warm = FPTree().update(x[stream[:300]],y[stream[:300]])
    pattern_rules(warm.mine(.1,3))
    for repeat in range(repeats):
        tree, previous = FPTree(), 0
        for stage,end in enumerate([300,400,500,600,700,800,849]):
            past, new = stream[:end], stream[previous:end]
            row = dict(repeat=repeat,stage=stage,train_n=end,added_n=end-previous)
            # Alternate order to reduce systematic warm-cache ordering bias.
            def incremental():
                start=perf_counter(); tree.update(x[new],y[new]); row['insert_seconds']=perf_counter()-start
                start=perf_counter(); patterns=tree.mine(.1,3); row['mine_seconds']=perf_counter()-start
                return patterns
            def full():
                start=perf_counter(); rebuilt=FPTree().update(x[past],y[past]); row['rebuild_insert_seconds']=perf_counter()-start
                start=perf_counter(); patterns=rebuilt.mine(.1,3); row['rebuild_mine_seconds']=perf_counter()-start
                return patterns
            if repeat%2:
                reference=full(); patterns=incremental()
            else:
                patterns=incremental(); reference=full()
            assert patterns == reference
            def fit_classifier(patterns, prefix=''):
                start=perf_counter(); rules=pattern_rules(patterns,64); row[prefix+'rule_select_seconds']=perf_counter()-start
                start=perf_counter(); model=SparseSugeno(rules,maxiter=500).fit(x[past],y[past]); row[prefix+'fuzzy_fit_seconds']=perf_counter()-start
                return model
            if repeat%2:
                full_fuzzy=fit_classifier(reference,'rebuild_'); fuzzy=fit_classifier(patterns)
            else:
                fuzzy=fit_classifier(patterns); full_fuzzy=fit_classifier(reference,'rebuild_')
            np.testing.assert_allclose(fuzzy.theta,full_fuzzy.theta,rtol=0,atol=0)
            row['optimizer_success']=fuzzy.optimization['success']
            row['total_seconds']=sum(row[k] for k in ('insert_seconds','mine_seconds','rule_select_seconds','fuzzy_fit_seconds'))
            row['speedup_mining']=(row['rebuild_insert_seconds']+row['rebuild_mine_seconds'])/(row['insert_seconds']+row['mine_seconds'])
            row['speedup_pipeline']=sum(row['rebuild_'+k] for k in ('insert_seconds','mine_seconds','rule_select_seconds','fuzzy_fit_seconds'))/row['total_seconds']
            row['nodes']=tree.nodes
            records.append(row)
            previous=end
        print(f'Timing repeat {repeat+1}/{repeats} completed',flush=True)
    summary=[]
    for stage in range(7):
        rows=[r for r in records if r['stage']==stage]
        entry=dict(stage=stage,train_n=rows[0]['train_n'])
        for key in ('insert_seconds','mine_seconds','rule_select_seconds','fuzzy_fit_seconds','total_seconds',
                    'rebuild_insert_seconds','rebuild_mine_seconds','rebuild_rule_select_seconds',
                    'rebuild_fuzzy_fit_seconds','speedup_mining','speedup_pipeline'):
            vals=[r[key] for r in rows]
            entry[key+'_median']=float(np.median(vals))
            entry[key+'_q25']=float(np.quantile(vals,.25))
            entry[key+'_q75']=float(np.quantile(vals,.75))
        summary.append(entry)
    write_csv(out/'timings_raw.csv',records)
    write_csv(out/'timings_summary.csv',summary)
    profiler=cProfile.Profile()
    profiler.runcall(tree.mine,.1,3)
    stats=pstats.Stats(profiler)
    top=[]
    for (filename,line,function),(primitive,calls,selftime,cumtime,_) in sorted(stats.stats.items(),key=lambda v:-v[1][3])[:15]:
        top.append(dict(file=Path(filename).name,line=line,function=function,calls=calls,
                        self_seconds=selftime,cumulative_seconds=cumtime))
    (out/'mining_profile.json').write_text(json.dumps(top,indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/diagnostics_seed42')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<3:
        raise ValueError('Use at least 3 timing repeats.')
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    file=ROOT/'data/dataset_1062.txt'; data=np.loadtxt(file)
    metadata=dict(seed=42,folds=10,min_support=.1,max_length=3,max_rules=64,boost_rounds=8,
                  fuzzy_maxiter=500,timing_repeats=args.repeats,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                  protocol='Same train-only rule set shared by both classifiers. No test-driven tuning.',
                  absence_columns_zero_based=sorted(ABSENCE_COLUMNS),
                  limitations=['Lesion-level folds; patient IDs unavailable.',
                               'Simulated cumulative order; no real timestamps.',
                               'Rule source combines mining objective, search limits, and selection; effects do not isolate each internal choice.',
                               'Budget controls match classwise counts if enough FP rules are available, not antecedent length or search space.',
                               'Both pipelines timed independently with alternating order; 5-repeat small-sample timings remain noisy.',
                               'Single seed and dataset; differences descriptive, not superiority/significance claims.'])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    run_ablation(data[:,1:],data[:,0].astype(int),out)
    timings(data[:,1:],data[:,0].astype(int),out,args.repeats)


if __name__=='__main__':
    main()
