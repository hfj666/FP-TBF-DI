"""Predeclared nested CV comparison of original vs complementary FP selection."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.fptree import FPTree, pattern_rules
from src.selection import CoveragePool
from src.models import StaticBiclusterBoost, SparseSugeno
from run_experiments import ROOT, metrics, write_csv
from run_diagnostics import rule_diagnostics

OUTPUT = ROOT/'results/selection_nested_seed20261030'
GRID = [None, .25, .5, .75]
CLASSIFIERS = ['boost','sugeno']


def fit_model(classifier, rules, x, y):
    start=perf_counter()
    if classifier=='boost':
        model=StaticBiclusterBoost().fit_rules(x,y,rules)
        extra={'optimizer_success':True,'optimizer_iterations':0}
    else:
        model=SparseSugeno(rules,maxiter=500).fit(x,y)
        extra={'optimizer_success':model.optimization['success'],
               'optimizer_iterations':model.optimization['iterations']}
    return model,dict(fit_seconds=perf_counter()-start,**extra)


def candidate_rules(patterns,x,y):
    start=perf_counter()
    pool=CoveragePool(patterns,x,y)
    rules={None:pattern_rules(patterns,64)}
    for alpha in GRID[1:]:
        rules[alpha]=pool.select(alpha,64)
    return rules,perf_counter()-start


def tune(x,y,fold,inner_rows,splits,global_indices):
    results={alpha:{c:[] for c in CLASSIFIERS} for alpha in GRID}
    for inner,(train,validation) in enumerate(StratifiedKFold(3,shuffle=True,random_state=1000+fold).split(x,y),1):
        splits.append(dict(outer_fold=fold,inner_fold=inner,train=global_indices[train].tolist(),
                           validation=global_indices[validation].tolist()))
        patterns=FPTree().update(x[train],y[train]).mine(.1,3)
        options,_=candidate_rules(patterns,x[train],y[train])
        for alpha in GRID:
            for classifier in CLASSIFIERS:
                model,extra=fit_model(classifier,options[alpha],x[train],y[train])
                m=metrics(y[validation],model.decision_function(x[validation]))
                m['balanced_accuracy']=(m['sensitivity']+m['specificity'])/2
                results[alpha][classifier].append(m)
                inner_rows.append(dict(outer_fold=fold,inner_fold=inner,alpha='original' if alpha is None else alpha,
                                       classifier=classifier,**m,**extra))
    averaged={alpha:{c:{key:float(np.mean([r[key] for r in results[alpha][c]]))
                       for key in ('balanced_accuracy','sensitivity','specificity')}
                     for c in CLASSIFIERS} for alpha in GRID}
    def feasible(alpha):
        return all(averaged[alpha][c]['sensitivity'] >= averaged[None][c]['sensitivity']-.01
                   for c in CLASSIFIERS)
    # One common alpha is selected for both classifiers, so both see the same rules.
    # Baseline is always eligible. Fixed tie order favors simpler unchanged policy.
    candidates=[a for a in GRID if feasible(a)]
    chosen=max(candidates,key=lambda a:round(np.mean([averaged[a][c]['balanced_accuracy'] for c in CLASSIFIERS]),12))
    trace={str(a):dict(feasible=feasible(a),joint_balanced_accuracy=float(np.mean(
           [averaged[a][c]['balanced_accuracy'] for c in CLASSIFIERS])),by_classifier=averaged[a]) for a in GRID}
    return chosen,trace


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/dataset_1062.txt'; a=np.loadtxt(source); x,y=a[:,1:],a[:,0].astype(int)
    protocol=dict(outer_seed=20261030,outer_folds=10,inner_folds=3,inner_seed='1000+outer_fold',
                  grid=['original',.25,.5,.75],support=.1,max_length=3,max_rules=64,confidence=.65,
                  boost_rounds=8,fuzzy_maxiter=500,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  score='alpha*normalized_quality + (1-alpha)*(new_own_class_coverage_fraction-max_same_class_Jaccard)',
                  quality='(purity-0.5)*sqrt(count)*sqrt(length); normalized per class',
                  tuning='max mean inner balanced accuracy across BOTH classifiers; each classifier inner sensitivity >= original minus .01; original eligible; baseline wins ties',
                  limits=['Exploratory reuse of same dataset after inspecting seed42; new seed is not independent external validation.',
                          'No patient IDs; lesion-level splits only.',
                          'Inner sensitivity tolerance 1 percentage point is a prespecified exploratory criterion, not clinical noninferiority.',
                          'No outer-test threshold tuning; all decision thresholds zero.',
                          'This changes selection only; no efficient online parameter update or novelty claim.'])
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    predictions,fold_rows,inner_rows,diagnostics,rules_dump,choices,splits=[],[],[],[],[],[],[]
    for fold,(train,test) in enumerate(StratifiedKFold(10,shuffle=True,random_state=20261030).split(x,y),1):
        splits.append(dict(outer_fold=fold,inner_fold=0,train=train.tolist(),validation=test.tolist()))
        chosen,trace=tune(x[train],y[train],fold,inner_rows,splits,train)
        choices.append(dict(fold=fold,alpha='original' if chosen is None else chosen,trace=trace))
        start=perf_counter(); patterns=FPTree().update(x[train],y[train]).mine(.1,3)
        mining_seconds=perf_counter()-start
        options,selection_seconds=candidate_rules(patterns,x[train],y[train])
        static=StaticBiclusterBoost().mine_rules(x[train],y[train])
        for policy,rules in [('original',options[None]),('selected',options[chosen]),('static',static.raw_rules())]:
            for split,indices in [('train',train),('test',test)]:
                summary,records=rule_diagnostics(rules,x[indices],y[indices],fold,policy,static.minimum,static.span)
                diagnostics.append(dict(split=split,**summary))
                if split=='train':
                    rules_dump.extend(records)
            for classifier in CLASSIFIERS:
                model,extra=fit_model(classifier,rules,x[train],y[train])
                scores=model.decision_function(x[test])
                m=metrics(y[test],scores); m['balanced_accuracy']=(m['sensitivity']+m['specificity'])/2
                name=policy+'_'+classifier
                fold_rows.append(dict(model=name,fold=fold,selected_alpha='original' if chosen is None else chosen,
                                      rules=len(rules),mining_seconds=mining_seconds,
                                      all_candidate_selection_seconds=selection_seconds,**m,**extra))
                predictions.extend(dict(model=name,fold=fold,row_index=int(i),label=int(y[i]),score=float(s))
                                   for i,s in zip(test,scores))
        # Save bounded checkpoints after complete outer folds, preserving work if interrupted.
        for name,rows in [('outer_folds',fold_rows),('inner_scores',inner_rows),('oof_predictions',predictions),
                          ('rule_diagnostics',diagnostics),('selected_rules',rules_dump)]:
            write_csv(OUTPUT/(name+'.csv'),rows)
        (OUTPUT/'choices.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
        (OUTPUT/'splits.json').write_text(json.dumps(splits),encoding='utf-8')
        print(f'Nested fold {fold}/10 complete; selected alpha={chosen}',flush=True)
    summary={}
    for name in sorted(set(r['model'] for r in fold_rows)):
        rows=[r for r in fold_rows if r['model']==name]
        pred=sorted([r for r in predictions if r['model']==name],key=lambda r:r['row_index'])
        summary[name]=dict(fold_mean={key:float(np.mean([r[key] for r in rows])) for key in
                    ('accuracy','balanced_accuracy','sensitivity','specificity','auc','rules')},
                    fold_std={key:float(np.std([r[key] for r in rows],ddof=1)) for key in
                    ('accuracy','balanced_accuracy','sensitivity','specificity','auc')},
                    pooled=metrics(y,np.array([r['score'] for r in pred])))
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
