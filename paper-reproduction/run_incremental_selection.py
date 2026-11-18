"""Three simulated streams: same-rule cold baselines vs exact caches vs warm fits."""
import json
import hashlib
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.model_selection import train_test_split
from src.models import SparseSugeno
from src.fptree import FPTree,pattern_rules
from src.selection import CoveragePool
from src.incremental import PatternCoverageIndex,CachedCoveragePool
from run_selection_study import tune
from run_experiments import ROOT,metrics,write_csv

OUT=ROOT/'results/incremental_selection'
POLICIES=['full_cold','fp_incremental_cold','fp_fast_selection_cold','indexed_cold','indexed_warm']


def rule_keys(rules):
    return [(r.columns,tuple(r.values),r.label,r.count,r.confidence) for r in rules]


def deep_size(obj,seen=None):
    # Reachable Python object estimate, not process RSS or peak memory.
    if seen is None: seen=set()
    if id(obj) in seen: return 0
    seen.add(id(obj)); size=sys.getsizeof(obj)
    if isinstance(obj,dict):
        size+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in obj.items())
    elif isinstance(obj,(tuple,list,set)):
        size+=sum(deep_size(v,seen) for v in obj)
    elif hasattr(obj,'__dict__'):
        size+=deep_size(obj.__dict__,seen)
    elif hasattr(obj,'__slots__'):
        size+=sum(deep_size(getattr(obj,k),seen) for k in obj.__slots__)
    return size


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    file=ROOT/'data/dataset_1062.txt'; a=np.loadtxt(file); x,y=a[:,1:],a[:,0].astype(int)
    protocol=dict(seeds=[17,29,43],repeats=3,initial_n=300,batch_size=100,holdout_fraction=.2,
                  support=.1,max_length=3,rule_budget=64,confidence=.65,maxiter=500,
                  tuning='Initial 300 only; 3-fold inner CV, same joint criterion/grid as nested study; freeze alpha for entire stream.',
                  policies=POLICIES,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                  exact_comparison='Same patterns, masks, rule order and cold predictions checked at every stage.',
                  warm='Transfer matching rule consequents; transfer widths only when selected-feature scaling unchanged; same loss/500 iteration limit.',
                  costs='Initial CV separate; all update/mining/pool/selection/fitting measured; evaluation/audits excluded; order rotates.',
                  limits=['Exploratory repeated use of 1062 cases; holdouts not independent external validation.',
                          'No patient IDs or timestamps; lesion-level simulated arrivals.',
                          'Enumerated index is bounded exhaustive itemset+bitmap cache, NOT FP-growth; retains all length<=3 itemsets, no historical raw rows.',
                          'Fuzzy warm fit still uses all accumulated training rows; not online learning with bounded memory.',
                          'Memory is reachable Python object estimate, not RSS/peak; shared external training arrays excluded.'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    timings,evaluations,predictions,choices,splits,inner_scores=[],[],[],[],[],[]
    for stream_id,seed in enumerate(protocol['seeds'],1):
        stream,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        np.random.default_rng(seed).shuffle(stream)
        start=perf_counter(); inner_splits=[]
        alpha,trace=tune(x[stream[:300]],y[stream[:300]],stream_id,inner_scores,inner_splits,stream[:300])
        tuning_seconds=perf_counter()-start
        choices.append(dict(seed=seed,alpha='original' if alpha is None else alpha,tuning_seconds=tuning_seconds,trace=trace))
        splits.append(dict(seed=seed,train_stream=stream.tolist(),test=test.tolist(),inner=inner_splits))
        repeat_reference={}
        for repeat in range(3):
            trees={name:FPTree() for name in ['fp_incremental_cold','fp_fast_selection_cold']}
            indices={name:PatternCoverageIndex(3) for name in ['indexed_cold','indexed_warm']}
            models={}; frozen=None; previous=0
            for stage,end in enumerate([300,400,500,600,700,800,849]):
                past,new=stream[:end],stream[previous:end]
                if repeat==0 and stage>0:
                    for policy,model in [*models.items(),('frozen',frozen)]:
                        scores=model.decision_function(x[new])
                        evaluations.append(dict(seed=seed,stage=stage,n=end,policy=policy,evaluation='next_batch',
                                                **metrics(y[new],scores)))
                        predictions.extend(dict(seed=seed,stage=stage,policy=policy,evaluation='next_batch',
                                                row_index=int(i),label=int(y[i]),score=float(s)) for i,s in zip(new,scores))
                stage_patterns,stage_rules,stage_scores={}, {}, {}
                offset=(repeat+stage)%len(POLICIES)
                order=POLICIES[offset:]+POLICIES[:offset]
                for policy in order:
                    row=dict(seed=seed,repeat=repeat,stage=stage,train_n=end,added_n=end-previous,policy=policy)
                    start=perf_counter()
                    if policy=='full_cold':
                        state=FPTree().update(x[past],y[past])
                    elif policy in trees:
                        state=trees[policy].update(x[new],y[new])
                    else:
                        state=indices[policy].update(x[new],y[new])
                    row['insert_seconds']=perf_counter()-start
                    start=perf_counter(); patterns=state.mine(.1,3) if isinstance(state,FPTree) else state.mine(.1)
                    row['mine_seconds']=perf_counter()-start
                    start=perf_counter()
                    if alpha is None:
                        rules=pattern_rules(patterns,64)
                    else:
                        if isinstance(state,FPTree):
                            constructor=CachedCoveragePool if policy=='fp_fast_selection_cold' else CoveragePool
                            pool=constructor(patterns,x[past],y[past])
                        else:
                            pool=state.pool(patterns)
                        rules=pool.select(alpha,64)
                    row['selection_seconds']=perf_counter()-start
                    old=models.get(policy) if policy=='indexed_warm' else None
                    start=perf_counter(); model=SparseSugeno(rules,maxiter=500).fit(x[past],y[past],initial_model=old)
                    row['fit_seconds']=perf_counter()-start
                    row['total_seconds']=sum(row[k] for k in ('insert_seconds','mine_seconds','selection_seconds','fit_seconds'))
                    row.update(optimizer_success=model.optimization['success'],iterations=model.optimization['iterations'],
                               transferred_rules=model.optimization['transferred_rules'],loss=model.optimization['loss'],
                               rules=len(rules),patterns=len(patterns),state_bytes=deep_size(state))
                    timings.append(row)
                    models[policy]=model
                    stage_patterns[policy]=patterns; stage_rules[policy]=rule_keys(rules)
                    scores=model.decision_function(x[test]); stage_scores[policy]=scores
                    if repeat==0:
                        evaluations.append(dict(seed=seed,stage=stage,n=end,policy=policy,evaluation='fixed_holdout',
                                                **metrics(y[test],scores)))
                        predictions.extend(dict(seed=seed,stage=stage,policy=policy,evaluation='fixed_holdout',
                                                row_index=int(i),label=int(y[i]),score=float(s)) for i,s in zip(test,scores))
                    else:
                        np.testing.assert_allclose(scores,repeat_reference[(stage,policy)],rtol=0,atol=1e-10)
                for policy in POLICIES:
                    assert stage_patterns[policy]==stage_patterns['full_cold']
                    assert stage_rules[policy]==stage_rules['full_cold']
                    if policy!='indexed_warm':
                        np.testing.assert_allclose(stage_scores[policy],stage_scores['full_cold'],rtol=0,atol=1e-10)
                if repeat==0:
                    repeat_reference.update({(stage,p):s.copy() for p,s in stage_scores.items()})
                if stage==0: frozen=models['full_cold']
                if repeat==0:
                    evaluations.append(dict(seed=seed,stage=stage,n=end,policy='frozen',evaluation='fixed_holdout',
                                            **metrics(y[test],frozen.decision_function(x[test]))))
                previous=end
            print(f'Stream seed={seed}, repeat {repeat+1}/3 complete; alpha={alpha}',flush=True)
        for name,records in [('timings',timings),('evaluation',evaluations),('predictions',predictions),('initial_tuning',inner_scores)]:
            write_csv(OUT/(name+'.csv'),records)
        (OUT/'choices.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
        (OUT/'splits.json').write_text(json.dumps(splits),encoding='utf-8')
    print('All exactness and repeat-prediction checks passed.',flush=True)


if __name__=='__main__':
    main()
