"""Closing prequential comparators; no full-table loader or holdout evaluation."""
import argparse,copy,gc,gzip,hashlib,json,shutil,tracemalloc
from time import perf_counter
import numpy as np
from src.fp_online import initialize,select_rules,rule_records
from src.fptree import FPTree
from src.models import SparseSugeno,StaticBiclusterBoost
from src.online import LocalSugeno
from src.fp_weighted import tree_groups,fit_weighted
from src.reference_comparators import CumulativeLogistic
from run_online_study import SGDLogistic
from run_original_closeout import load_stream
from run_incremental_selection import deep_size
from run_experiments import ROOT,metrics,write_csv

UPDATING=('fp_local','fp_weighted_cumulative','fixed_rules_cumulative','logistic_online','logistic_cumulative','static_draft_cumulative')
FROZEN=('fp_frozen','static_draft_frozen')


def snapshot(m):
    if isinstance(m,(SparseSugeno,LocalSugeno)):
        return dict(kind='sugeno',rules=rule_records(m.rules),theta=m.theta.tolist(),minimum=m.minimum.tolist(),span=m.span.tolist())
    if isinstance(m,StaticBiclusterBoost):
        return dict(kind='static_draft',rules=rule_records(m.rules),pairs=m.pairs,selected=m.selected,
            minimum=m.minimum.tolist(),span=m.span.tolist(),training_max=m.training_max.tolist())
    return dict(kind='logistic',coef=m.model.coef_.tolist(),intercept=m.model.intercept_.tolist(),
        classes=m.model.classes_.tolist(),minimum=m.minimum.tolist(),span=m.span.tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/direct_comparison_v2');ap.add_argument('--memory',action='store_true');args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=False)
    sources=['run_direct_comparison.py','src/reference_comparators.py','tests/test_reference_comparators.py','src/models.py',
        'src/fptree.py','src/fp_online.py','src/online.py','src/fp_weighted.py','run_online_study.py','run_original_closeout.py',
        'run_experiments.py','run_incremental_selection.py','reference.txt','paper/直接对照复现边界.md',
        'results/online_study/splits.json','results/online_study/choices.json','data/dataset_1062.txt']
    protocol=dict(arms=[*UPDATING,*FROZEN],settings='No tuning. FP original L3/support.1/min5/purity>.65/64,frozen scale,500 L-BFGS or inherited initial lr/active for local. Weighted cumulative SAME previous implementation. Fixed rules+metadata frozen,raw prefix joint fit500. Logistic online inherits initial-only alpha/eta,initial SGD fit2000/tol1e-4;later partial_fit only new batch. Logistic cumulative fixed C1/lbfgs500,initial scale frozen.',
        static='PAPER-INSPIRED DRAFT,NOT author replica: existing StaticBiclusterBoost complete linkage/seed_distance0/MSR1e-12/min5/purity>.65/8 rounds;no rule cap. Native current-prefix minmax/FSDND normalization retained,unlike frozen-scale FP/logistic. Initial frozen and cumulative full rebuild variants. No auto fallback or retuning on failure. Original paper10fold vs current simulated stream,do NOT compare published percentage directly.',
        data='Per split seed17/29/43 parse ONLY saved849 allowed stream,initial300,suffix549,batch50/100. All8 predict entire batch BEFORE labels/update. Per-split213 holdout never parsed,mined,fit,predicted,scored,tuned. Repeated internal development,overlapping splits,no independent clinical validation.',
        comparability='Methods have different learning/history/budget semantics explicitly reported. Cumulative arms use allowed history;online logistic parameters only new batch;FP local updates cumulative compressed patterns+new-batch parameters. Neither head nor scaling is fully matched to static draft. No claim all methods optimally tuned or publication superiority.',
        cost='3 ordinary repeats,rotated updating-arm order by stage+repeat. Shared FP insert/query/selection counted ONLY in composed cost of fp_local and fp_weighted_cumulative. Other arms do not need FP maintenance after init. Each cumulative raw arm gets prefix and old-row count,FP weighted restores history from tree;no no-history claim. Initial/prediction separate. Composed costs not independent physical pipeline wall times. Independent memory1repeat per shared-FP and arm-update phase extra allocations,not total RSS. Reachable persistent estimates include tree for dynamic FP or raw input bytes for raw cumulative arms;driver future arrays/diagnostics excluded.',
        mode='memory' if args.memory else 'timing',repeats=1 if args.memory else 3,
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in sources:
        if p.endswith('.py'):
            target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
    splits=json.loads((ROOT/'results/online_study/splits.json').read_text());choices=json.loads((ROOT/'results/online_study/choices.json').read_text())
    updates=[];predictions=[];summaries=[];initials=[];memory=[];audit=[]
    for split in splits:
        seed=split['seed'];ids=np.array(split['train_stream']);assert len(set(ids))==849 and len(split['holdout'])==213 and not set(ids)&set(split['holdout'])
        x,y=load_stream(ROOT/'data/dataset_1062.txt',ids);tree0,base,init=initialize(x[:300],y[:300]);scaling=(base.minimum,base.span)
        fixed=select_rules(tree0.mine(.1,3));choice=next(c for c in choices if c['seed']==seed)
        t=perf_counter();online=SGDLogistic(x[:300],y[:300],scaling,choice['sgd_alpha'],choice['sgd_eta0'],seed);online_initial=perf_counter()-t
        t=perf_counter();logistic=CumulativeLogistic(x[:300],y[:300],scaling);logistic_initial=perf_counter()-t
        t=perf_counter();static=StaticBiclusterBoost().fit(x[:300],y[:300]);static_initial=perf_counter()-t
        initials.append(dict(seed=seed,fp_seconds=init['total_seconds'],fp_success=init['success'],online_logistic_seconds=online_initial,
            cumulative_logistic_seconds=logistic_initial,static_seconds=static_initial,static_rules=len(static.rules),static_pairs=len(static.pairs),
            static_rounds=len(static.selected),sgd_alpha=choice['sgd_alpha'],sgd_eta0=choice['sgd_eta0']))
        references={}
        for batch in (50,100):
            for rep in range(protocol['repeats']):
                tree=copy.deepcopy(tree0)
                models=dict(fp_local=LocalSugeno(base,choice['lr'],choice['active']),fp_weighted_cumulative=copy.deepcopy(base),
                    fixed_rules_cumulative=copy.deepcopy(base),logistic_online=copy.deepcopy(online),logistic_cumulative=copy.deepcopy(logistic),
                    static_draft_cumulative=copy.deepcopy(static),fp_frozen=copy.deepcopy(base),static_draft_frozen=copy.deepcopy(static))
                scores={p:[] for p in models}
                for stage,start in enumerate(range(300,849,batch),1):
                    end=min(start+batch,849);xx=x[start:end];pred={};pt={}
                    for p,m in models.items():
                        t=perf_counter();pred[p]=m.decision_function(xx);pt[p]=perf_counter()-t
                    yy=y[start:end]
                    for p,s in pred.items():
                        scores[p].extend(s)
                        if rep==0 and not args.memory:predictions.extend(dict(seed=seed,batch_size=batch,stage=stage,policy=p,row_index=int(i),label=int(v),score=float(z)) for i,v,z in zip(ids[start:end],yy,s))
                    if args.memory:gc.collect();tracemalloc.start()
                    t=perf_counter();tree.update(xx,yy);insert=perf_counter()-t
                    t=perf_counter();rules=select_rules(tree.mine(.1,3));query_select=perf_counter()-t
                    if args.memory:
                        cur,peak=tracemalloc.get_traced_memory();tracemalloc.stop();memory.append(dict(seed=seed,batch_size=batch,stage=stage,phase='shared_fp',extra_peak_bytes=peak))
                    offset=(stage+rep)%len(UPDATING)
                    for p in UPDATING[offset:]+UPDATING[:offset]:
                        if args.memory:gc.collect();tracemalloc.start()
                        t=perf_counter()
                        if p=='fp_local':models[p].replace_rules(rules);models[p].partial_fit(xx,yy)
                        elif p=='fp_weighted_cumulative':
                            gx,n,k,v=tree_groups(tree,18);models[p]=fit_weighted(rules,gx,n,k,scaling)
                        elif p=='fixed_rules_cumulative':models[p]=SparseSugeno(fixed,maxiter=500).fit(x[:end],y[:end],scaling=scaling)
                        elif p=='logistic_online':models[p].partial_fit(xx,yy)
                        elif p=='logistic_cumulative':models[p].fit(x[:end],y[:end])
                        elif p=='static_draft_cumulative':models[p]=StaticBiclusterBoost().fit(x[:end],y[:end])
                        seconds=perf_counter()-t;m=models[p]
                        if args.memory:
                            cur,peak=tracemalloc.get_traced_memory();tracemalloc.stop();memory.append(dict(seed=seed,batch_size=batch,stage=stage,phase=p,extra_peak_bytes=peak))
                        needs_fp=p in ('fp_local','fp_weighted_cumulative');raw_history=p in ('fixed_rules_cumulative','logistic_cumulative','static_draft_cumulative')
                        state=deep_size((tree,m)) if needs_fp else deep_size(m)
                        if raw_history:state+=int(x[:end].nbytes+y[:end].nbytes)
                        opt=m.optimization if isinstance(m,SparseSugeno) else {}
                        updates.append(dict(seed=seed,batch_size=batch,repeat=rep,stage=stage,policy=p,
                            arm_seconds=seconds,fp_insert_seconds=insert if needs_fp else 0.,fp_query_select_seconds=query_select if needs_fp else 0.,
                            composed_seconds=seconds+(insert+query_select if needs_fp else 0),prediction_seconds=pt[p],persistent_bytes=state,
                            history_raw_rows_passed=start if raw_history else 0,
                            reconstructed_history_vectors=len(gx) if p=='fp_weighted_cumulative' else 0,
                            rules=len(m.rules) if hasattr(m,'rules') else 0,pairs=len(m.pairs) if hasattr(m,'pairs') else 0,
                            optimizer_success=opt.get('success',''),optimizer_iterations=opt.get('iterations',''),optimizer_evaluations=opt.get('function_evaluations','')))
                    if rep==0:
                        record=dict(seed=seed,batch_size=batch,stage=stage,n=end,models={p:snapshot(m) for p,m in models.items()})
                        h=hashlib.sha256(json.dumps(record,sort_keys=True,default=lambda v:v.item()).encode()).hexdigest();audit.append(dict(seed=seed,batch_size=batch,stage=stage,sha256=h))
                        if not args.memory:
                            with gzip.open(out/f'state_s{seed}_b{batch}_t{stage}.json.gz','wt',encoding='utf-8') as f:json.dump(record,f,default=lambda v:v.item())
                for p,s in scores.items():
                    s=np.array(s);assert len(s)==549
                    if rep==0:
                        references[batch,p]=s
                        if not args.memory:summaries.append(dict(seed=seed,batch_size=batch,policy=p,**metrics(y[300:],s)))
                    else:np.testing.assert_array_equal(s,references[batch,p])
                print(f'direct comparison seed={seed} batch={batch} repeat={rep+1}:8 arms complete',flush=True)
        for name,rows in [('summary',summaries),('predictions',predictions),('updates',updates),('initial',initials),('memory',memory)]:
            if rows:write_csv(out/f'{name}.csv',rows)
        (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'completion.json').write_text(json.dumps(dict(completed=True,holdout_rows_parsed_or_evaluated=0,predictions=len(predictions),updates=len(updates),repeats_exact=True)),encoding='utf-8')


if __name__=='__main__':main()
