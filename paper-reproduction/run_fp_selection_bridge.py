"""Restore existing coverage selection in the unified FP pipeline, no tuning."""
import copy,gzip,hashlib,json,shutil
from time import perf_counter
import numpy as np
from src.fptree import FPTree
from src.fp_online import select_rules,rule_records
from src.models import SparseSugeno
from src.online import LocalSugeno
from src.selection import CoveragePool
from run_original_closeout import load_stream
from run_experiments import ROOT,metrics,write_csv

OUT=ROOT/'results/fp_selection_bridge'
def main():
    OUT.mkdir(exist_ok=False)
    paths=['run_fp_selection_bridge.py','src/fptree.py','src/fp_online.py','src/models.py','src/online.py','src/selection.py','run_original_closeout.py','run_experiments.py','results/online_study/splits.json','results/online_study/choices.json','data/dataset_1062.txt']
    protocol=dict(settings='Existing selectors original quality vs CoveragePool alpha=.5, shared FP support.1/length3/purity>.65/min5/64=32perclass. Initial300 each selector own cold Sugeno500; SAME initial-only frozen scaling. Cross local inherited lr/active vs cold cumulative Sugeno500 plus own frozen initial controls. Local/cumulative differ optimizer and history: operational paths, NOT pure data-window causal effect. No tuning or new mechanism.',
        data='Each seed17/29/43 allowed849 only,initial300,suffix549,batch50/100. All predict whole batch BEFORE reveal. Per-split213 excluded this run; historically some holdouts evaluated and full-data CV performed, NOT untouched independent test sets. Reused overlapping development simulation.',
        cost='Single run, no speed superiority. CoveragePool scans revealed raw prefix; even local parameter arm is NOT no-history end-to-end. Count selection and fit history separately. All last-batch updates included.',
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for p in paths:
        if p.endswith('.py'):
            t=OUT/'source_snapshot'/p;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,t)
    preds=[];summaries=[];updates=[];initials=[];audit=[]
    splits=json.loads((ROOT/'results/online_study/splits.json').read_text());choices=json.loads((ROOT/'results/online_study/choices.json').read_text())
    for split in splits:
        seed=split['seed'];ids=np.array(split['train_stream']);assert len(set(ids))==849 and not set(ids)&set(split['holdout'])
        x,y=load_stream(ROOT/'data/dataset_1062.txt',ids);scale=(x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.))
        tree0=FPTree().update(x[:300],y[:300]);patterns=tree0.mine(.1,3)
        initial_rules=dict(original=select_rules(patterns),coverage=CoveragePool(patterns,x[:300],y[:300]).select(.5,64))
        bases={}
        for selector,rules in initial_rules.items():
            bases[selector]=SparseSugeno(rules,maxiter=500).fit(x[:300],y[:300],scaling=scale)
            initials.append(dict(seed=seed,selector=selector,**bases[selector].optimization))
        choice=next(c for c in choices if c['seed']==seed)
        for batch in (50,100):
            tree=copy.deepcopy(tree0);models={}
            for sel,base in bases.items():
                models[sel+'_local']=LocalSugeno(base,choice['lr'],choice['active'])
                models[sel+'_cumulative']=copy.deepcopy(base);models[sel+'_frozen']=copy.deepcopy(base)
            scores={p:[] for p in models}
            for stage,start in enumerate(range(300,849,batch),1):
                end=min(start+batch,849);xx=x[start:end]
                pp={p:m.decision_function(xx) for p,m in models.items()};yy=y[start:end]
                for p,s in pp.items():
                    scores[p].extend(s);preds.extend(dict(seed=seed,batch_size=batch,stage=stage,policy=p,row_index=int(i),label=int(v),score=float(z)) for i,v,z in zip(ids[start:end],yy,s))
                tree.update(xx,yy);patterns=tree.mine(.1,3)
                selected={};select_times={}
                for sel in ('original','coverage'):
                    t=perf_counter();selected[sel]=select_rules(patterns) if sel=='original' else CoveragePool(patterns,x[:end],y[:end]).select(.5,64);select_times[sel]=perf_counter()-t
                for sel,rules in selected.items():
                    assert len(rules)==64
                    for mode in ('local','cumulative'):
                        p=sel+'_'+mode;t=perf_counter()
                        if mode=='local':models[p].replace_rules(rules);models[p].partial_fit(xx,yy)
                        else:models[p]=SparseSugeno(rules,maxiter=500).fit(x[:end],y[:end],scaling=scale)
                        updates.append(dict(seed=seed,batch_size=batch,stage=stage,policy=p,fit_seconds=perf_counter()-t,selection_seconds=select_times[sel],selection_history_rows=start if sel=='coverage' else 0,parameter_history_rows=start if mode=='cumulative' else 0,**(models[p].optimization if mode=='cumulative' else {})))
                record=dict(seed=seed,batch_size=batch,stage=stage,prefix=end,selected={s:rule_records(r) for s,r in selected.items()},models={p:dict(rules=rule_records(m.rules),theta=m.theta.tolist(),minimum=m.minimum.tolist(),span=m.span.tolist()) for p,m in models.items()})
                raw=json.dumps(record,sort_keys=True);h=hashlib.sha256(raw.encode()).hexdigest()
                with gzip.open(OUT/f'state_s{seed}_b{batch}_t{stage}.json.gz','wt',encoding='utf-8') as f:f.write(raw)
                audit.append(dict(seed=seed,batch_size=batch,stage=stage,sha256=h))
            for p,s in scores.items():summaries.append(dict(seed=seed,batch_size=batch,policy=p,**metrics(y[300:],np.array(s))))
            print(f'bridge seed={seed} batch={batch} complete',flush=True)
    for n,r in [('predictions',preds),('summary',summaries),('updates',updates),('initial',initials)]:write_csv(OUT/f'{n}.csv',r)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (OUT/'completion.json').write_text(json.dumps(dict(completed=True,predictions=len(preds),updates=len(updates),holdout_rows_parsed_or_evaluated=0)),encoding='utf-8')

if __name__=='__main__':main()
