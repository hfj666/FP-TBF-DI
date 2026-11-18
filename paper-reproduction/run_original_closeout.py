"""Frozen closeout on original per-cohort streams; never evaluate holdouts."""
import copy
import hashlib
import json
from time import perf_counter
import numpy as np
from run_experiments import ROOT,metrics,write_csv
from run_incremental_selection import deep_size
from run_online_study import SGDLogistic
from src.fast_coverage_index import FastPatternCoverageIndex
from src.budgeted_rules import select_budgeted
from src.predictive_exchange import PredictivePipeline
from src.models import SparseSugeno
from src.online import LocalSugeno,EClass0

POLICIES=('retained','coverage','prediction','protected','global','frozen','eclass0','logistic_online')


def load_stream(path,indices):
    """Parse only approved stream rows, retaining their saved order."""
    wanted=set(map(int,indices));rows={}
    with path.open() as f:
        for i,line in enumerate(f):
            if i in wanted:rows[i]=np.fromstring(line,sep=' ')
    a=np.array([rows[int(i)] for i in indices])
    if a.shape!=(849,19):raise ValueError(f'unexpected stream shape {a.shape}')
    return a[:,1:],a[:,0].astype(int)


def main():
    out=ROOT/'results/original_predictive_closeout';out.mkdir(exist_ok=False)
    old=ROOT/'results/online_study'
    splits=json.loads((old/'splits.json').read_text());choices=json.loads((old/'choices.json').read_text())
    paths=[ROOT/'run_original_closeout.py',ROOT/'src/predictive_exchange.py',ROOT/'src/budgeted_rules.py',ROOT/'src/online.py',ROOT/'src/models.py',old/'splits.json',old/'choices.json',ROOT/'data/dataset_1062.txt']
    protocol=dict(seeds=[17,29,43],batches=[50,100],initial=300,suffix=549,policies=POLICIES,
        frozen_settings='Original task existing protocol: length<=3 support.1 purity>.65 min5,64 rules class-balanced,alpha.5; at most4 voluntary exchanges; inherited initial-only lr/active/eClass/logistic choices. No tuning. Public pilot rule capacity32/length2 belongs to categorical tasks; original task capacity64/length3 retained.',
        holdout='Each saved cohort has its own213 holdout rows; not a newly constructed universal213 set. Assert disjoint train_stream/holdout per cohort. Parse only train_stream rows. No holdout features/labels to scaler, mining, fit, prediction, scoring or selection. Raw-file hash scans bytes for provenance only. No cross-cohort pooling of training state.',
        evaluation='Predict full batch before updates.549 suffix rows once per policy per cohort/batch. Reused data and overlapping cohorts: internal simulation, no independent validation.',
        primary='Balanced accuracy; sensitivity/specificity,FP/FN,accuracy,AUC secondary. Mean across six cohort/batch configurations; report each configuration.',
        decision_rule='A candidate route continues only if SAME arm prediction or protected (1) has balanced-accuracy wins on >=4 of6 configs against BOTH retained and coverage; (2) mean balanced-accuracy gain>=.005 against BOTH; (3) mean sensitivity decline no more than.005 against EITHER. Engineering support additionally requires total update time and total rule arrivals <=2x coverage. If no arm passes predictive criteria, stop this current mechanism route as a methods-paper mainline. Passing prediction but failing cost means prediction signal only, not deployable/novel method. Thresholds are pragmatic go/no-go, not clinical noninferiority or significance. Frozen before new results.',
        stop='No new variants, parameter searches, tolerances or retrospective winner substitution after results. Combine prior public-task mixed results in final judgment, without claiming novelty from success here.',
        cost='One rotated sequential timing run. End reachable persistent bytes, not peak/RSS. No raw historical rows passed to parameter updates; cumulative history bitmaps persist.',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    predictions=[];updates=[];summaries=[];initials=[];checks=[]
    for split in splits:
        seed=split['seed'];stream=np.array(split['train_stream']);holdout=set(split['holdout'])
        assert len(stream)==849 and len(set(stream))==849 and len(holdout)==213 and not set(stream)&holdout
        x,y=load_stream(ROOT/'data/dataset_1062.txt',stream)
        choice=next(c for c in choices if c['seed']==seed)
        scaling=(x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.))
        t=perf_counter();index=FastPatternCoverageIndex(3).update(x[:300],y[:300])
        rules,keys,_=select_budgeted(index.pool(index.mine(.1)),global_rebuild=True,max_rules=64)
        base=SparseSugeno(rules,maxiter=500).fit(x[:300],y[:300],scaling=scaling)
        assert base.optimization['success']
        initials.append(dict(seed=seed,rules=len(rules),initial_seconds=perf_counter()-t,**{k:choice[k] for k in ('lr','active','sigma_floor','sgd_alpha','sgd_eta0')}))
        for batch_size in (50,100):
            arms={p:PredictivePipeline(index,LocalSugeno(base,choice['lr'],choice['active']),keys,p,max_rules=64) for p in POLICIES[:5]}
            other=dict(frozen=copy.deepcopy(base),eclass0=EClass0(*scaling,sigma_floor=choice['sigma_floor']).partial_fit(x[:300],y[:300]),
                logistic_online=SGDLogistic(x[:300],y[:300],scaling,choice['sgd_alpha'],choice['sgd_eta0'],seed))
            scores={p:[] for p in POLICIES}
            for stage,start in enumerate(range(300,849,batch_size),1):
                stop=min(849,start+batch_size);xx=x[start:stop];yy=y[start:stop]
                assert not set(stream[start:stop])&holdout
                staged={p:(arms[p].model if p in arms else other[p]).decision_function(xx) for p in POLICIES}
                if stage==1:
                    for p in POLICIES[:5]:np.testing.assert_array_equal(staged[p],staged['frozen'])
                for p,score in staged.items():
                    scores[p].extend(score)
                    predictions.extend(dict(seed=seed,batch_size=batch_size,stage=stage,policy=p,row_index=int(i),label=int(v),score=float(s)) for i,v,s in zip(stream[start:stop],yy,score))
                offset=stage%len(POLICIES)
                for p in POLICIES[offset:]+POLICIES[:offset]:
                    t=perf_counter()
                    if p in arms:stats=arms[p].update(xx,yy)
                    else:
                        if p!='frozen':other[p].partial_fit(xx,yy)
                        stats={}
                    seconds=perf_counter()-t
                    updates.append(dict(seed=seed,batch_size=batch_size,stage=stage,policy=p,seconds=seconds,
                        state_bytes=deep_size(arms[p] if p in arms else other[p]) if stop==849 else 0,history_raw_fit_rows=0,**stats))
            for p,score in scores.items():
                assert len(score)==549
                m=metrics(y[300:],np.array(score));m['balanced_accuracy']=(m['sensitivity']+m['specificity'])/2
                summaries.append(dict(seed=seed,batch_size=batch_size,policy=p,**m))
            checks.append(dict(seed=seed,batch_size=batch_size,holdout_evaluations=0,train=849,initial=300,evaluated_suffix=549,holdout_disjoint=True))
            for name,rows in [('predictions',predictions),('updates',updates),('summary',summaries),('initial',initials)]:write_csv(out/(name+'.csv'),rows)
            print(f'original seed={seed} batch={batch_size} complete',flush=True)
    pred=[r for r in updates if r['policy'] in ('prediction','protected')]
    audit=dict(configurations=checks,prediction_rows=len(predictions),summary_rows=len(summaries),
        holdout_fit_or_prediction_rows=0,protected_violations=sum(r['pre_sgd_new_errors'] for r in pred if r['policy']=='protected'),
        swap_budget_violations=sum(r['swaps']>4 for r in pred),negative_descent_violations=sum(r['empirical_loss_drop']<0 for r in pred))
    assert audit['protected_violations']==audit['swap_budget_violations']==audit['negative_descent_violations']==0
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(audit)


if __name__=='__main__':main()
