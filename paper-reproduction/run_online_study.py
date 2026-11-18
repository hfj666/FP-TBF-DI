"""Batch-prequential online study; all choices made before stream suffix is read."""
import copy
import hashlib
import json
from time import perf_counter
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.linear_model import SGDClassifier
from src.incremental import PatternCoverageIndex
from src.models import SparseSugeno
from src.online import EClass0,LocalSugeno
from run_experiments import ROOT,metrics,write_csv
from run_incremental_selection import deep_size

OUT=ROOT/'results/online_study'
POLICIES=['full_retrain','local_dynamic','local_fixed','eclass0','sgd_logistic','frozen']


def selected_rules(index):
    patterns=index.mine(.1);pool=index.pool(patterns);rules=pool.select(.5,64)
    visits=sum(min(32,len(p))*len(p)-min(32,len(p))*(min(32,len(p))-1)//2 for p in pool.pools.values())
    return rules,dict(compressed_patterns_examined=len(index.stats),candidate_score_evaluations=visits,
                      compressed_history_span=index.n)


def initial_fit(x,y,scaling):
    index=PatternCoverageIndex(3).update(x,y)
    rules,_=selected_rules(index)
    model=SparseSugeno(rules,maxiter=500).fit(x,y,scaling=scaling)
    return index,model


class FullRetrain:
    def __init__(self,index,model,x,y,scaling):
        self.index=copy.deepcopy(index);self.model=copy.deepcopy(model)
        self.history_x=x.copy();self.history_y=y.copy();self.scaling=scaling
    def decision_function(self,x):return self.model.decision_function(x)
    def update(self,x,y):
        previous=len(self.history_y)
        start=perf_counter();self.index.update(x,y);rules,counters=selected_rules(self.index)
        structure_seconds=perf_counter()-start
        self.history_x=np.vstack([self.history_x,x]);self.history_y=np.r_[self.history_y,y]
        start=perf_counter();self.model=SparseSugeno(rules,maxiter=500).fit(self.history_x,self.history_y,scaling=self.scaling)
        return dict(history_rows_passed_to_fit=previous,retained_raw_history_rows=len(self.history_y),
                    history_row_objective_evaluations=previous*self.model.optimization['function_evaluations'],
                    new_row_passes=2,structure_seconds=structure_seconds,parameter_seconds=perf_counter()-start,
                    parameter_rule_updates=0,**counters,optimizer_success=self.model.optimization['success'],
                    optimizer_evaluations=self.model.optimization['function_evaluations'],rules=len(rules),
                    rules_new=0,rules_removed=0)


class LocalPipeline:
    def __init__(self,index,model,lr,active,dynamic):
        self.index=copy.deepcopy(index) if dynamic else None
        self.model=LocalSugeno(model,lr,active)
    def decision_function(self,x):return self.model.decision_function(x)
    def update(self,x,y):
        counters=dict(compressed_patterns_examined=0,candidate_score_evaluations=0,compressed_history_span=0)
        change=dict(new=0,removed=0)
        start=perf_counter()
        if self.index is not None:
            self.index.update(x,y);rules,counters=selected_rules(self.index)
            change=self.model.replace_rules(rules)
        structure_seconds=perf_counter()-start
        before=self.model.parameter_rule_updates
        start=perf_counter();self.model.partial_fit(x,y)
        return dict(history_rows_passed_to_fit=0,retained_raw_history_rows=0,history_row_objective_evaluations=0,
                    new_row_passes=2 if self.index is not None else 1,structure_seconds=structure_seconds,
                    parameter_seconds=perf_counter()-start,parameter_rule_updates=self.model.parameter_rule_updates-before,
                    **counters,optimizer_success=True,optimizer_evaluations=0,rules=len(self.model.rules),
                    rules_new=change['new'],rules_removed=change['removed'])


class EClassPipeline:
    def __init__(self,model):self.model=copy.deepcopy(model)
    def decision_function(self,x):return self.model.decision_function(x)
    def update(self,x,y):
        added,removed=self.model.rules_added,self.model.rules_removed
        start=perf_counter();self.model.partial_fit(x,y)
        return dict(history_rows_passed_to_fit=0,retained_raw_history_rows=0,history_row_objective_evaluations=0,
                    new_row_passes=1,structure_seconds=0.,parameter_seconds=perf_counter()-start,
                    parameter_rule_updates=len(y),compressed_patterns_examined=0,candidate_score_evaluations=0,
                    compressed_history_span=0,optimizer_success=True,optimizer_evaluations=0,rules=self.model.rule_count,
                    rules_new=self.model.rules_added-added,rules_removed=self.model.rules_removed-removed)


class Frozen:
    def __init__(self,model):self.model=copy.deepcopy(model)
    def decision_function(self,x):return self.model.decision_function(x)
    def update(self,x,y):
        return dict(history_rows_passed_to_fit=0,retained_raw_history_rows=0,history_row_objective_evaluations=0,
                    new_row_passes=0,structure_seconds=0.,parameter_seconds=0.,parameter_rule_updates=0,
                    compressed_patterns_examined=0,candidate_score_evaluations=0,compressed_history_span=0,
                    optimizer_success=True,optimizer_evaluations=0,rules=len(self.model.rules),rules_new=0,rules_removed=0)


class SGDLogistic:
    def __init__(self,x,y,scaling,alpha,eta0,seed):
        self.minimum=scaling[0].copy();self.span=scaling[1].copy()
        self.model=SGDClassifier(loss='log_loss',alpha=alpha,learning_rate='constant',eta0=eta0,
                                 max_iter=2000,tol=1e-4,random_state=seed,shuffle=True)
        self.model.fit((x-self.minimum)/self.span,y)
    def decision_function(self,x):return self.model.decision_function((x-self.minimum)/self.span)
    def partial_fit(self,x,y):
        self.model.partial_fit((x-self.minimum)/self.span,y)
        return self
    def update(self,x,y):
        start=perf_counter();self.partial_fit(x,y)
        return dict(history_rows_passed_to_fit=0,retained_raw_history_rows=0,history_row_objective_evaluations=0,
                    new_row_passes=1,structure_seconds=0.,parameter_seconds=perf_counter()-start,
                    parameter_rule_updates=0,compressed_patterns_examined=0,candidate_score_evaluations=0,
                    compressed_history_span=0,optimizer_success=True,optimizer_evaluations=0,rules=0,
                    rules_new=0,rules_removed=0)


def initial_tuning(x,y,seed,initial_indices):
    records,splits=[],[]
    local_grid=[(lr,k) for lr in [.01,.05,.1] for k in [4,8]]
    for inner,(train,val) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+100).split(x,y),1):
        splits.append(dict(inner=inner,train=initial_indices[train].tolist(),validation=initial_indices[val].tolist()))
        scaling=(x[train].min(0),np.maximum(np.ptp(x[train],axis=0),1.))
        index,base=initial_fit(x[train],y[train],scaling)
        models={f'local_{lr}_{k}':LocalSugeno(base,lr,k) for lr,k in local_grid}
        for floor in [.03,.1,.25,.5,1.]:models[f'eclass_{floor}']=EClass0(*scaling,sigma_floor=floor).partial_fit(x[train],y[train])
        for alpha in [.0001,.001,.01]:
            for eta in [.01,.1]:
                models[f'sgd_{alpha}_{eta}']=SGDLogistic(x[train],y[train],scaling,alpha,eta,seed)
        for name,model in models.items():
            scores=[]
            # Same delayed-label protocol at fixed 20-case validation batches.
            for start in range(0,len(val),20):
                batch=val[start:start+20]
                scores.extend(model.decision_function(x[batch]));model.partial_fit(x[batch],y[batch])
            m=metrics(y[val],np.array(scores));m['balanced_accuracy']=(m['sensitivity']+m['specificity'])/2
            records.append(dict(inner=inner,configuration=name,**m))
    averages={name:{k:float(np.mean([r[k] for r in records if r['configuration']==name])) for k in
                ['balanced_accuracy','sensitivity','specificity']} for name in models}
    # Fixed order resolves ties; no sensitivity/noninferiority guarantee is imposed.
    local_names=[f'local_{lr}_{k}' for lr,k in local_grid]
    local=max(local_names,key=lambda n:round(averages[n]['balanced_accuracy'],12))
    eclass=max([f'eclass_{floor}' for floor in [.03,.1,.25,.5,1.]],key=lambda n:round(averages[n]['balanced_accuracy'],12))
    sgd=max([name for name in models if name.startswith('sgd_')],key=lambda n:round(averages[n]['balanced_accuracy'],12))
    _,lr,active=local.split('_');floor=float(eclass.split('_')[1])
    _,alpha,eta=sgd.split('_')
    return dict(lr=float(lr),active=int(active),sigma_floor=floor,sgd_alpha=float(alpha),sgd_eta0=float(eta),means=averages),records,splits


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    file=ROOT/'data/dataset_1062.txt';a=np.loadtxt(file);x,y=a[:,1:],a[:,0].astype(int)
    protocol=dict(seeds=[17,29,43],batches=[50,100],repeats=3,initial=300,holdout=.2,
                  policies=POLICIES,alpha=.5,support=.1,max_length=3,max_rules=64,maxiter=500,
                  initialization='One initial train-only minmax scaler per stream, frozen for ALL methods. No clipping.',
                  tuning='Only initial300, 3-fold delayed-20 prequential validation; maximize balanced accuracy. Local SGD lr .01/.05/.1, active4/8. eClass floor .03/.1/.25/.5/1 (expanded after initial-only sensitivity check; prior narrow grid retained). Logistic SGD alpha .0001/.001/.01, eta0 .01/.1. Same frozen choice for both later batch sizes.',
                  evaluation='Predict ENTIRE new batch before any policy receives those labels; update all using exactly same batch; heldout only after update.',
                  counters='History rows passed to fit = distinct old raw rows supplied at each update, not measured disk reads. Objective row evaluations count old rows*nfev on cached features. Compressed index visits/span counted separately.',
                  sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                  limits=['Same1062 dataset repeatedly explored; not external confirmation.',
                          'No patient IDs, dates, independent cases or DASSD full text; none required to run simulation.',
                          'EClass zero-order author-draft formula implementation, NOT certified 2008 author implementation.',
                          'Local fixed freezes patterns; local dynamic reads compressed historical state and recomputes global selection.',
                          'No clinical noninferiority claim; memory is reachable persistent Python state, not peak RSS.'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    timings,predictions,holdouts,choices,splits,tuning_rows=[],[],[],[],[],[]
    for seed in protocol['seeds']:
        stream,test=train_test_split(np.arange(1062),test_size=.2,stratify=y,random_state=seed)
        np.random.default_rng(seed).shuffle(stream);initial=stream[:300]
        start=perf_counter();choice,records,inner=initial_tuning(x[initial],y[initial],seed,initial)
        choice.update(seed=seed,tuning_seconds=perf_counter()-start);choices.append(choice)
        tuning_rows.extend(dict(seed=seed,**r) for r in records)
        splits.append(dict(seed=seed,train_stream=stream.tolist(),holdout=test.tolist(),initial_inner=inner))
        scaling=(x[initial].min(0),np.maximum(np.ptp(x[initial],axis=0),1.))
        index,base=initial_fit(x[initial],y[initial],scaling)
        eclass=EClass0(*scaling,sigma_floor=choice['sigma_floor']).partial_fit(x[initial],y[initial])
        reference={}
        for batch_size in [50,100]:
            for repeat in range(3):
                models={'full_retrain':FullRetrain(index,base,x[initial],y[initial],scaling),
                        'local_dynamic':LocalPipeline(index,base,choice['lr'],choice['active'],True),
                        'local_fixed':LocalPipeline(index,base,choice['lr'],choice['active'],False),
                        'eclass0':EClassPipeline(eclass),'sgd_logistic':SGDLogistic(x[initial],y[initial],scaling,
                            choice['sgd_alpha'],choice['sgd_eta0'],seed),'frozen':Frozen(base)}
                for stage,start in enumerate(range(300,849,batch_size)):
                    end=min(start+batch_size,849);new=stream[start:end]
                    offset=(repeat+stage)%len(POLICIES);order=POLICIES[offset:]+POLICIES[:offset]
                    # All predictions occur before ANY update consumes this batch's labels.
                    staged={}
                    for policy in order:
                        t=perf_counter();score=models[policy].decision_function(x[new]);prediction_seconds=perf_counter()-t
                        staged[policy]=(score,prediction_seconds)
                    for policy in order:
                        model=models[policy];score,prediction_seconds=staged[policy]
                        if repeat==0:
                            predictions.extend(dict(seed=seed,batch_size=batch_size,stage=stage,policy=policy,
                                    row_index=int(i),label=int(y[i]),score=float(s),train_n_before=start) for i,s in zip(new,score))
                            reference[seed,batch_size,stage,policy]=score.copy()
                        else:np.testing.assert_allclose(score,reference[seed,batch_size,stage,policy],rtol=0,atol=1e-10)
                        t=perf_counter();stats=model.update(x[new],y[new]);update_seconds=perf_counter()-t
                        if policy not in ['full_retrain']:
                            assert stats['history_rows_passed_to_fit']==0 and stats['retained_raw_history_rows']==0
                        else:assert stats['history_rows_passed_to_fit']==start and len(model.history_y)==end
                        if policy=='local_dynamic':assert model.index.n==end
                        if policy=='eclass0':assert sum(s.n for s in model.model.classes.values())==end
                        timings.append(dict(seed=seed,batch_size=batch_size,repeat=repeat,stage=stage,policy=policy,
                            train_n_before=start,train_n_after=end,new_n=len(new),prediction_seconds=prediction_seconds,
                            update_seconds=update_seconds,state_bytes=deep_size(model),**stats))
                        if repeat==0:
                            holdouts.append(dict(seed=seed,batch_size=batch_size,stage=stage,train_n=end,policy=policy,
                                                 **metrics(y[test],model.decision_function(x[test]))))
                    # Same updated pattern counts for full and local dynamic pipelines.
                    assert models['full_retrain'].index.stats==models['local_dynamic'].index.stats
                print(f'Online seed={seed}, batch={batch_size}, repeat {repeat+1}/3 complete; '
                      f'lr={choice["lr"]}, active={choice["active"]}, eClass floor={choice["sigma_floor"]}',flush=True)
            for name,rows in [('timings',timings),('predictions',predictions),('holdouts',holdouts),('initial_tuning',tuning_rows)]:
                write_csv(OUT/(name+'.csv'),rows)
            (OUT/'choices.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
            (OUT/'splits.json').write_text(json.dumps(splits),encoding='utf-8')


if __name__=='__main__':main()
