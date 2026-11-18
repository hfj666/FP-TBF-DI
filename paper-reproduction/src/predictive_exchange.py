"""Exact empirical-risk exchanges for normalized Sugeno inference.

Not a novelty claim. Monotonicity applies to fixed parameters on current batch
BEFORE final SGD only; it is not a generalization or post-SGD guarantee.
"""
import copy
import numpy as np
from .budgeted_rules import identity, select_budgeted


def balanced_weights(y):
    y=np.asarray(y)
    if not all(np.any(y==c) for c in (-1,1)):
        return None
    return np.array([.5/np.sum(y==c) for c in y])


def exchange_scores(numerator, denominator, old_fire, old_c, new_fire, new_c):
    """Exact remove/add normalization; all inputs share an activation scale."""
    den=denominator-old_fire+new_fire
    if np.any(den<=0):
        raise FloatingPointError('nonpositive activation denominator')
    return (numerator-old_fire*old_c+new_fire*new_c)/den


def select_predictive(pool, previous, model, x, y, protected=False, max_swaps=4, max_rules=32):
    rules,keys,stats=select_budgeted(pool,previous,max_rules=max_rules,max_swaps=0)
    entries=[r for c in (-1,1) for r in pool.pools[c]]
    all_keys=[identity(r) for r in entries];lookup={k:i for i,k in enumerate(all_keys)}
    selected=[lookup[k] for k in keys]
    expanded=copy.deepcopy(model);expanded.replace_rules([r[1] for r in entries])
    k=len(entries);z=(x-expanded.minimum)/expanded.span
    distances=np.column_stack([np.mean((z[:,list(r.columns)]-r.values)**2,axis=1) for r in expanded.rules])
    logs=-.5*distances*np.exp(-2*expanded.theta[:k])+np.log([r.confidence for r in expanded.rules])
    fire=np.exp(logs-logs.max(axis=1)[:,None]);consequents=expanded.theta[k:]
    labels=np.array([r[1].label for r in entries]);w=balanced_weights(y)
    n=fire[:,selected]@consequents[selected];d=fire[:,selected].sum(axis=1)
    if np.any(d<=0):raise FloatingPointError('activation underflow')
    initial=n/d;trace=[];evaluated=0
    if w is not None:
        for _ in range(max_swaps):
            score=n/d;loss=float(w@np.logaddexp(0.,-y*score));best_loss=loss;best=None
            correct=np.where(score>=0,1,-1)==y
            available=np.ones(k,dtype=bool);available[selected]=False
            for pos,outgoing in enumerate(selected):
                incoming=np.flatnonzero(available & (labels==labels[outgoing]))
                if not len(incoming):continue
                proposals=exchange_scores(n[:,None],d[:,None],fire[:,outgoing,None],consequents[outgoing],fire[:,incoming],consequents[incoming])
                losses=w@np.logaddexp(0.,-y[:,None]*proposals);evaluated+=len(incoming)
                if protected:
                    harm=np.any((np.where(proposals>=0,1,-1)!=y[:,None]) & correct[:,None],axis=0)
                    losses[harm]=np.inf
                j=int(np.argmin(losses))
                if losses[j]<best_loss-1e-12:
                    best_loss=float(losses[j]);best=(pos,outgoing,int(incoming[j]))
            if best is None:break
            pos,old,new=best
            n=n-fire[:,old]*consequents[old]+fire[:,new]*consequents[new]
            d=d-fire[:,old]+fire[:,new];selected[pos]=new
            trace.append(loss-best_loss)
    final=n/d
    stats.update(discretionary_swaps=len(trace),risk_pairs=evaluated,empirical_loss_drop=sum(trace),
                 pre_sgd_new_errors=int(np.sum((np.where(initial>=0,1,-1)==y)&(np.where(final>=0,1,-1)!=y))))
    return [entries[i][1] for i in selected],[all_keys[i] for i in selected],stats


class PredictivePipeline:
    def __init__(self,index,model,keys,policy,max_rules=32):
        self.index=copy.deepcopy(index);self.model=copy.deepcopy(model);self.keys=keys.copy();self.policy=policy
        self.max_rules=max_rules

    def update(self,x,y):
        self.index.update(x,y);pool=self.index.pool(self.index.mine(.1));before=set(self.keys)
        if self.policy in ('prediction','protected'):
            rules,keys,stats=select_predictive(pool,self.keys,self.model,x,y,protected=self.policy=='protected',max_rules=self.max_rules)
        else:
            rules,keys,stats=select_budgeted(pool,self.keys,max_rules=self.max_rules,
                max_swaps=4 if self.policy=='coverage' else 0,global_rebuild=self.policy=='global')
        self.model.replace_rules(rules)
        pre=self.model.decision_function(x);self.model.partial_fit(x,y);post=self.model.decision_function(x)
        self.keys=keys;after=set(keys)
        return dict(new=len(after-before),removed=len(before-after),rules=len(after),swaps=stats['discretionary_swaps'],
            risk_pairs=stats.get('risk_pairs',0),empirical_loss_drop=stats.get('empirical_loss_drop',float('nan')),
            pre_sgd_new_errors=stats.get('pre_sgd_new_errors',float('nan')),
            sgd_new_errors=int(np.sum((np.where(pre>=0,1,-1)==y)&(np.where(post>=0,1,-1)!=y))),
            forced_added=stats['forced_added'],forced_removed=stats['forced_removed'])
