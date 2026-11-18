"""Exact complete-transaction grouping from the unpruned cumulative FP tree."""
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from .models import SparseSugeno,Rule


def tree_groups(tree,dimensions):
    records=[];stack=[(tree.root,())];visited=0
    while stack:
        node,path=stack.pop();visited+=1
        if node is not tree.root and not node.children:
            if tuple(j for j,v in path)!=tuple(range(dimensions)):
                raise ValueError('Only complete fixed-dimension base-tree paths are supported')
            records.append((tuple(v for j,v in path),node.count,node.positive))
        else:
            stack.extend((child,path+(item,)) for item,child in node.children.items())
    records.sort()
    if not records or sum(r[1] for r in records)!=tree.n:
        raise ValueError('Incomplete paths or empty tree')
    x=np.array([r[0] for r in records],dtype=float)
    n=np.array([r[1] for r in records],dtype=float);positive=np.array([r[2] for r in records],dtype=float)
    if np.any((positive<0)|(positive>n)):raise ValueError('Invalid class counts')
    return x,n,positive,visited


def prepare(rules,x,n,positive,scaling):
    model=SparseSugeno(rules,maxiter=500)
    model.minimum=np.asarray(scaling[0]).copy();model.span=np.asarray(scaling[1]).copy()
    model.rules=[Rule(r.columns,(r.values-model.minimum[list(r.columns)])/model.span[list(r.columns)],
                      r.label,r.confidence,r.count) for r in rules]
    d2=model._distances(x);k=len(rules);prior=np.log([r.confidence for r in rules]);total=float(np.sum(n))
    def objective(theta):
        a=d2*np.exp(-2*theta[:k]);c=theta[k:]
        logs=-.5*a+prior;w=np.exp(logs-logsumexp(logs,axis=1)[:,None]);score=w@c
        error=(n*expit(score)-positive)/total
        value=np.sum(n*np.logaddexp(0,score)-positive*score)/total+.001*np.mean(c*c)
        gc=w.T@error+.002*c/k
        gw=np.sum(error[:,None]*w*(c-score[:,None])*a,axis=0)
        return value,np.r_[gw,gc]
    return model,objective


def fit_weighted(rules,x,n,positive,scaling):
    model,objective=prepare(rules,x,n,positive,scaling);k=len(rules)
    initial=np.r_[np.full(k,np.log(.25)),[2*r.label for r in rules]]
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',
        bounds=[(np.log(.03),np.log(2.))]*k+[(-10.,10.)]*k,options={'maxiter':500})
    if not np.isfinite(result.x).all():raise ValueError('Nonfinite fit')
    model.theta=result.x
    model.optimization=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
        function_evaluations=int(result.nfev),loss=float(result.fun),objective_rows=len(x)*int(result.nfev))
    return model
