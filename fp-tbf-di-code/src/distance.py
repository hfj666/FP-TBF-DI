"""Distance correction used by branch B."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .models import Rule, SparseSugeno


def header_table(tree):
    return {item:(sum(n.count for n in nodes),sum(n.positive for n in nodes))
            for item,nodes in tree.header.items()}


class DistanceSugeno(SparseSugeno):
    def __init__(self,rules,mode,table=None):
        super().__init__(rules,maxiter=500)
        self.raw_rules=[Rule(r.columns,np.array(r.values).copy(),r.label,r.confidence,r.count) for r in rules]
        self.mode=mode
        self.table={} if table is None else dict(table)

    def _distances(self,x):
        if self.mode=='numeric_max':
            z=(x-self.minimum)/self.span
            return np.column_stack([np.max((z[:,list(r.columns)]-r.values)**2,axis=1) for r in self.rules])
        distances=[]
        for r in self.raw_rules:
            pieces=[]
            for j,v in zip(r.columns,r.values,strict=True):
                vv=int(v)
                if self.mode=='overlap_mean': d=(x[:,j]!=vv).astype(float)
                elif self.mode=='hvdm_mean':
                    nc,np_=self.table[j,vv];center=np_/nc
                    d=np.array([(self.table[j,int(a)][1]/self.table[j,int(a)][0]-center)**2
                                if (j,int(a)) in self.table else 1. for a in x[:,j]])
                else: raise ValueError(self.mode)
                pieces.append(d)
            distances.append(np.mean(pieces,axis=0))
        return np.column_stack(distances)

    def snapshot(self):
        def records(rr):return [dict(columns=list(r.columns),values=r.values.tolist(),label=int(r.label),confidence=float(r.confidence),count=int(r.count)) for r in rr]
        return dict(mode=self.mode,raw_rules=records(self.raw_rules),rules=records(self.rules),
            table=[[j,v,n,p] for (j,v),(n,p) in sorted(self.table.items())],
            minimum=self.minimum.tolist(),span=self.span.tolist(),theta=self.theta.tolist(),optimization=self.optimization)


def offset_fit(base, feature, y):
    n = len(y)

    def objective(a):
        score = base + feature*a[0]
        loss = np.mean(np.logaddexp(0,score)-y*score) + .5*a[0]**2/n
        grad = np.mean((expit(score)-y)*feature) + a[0]/n
        return loss,np.array([grad])

    return minimize(objective,[0.],jac=True,method='L-BFGS-B',options={'maxiter':500})
