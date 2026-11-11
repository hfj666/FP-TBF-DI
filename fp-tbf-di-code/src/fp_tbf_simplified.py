"""FP-TBF with C = base -> feature-deletion stability -> stable interactions.

This path never accepts or stores historical prediction scores/mistake labels.
It still retains the revealed training prefix for cumulative fitting.
"""
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from .distance import DistanceSugeno, header_table, offset_fit
from .fptree import FPTree
from .models import SparseSugeno
from .projected_selection import select
from .reference_comparators import CumulativeLogistic
from .selection import CoveragePool


def _fit_score(train,y,query):
    return LogisticRegression(C=1,solver='liblinear',max_iter=1000).fit(train,y).decision_function(query)


def _interactions(train,query):
    expansion=PolynomialFeatures(2,interaction_only=True,include_bias=False).fit(train)
    scale=np.sqrt(2/17)
    return [np.column_stack((block,expansion.transform(block)[:,18:]*scale)) for block in (train,query)]


def predict_c(x,y,query,base):
    """Only the revealed prefix, query features, and precomputed base are inputs."""
    started=perf_counter()
    reference=np.array(base,dtype=float,copy=True)
    negative=set(map(tuple,x[y == -1]))
    allowed=np.array([tuple(row) not in negative for row in query])
    record=dict(D_fits=0,I_fits=0,D_upgrades=0,I_upgrades=0)
    eligible=np.flatnonzero((reference<0)&allowed)
    tick=perf_counter()
    if len(eligible):
        encoder=OneHotEncoder(handle_unknown='ignore').fit(x)
        train,test=encoder.transform(x),encoder.transform(query[eligible])
        full=_fit_score(train,y,test)
        stable=full>=0
        borders=np.cumsum([0,*[len(v) for v in encoder.categories_]])
        for first,last in zip(borders[:-1],borders[1:],strict=True):
            keep=(np.arange(train.shape[1])<first)|(np.arange(train.shape[1])>=last)
            stable &= _fit_score(train[:,keep],y,test[:,keep])>=0
        reference[eligible[stable]]=full[stable]
        record.update(D_fits=19,D_upgrades=int(stable.sum()))
    record['D_seconds']=perf_counter()-tick
    after_d=reference.copy()
    eligible=np.flatnonzero((reference<0)&allowed)
    tick=perf_counter()
    if len(eligible):
        scaler=StandardScaler().fit(x)
        train,test=_interactions(scaler.transform(x),scaler.transform(query[eligible]))
        full=_fit_score(train,y,test)
        stable=full>=0
        rng=np.random.default_rng(1105)
        groups=[np.flatnonzero(y==label) for label in (-1,1)]
        record['I_fits']=1
        for _ in range(32):
            if not stable.any():break
            sample=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups])
            stable[stable]=_fit_score(train[sample],y[sample],test[stable])>=0
            record['I_fits']+=1
        reference[eligible[stable]]=full[stable]
        record['I_upgrades']=int(stable.sum())
    record.update(I_seconds=perf_counter()-tick,seconds=perf_counter()-started)
    assert np.isfinite(reference).all()
    np.testing.assert_array_equal(reference[base>=0],np.asarray(base)[base>=0])
    return reference,after_d,record


class SimplifiedFPTBF:
    """Cold cumulative fits with the same A/B/base definitions as the full model."""
    def fit(self,x,y,tree=None,*,timing=None):
        self.x=np.array(x,copy=True);self.y=np.array(y,copy=True)
        if len(x)<300:raise ValueError('At least 300 initially labelled rows required.')
        self.scale=x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.)
        tree=FPTree().update(x,y) if tree is None else tree
        if tree.n!=len(x):raise ValueError('Tree must match the revealed prefix.')
        mining_start=perf_counter()
        patterns=tree.mine(.1,3)
        selection_start=perf_counter()
        pool=CoveragePool(patterns,x,y)
        if any(len(v)<32 for v in pool.pools.values()):raise ValueError('32 eligible rules per class required.')
        rules,selection=select(pool,x,y,self.scale,joint=True)
        fitting_start=perf_counter()
        self.a=SparseSugeno(rules,maxiter=2000).fit(x,y,scaling=self.scale)
        coverage_start=perf_counter()
        rules=pool.select(.5,64)
        coverage_end=perf_counter()
        self.base=SparseSugeno(rules,maxiter=500).fit(x,y,scaling=self.scale)
        self.h=DistanceSugeno(rules,'hvdm_mean',header_table(tree)).fit(x,y,scaling=self.scale)
        self.logistic=CumulativeLogistic(x,y,self.scale)
        beta=offset_fit(self.logistic.decision_function(x),self.h.decision_function(x)/10,y==1)
        if not beta.success:raise RuntimeError(str(beta.message))
        self.beta=float(beta.x[0])
        self.optimization=dict(A=self.a.optimization,base=self.base.optimization,
                               H=self.h.optimization,selector_stop=selection['selector_stop'])
        if timing is not None:
            coverage_time=coverage_end-coverage_start
            timing.update(mining=selection_start-mining_start,
                          selection=fitting_start-selection_start+coverage_time,
                          fitting=perf_counter()-fitting_start-coverage_time)
        return self

    def core_scores(self,query):
        base=self.base.decision_function(query)
        a=self.a.decision_function(query)
        logistic=self.logistic.decision_function(query)
        b=logistic+self.h.decision_function(query)/10*self.beta
        return dict(base=base,logistic=logistic,A=a,B=b)

    def decision_scores(self,query):
        scores=self.core_scores(query)
        c,_,details=predict_c(self.x,self.y,query,scores['base'])
        self.last_c_details=details
        return np.column_stack((scores['base'],scores['logistic'],scores['A'],scores['B'],c,
                                np.mean([scores['A'],scores['B'],c],axis=0)))

    def decision_function(self,query):
        return self.decision_scores(query)[:,-1]
