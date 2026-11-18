"""Curvature-matched radial Cauchy shape; mean distance and head unchanged."""
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from src.models import Rule,SparseSugeno
from activation import DistanceSugeno

def objective(theta,d2,target,confidence):
    k=len(confidence);lw,c=theta[:k],theta[k:]
    a=.5*d2*np.exp(-2*lw)
    lf=-np.log1p(a)+np.log(confidence);w=np.exp(lf-logsumexp(lf,axis=1,keepdims=True));h=w@c
    err=(expit(h)-target)/len(target)
    loss=np.mean(np.logaddexp(0,h)-target*h)+.001*np.mean(c*c)
    gw=np.sum(err[:,None]*w*(c-h[:,None])*(2*a/(1+a)),axis=0)
    gc=w.T@err+.002*c/k
    return float(loss),np.r_[gw,gc]

class CauchySugeno(DistanceSugeno):
    def _distances(self,x):return SparseSugeno._distances(self,x)
    def fit(self,x,y,initial_model=None,scaling=None):
        assert initial_model is None and scaling is not None
        self.minimum=np.asarray(scaling[0]).copy();self.span=np.asarray(scaling[1]).copy()
        self.rules=[Rule(r.columns,(r.values-self.minimum[list(r.columns)])/self.span[list(r.columns)],r.label,r.confidence,r.count) for r in self.raw_rules]
        k=len(self.rules);d2=self._distances(x);target=(y==1).astype(float);q=np.array([r.confidence for r in self.rules])
        initial=np.r_[np.full(k,np.log(.25)),[2*r.label for r in self.rules]]
        result=minimize(lambda theta:objective(theta,d2,target,q),initial,jac=True,method='L-BFGS-B',bounds=[(np.log(.03),np.log(2.))]*k+[(-10.,10.)]*k,options={'maxiter':500})
        self.theta=result.x;self.optimization=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),loss=float(result.fun),transferred_rules=0,function_evaluations=int(result.nfev));assert np.isfinite(result.x).all()
        return self
    def firing_weights(self,x):
        k=len(self.rules);a=.5*self._distances(x)*np.exp(-2*self.theta[:k]);lf=-np.log1p(a)+np.log([r.confidence for r in self.rules])
        return np.exp(lf-logsumexp(lf,axis=1,keepdims=True))
