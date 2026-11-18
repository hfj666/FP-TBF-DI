"""UR Eq7 adaptation to existing sparse zero-order BCE Sugeno."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from .models import SparseSugeno,Rule

def objective(theta,d2,target,confidence,weight):
    n,k=d2.shape;lw,c=theta[:k],theta[k:];a=d2*np.exp(-2*lw)
    logits=-.5*a+np.log(confidence);w=np.exp(logits-logsumexp(logits,axis=1)[:,None]);h=w@c
    err=(expit(h)-target)/n;mean=w.mean(0);delta=mean-.5
    # tau=1/C=.5 per source Eq7; replacing by1/K changes only a constant.
    ur=float(np.sum(delta**2));loss=np.mean(np.logaddexp(0,h)-target*h)+.001*np.mean(c*c)+weight*ur
    q=2*delta/n
    ur_logits=w*(q-np.sum(w*q,axis=1)[:,None])
    gw=np.sum(err[:,None]*w*(c-h[:,None])*a+weight*ur_logits*a,axis=0)
    gc=w.T@err+.002*c/k
    return float(loss),np.r_[gw,gc],dict(ur=ur,concentration=float(mean@mean),max_mass=float(mean.max()),effective_rules=float(1/(mean@mean)))

class UniformSugeno(SparseSugeno):
    def fit(self,x,y,scaling,ur_weight=.1):
        self.minimum=np.array(scaling[0]);self.span=np.array(scaling[1]);self.ur_weight=ur_weight
        self.rules=[Rule(r.columns,(r.values-self.minimum[list(r.columns)])/self.span[list(r.columns)],r.label,r.confidence,r.count) for r in self.rules]
        d2=self._distances(x);target=(y==1).astype(float);confidence=np.array([r.confidence for r in self.rules]);k=len(self.rules)
        theta=np.r_[np.full(k,np.log(.25)),[2*r.label for r in self.rules]]
        fun=lambda t:objective(t,d2,target,confidence,ur_weight)[:2]
        result=minimize(fun,theta,jac=True,method='L-BFGS-B',bounds=[(np.log(.03),np.log(2))]*k+[(-10.,10.)]*k,options={'maxiter':500})
        self.theta=result.x;_,_,diagnostic=objective(result.x,d2,target,confidence,ur_weight)
        self.optimization=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),loss=float(result.fun),function_evaluations=int(result.nfev),**diagnostic)
        assert np.isfinite(result.x).all()
        return self
