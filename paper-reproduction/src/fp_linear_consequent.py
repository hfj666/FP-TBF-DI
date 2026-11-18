"""Fixed Gaussian gates, convex constant/shared/local affine BCE consequents."""
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def design_matrix(weights,z,mode):
    n,k=weights.shape
    if mode=='constant_refit':return weights.copy(),np.full(k,.001/k)
    if mode=='shared_linear':return np.column_stack((weights,z)),np.r_[np.full(k,.001/k),np.full(z.shape[1],.001)]
    if mode=='local_linear':
        features=np.column_stack((np.ones(n),z))
        return (weights[:,:,None]*features[:,None,:]).reshape(n,-1),np.full(k*features.shape[1],.001/k)
    raise ValueError('Unknown consequent mode')

def objective(beta,design,target,penalty):
    h=design@beta
    return float(np.mean(np.logaddexp(0,h)-target*h)+np.sum(penalty*beta**2)),design.T@(expit(h)-target)/len(target)+2*penalty*beta

class LinearConsequent:
    def __init__(self,base,mode):self.base=copy.deepcopy(base);self.mode=mode
    def _design(self,x):
        z=(x-self.base.minimum)/self.base.span
        return design_matrix(self.base.firing_weights(x),z,self.mode)
    def fit(self,x,y):
        design,penalty=self._design(x);k=len(self.base.rules);c=self.base.theta[k:]
        initial=np.zeros(design.shape[1])
        if self.mode=='local_linear':initial.reshape(k,-1)[:,0]=c
        else:initial[:k]=c
        target=(y==1).astype(float);before=objective(initial,design,target,penalty)[0]
        result=minimize(lambda b:objective(b,design,target,penalty),initial,jac=True,method='L-BFGS-B',bounds=[(-10.,10.)]*len(initial),options={'maxiter':500})
        assert np.isfinite(result.x).all() and result.fun<=before+1e-10
        self.beta=result.x
        self.optimization=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),training_before=before,training_after=float(result.fun),parameters=len(initial))
        return self
    def decision_function(self,x):return self._design(x)[0]@self.beta
