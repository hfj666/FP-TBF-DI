from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


@dataclass
class Rule:
    columns: tuple
    values: np.ndarray
    label: int
    confidence: float
    count: int


class SparseSugeno:
    """ANFIS-inspired zero-order sparse Sugeno rules with learned widths/consequents.

    Antecedent centers/subspaces are mined rules. Joint BCE optimization trains
    Gaussian premise widths and constant consequents. Not the proposal's full
    five-layer ANFIS or hybrid least-squares training algorithm.
    """
    def __init__(self, rules, maxiter=200):
        if not rules:
            raise ValueError('No rules available.')
        self.rules, self.maxiter = rules, maxiter

    def _distances(self, x):
        z = (x-self.minimum)/self.span
        return np.column_stack([np.mean((z[:, list(r.columns)]-r.values)**2, axis=1)
                                for r in self.rules])

    def fit(self, x, y, initial_model=None, scaling=None):
        # Rules enter with RAW categorical score centers; scale on current train only.
        raw_rules = self.rules
        self.minimum = x.min(0) if scaling is None else np.asarray(scaling[0]).copy()
        self.span = np.maximum(x.max(0)-self.minimum, 1.) if scaling is None else np.asarray(scaling[1]).copy()
        self.rules = [Rule(r.columns, (r.values-self.minimum[list(r.columns)])/self.span[list(r.columns)],
                           r.label, r.confidence, r.count) for r in self.rules]
        d2 = self._distances(x)
        target = (y == 1).astype(float)
        k = len(self.rules)
        logprior = np.log([r.confidence for r in self.rules])
        def objective(theta):
            logwidth, consequents = theta[:k], theta[k:]
            a = d2*np.exp(-2*logwidth)
            logfire = -.5*a + logprior
            weights = np.exp(logfire-logsumexp(logfire, axis=1)[:, None])
            score = weights @ consequents
            err = (expit(score)-target)/len(target)
            loss = np.mean(np.logaddexp(0, score)-target*score)+.001*np.mean(consequents**2)
            grad_c = weights.T@err + .002*consequents/k
            grad_w = np.sum(err[:, None]*weights*(consequents-score[:, None])*a, axis=0)
            return loss, np.r_[grad_w, grad_c]
        initial = np.r_[np.full(k, np.log(.25)), [2*r.label for r in self.rules]]
        transferred = 0
        if initial_model is not None:
            old = initial_model
            old_rules = {(r.columns, tuple(np.round(r.values*old.span[list(r.columns)]+old.minimum[list(r.columns)],9))): i
                         for i,r in enumerate(old.rules)}
            for i,r in enumerate(raw_rules):
                previous=old_rules.get((r.columns,tuple(np.round(r.values,9))))
                if previous is not None:
                    initial[k+i]=old.theta[len(old.rules)+previous]
                    if np.array_equal(self.span[list(r.columns)],old.span[list(r.columns)]):
                        initial[i]=old.theta[previous]
                    transferred+=1
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          bounds=[(np.log(.03), np.log(2.))]*k+[(-10., 10.)]*k,
                          options={'maxiter': self.maxiter})
        self.theta = result.x
        self.optimization = {'success': bool(result.success), 'message': str(result.message),
                             'iterations': int(result.nit), 'loss': float(result.fun),
                             'transferred_rules': transferred,
                             'function_evaluations': int(result.nfev)}
        return self

    def firing_weights(self, x):
        k = len(self.rules)
        logfire = -.5*self._distances(x)*np.exp(-2*self.theta[:k])+np.log([r.confidence for r in self.rules])
        return np.exp(logfire-logsumexp(logfire, axis=1)[:, None])

    def decision_function(self, x):
        return self.firing_weights(x)@self.theta[len(self.rules):]
