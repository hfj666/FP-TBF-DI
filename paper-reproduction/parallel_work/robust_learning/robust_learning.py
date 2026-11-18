"""Isolated literature-component objectives; shared source remains read-only."""
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from src.models import SparseSugeno


def mask_bank(n, k, seed, count=16, keep=.5):
    m = np.random.default_rng(seed).random((count, n, k)) <= keep
    m[np.where(~m.any(axis=2))] = True
    return m


def objective(theta, d2, target, confidence, mode, masks=None):
    n, k = d2.shape
    a = d2 * np.exp(-2 * theta[:k])
    logits = -.5 * a + np.log(confidence)
    c = theta[k:]
    if mode == 'droprule':
        logits = np.where(masks, logits[None, :, :], -np.inf)
        w = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
        h = np.sum(w * c, axis=2)
        err = (expit(h) - target) / (n * len(masks))
        loss = np.mean(np.logaddexp(0, h) - target * h)
        gc = np.sum(w * err[:, :, None], axis=(0, 1))
        gw = np.sum(err[:, :, None] * w * (c-h[:, :, None]) * a, axis=(0, 1))
    else:
        w = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        h = w @ c
        p = expit(h)
        err = p - target
        if mode == 'baseline':
            losses = np.logaddexp(0, h) - target * h
        elif mode == 'gce':
            logpy = -np.logaddexp(0, -(2*target-1)*h)
            losses = -np.expm1(.7*logpy)/.7
            err = err*np.exp(.7*logpy)
        elif mode == 'focal':
            logpy = -np.logaddexp(0, -(2*target-1)*h)
            py = np.exp(logpy)
            losses = -(1-py)**2*logpy
            err = err*((1-py)**2-2*py*(1-py)*logpy)
        else:
            raise ValueError(mode)
        err = err/n
        loss = np.mean(losses)
        gc = w.T @ err
        gw = np.sum(err[:, None]*w*(c-h[:, None])*a, axis=0)
    return float(loss+.001*np.mean(c*c)), np.r_[gw, gc+.002*c/k]


def fit_component(base, x, y, mode, bank_seed):
    m = SparseSugeno(base.rules, maxiter=500)
    m.minimum, m.span = base.minimum.copy(), base.span.copy()
    d2 = m._distances(x)
    target = (y == 1).astype(float)
    k = len(m.rules)
    masks = mask_bank(len(x), k, bank_seed) if mode == 'droprule' else None
    initial = np.r_[np.full(k, np.log(.25)), [2*r.label for r in m.rules]]
    f = lambda t: objective(t, d2, target, np.array([r.confidence for r in m.rules]), mode, masks)
    result = minimize(f, initial, jac=True, method='L-BFGS-B',
                      bounds=[(np.log(.03), np.log(2.))]*k+[(-10.,10.)]*k,
                      options={'maxiter':500})
    m.theta = result.x
    m.optimization = dict(success=bool(result.success), message=str(result.message),
                          iterations=int(result.nit), function_evaluations=int(result.nfev),
                          loss=float(result.fun), bank_seed=bank_seed, parameter_count=2*k)
    if not np.isfinite(m.theta).all():
        raise ValueError('Nonfinite fit')
    return m
