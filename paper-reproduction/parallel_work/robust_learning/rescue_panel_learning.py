"""Four predeclared learners for the historical baseline-negative population."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.covariance import LedoitWolf
from sklearn.svm import SVC


METHODS = ('shrinkage_lda', 'categorical_bayes', 'case_neighbors', 'rbf_margin')


def lda_scores(x, y, test):
    pos = y == 1
    means = [x[~pos].mean(axis=0), x[pos].mean(axis=0)]
    centered = x - np.where(pos[:, None], means[1], means[0])
    covariance = LedoitWolf(assume_centered=True).fit(centered).covariance_
    if not np.any(covariance):
        return None
    direction = solve(covariance, means[1] - means[0], assume_a='pos')
    return (test - (means[0]+means[1])/2) @ direction + np.log(pos.sum()/(~pos).sum())


def bayes_scores(x, y, test):
    x, test = x.copy(), test.copy()
    x[:, -1] = np.minimum(4, np.floor(x[:, -1]*10))
    test[:, -1] = np.minimum(4, np.floor(test[:, -1]*10))
    counts = [(y == label).sum() for label in (-1,1)]
    odds = np.full(len(test), np.log(counts[1]/counts[0]))
    for j in range(x.shape[1]):
        values, encoded = np.unique(x[:, j], return_inverse=True)
        indices = np.searchsorted(values,test[:,j])
        known = (indices < len(values)) & (values[np.minimum(indices,len(values)-1)] == test[:,j])
        mapped = np.where(known,indices,len(values))
        for label, n in zip((-1,1), counts, strict=True):
            observed = np.bincount(encoded[y==label],minlength=len(values)+1)
            odds += label*(np.log(observed[mapped]+1)-np.log(n+len(values)+1))
    return odds


def model_scores(method, x, y, test):
    if method == 'shrinkage_lda':
        return lda_scores(x,y,test)
    if method == 'categorical_bayes':
        return bayes_scores(x,y,test)
    if method == 'case_neighbors':
        if len(y) < 5:
            return None
        nearby = np.argsort(cdist(test,x,'sqeuclidean'),axis=1,kind='stable')[:,:5]
        return np.sum(y[nearby]==1,axis=1)-2.5
    if method == 'rbf_margin':
        model = SVC(C=1.,gamma='scale',kernel='rbf',max_iter=10_000).fit(x,y)
        if model.fit_status_:
            return None
        return model.decision_function(test)
    raise ValueError(method)


def predict(history, labels, current):
    results = {name: current[:, -1].copy() for name in METHODS}
    h_negative, c_negative = history[:, -1] < 0, current[:, -1] < 0
    y = labels[h_negative]
    status = dict(screened=int(c_negative.sum()), historical_negatives=len(y),
                  historical_misses=int(np.sum(y==1)), fitted=[])
    if not c_negative.any() or len(np.unique(y)) < 2:
        return results,status
    x,test = history[h_negative].copy(),current[c_negative].copy()
    x[:,-1],test[:,-1] = expit(x[:,-1]),expit(test[:,-1])
    for method in METHODS:
        score = model_scores(method,x,y,test)
        if score is None:
            continue
        assert np.isfinite(score).all()
        status['fitted'].append(method)
        results[method][c_negative] = np.where(score>=0,score,current[c_negative,-1])
    return results,status
